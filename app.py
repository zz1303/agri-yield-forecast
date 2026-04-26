import streamlit as st
import pandas as pd
import numpy as np
import os
import joblib
from src.preprocess import DataPreprocessor
import matplotlib.pyplot as plt
import plotly.express as px
import shap
from src.predictor import YieldPredictor  

@st.cache_data
def get_shap_values(_model, X_data, sample_size=200):
    """
    计算 SHAP 值（带缓存和采样）
    :param sample_size: 只抽取 200 条数据进行计算，足够画出完美的图表，且速度极快
    """
    # 如果数据量大于 sample_size，进行随机采样
    if len(X_data) > sample_size:
        X_sample = X_data.sample(n=sample_size, random_state=42)
    else:
        X_sample = X_data
        
    # 使用 TreeExplainer (针对 XGBoost 最快)
    explainer = shap.TreeExplainer(_model)
    # check_additivity=False 可以防止 XGBoost 版本兼容性问题带来的报错
    shap_values = explainer.shap_values(X_sample, check_additivity=False)
    
    return explainer, X_sample, shap_values

# 保持你原来的 X_train 加载
X_train = joblib.load(os.path.join(os.path.dirname(__file__), "models", "X_train.pkl"))

# 添加动态选项获取
@st.cache_data
def load_crop_data():
    """
    加载原始数据并返回DataFrame
    """
    data_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "example_data.csv")
    df = pd.read_csv(data_path)
    return df

# 从数据中动态获取选项
df = load_crop_data()
crop_options = sorted(df['Crop'].unique())
region_options = sorted(df['Region'].unique())
soil_type_options = sorted(df['Soil_Type'].unique())
weather_cond_options = sorted(df['Weather_Condition'].unique())

# 获取数值范围
rainfall_min, rainfall_max = df['Rainfall_mm'].min(), df['Rainfall_mm'].max()
temp_min, temp_max = df['Temperature_Celsius'].min(), df['Temperature_Celsius'].max()

# 应用主入口
COLUMN_CONFIG = {
    "Crop": "Crop",
    "Region": "Region",
    "Soil_Type": "Soil_Type",
    "Temperature": "Temperature_Celsius",
    "Rainfall": "Rainfall_mm",
    "Fertilizer_Used": "Fertilizer_Used",
    "Irrigation_Used": "Irrigation_Used",
    "Weather_Condition": "Weather_Condition",
    "Yield": "Yield_tons_per_hectare"
}

# 初始化预测器
MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "yield_model.pkl")
FEATURE_PATH = os.path.join(os.path.dirname(__file__), "models", "feature_columns.pkl")
predictor = YieldPredictor(model_path=MODEL_PATH, feature_path=FEATURE_PATH)

# 数据处理器
processor = DataPreprocessor(config=COLUMN_CONFIG)

# 主界面
st.set_page_config(page_title="农业单产预测与辅助决策", layout="wide")
st.title("农业单产预测与辅助决策系统 🌾")

# Sidebar 设置导航
menu = st.sidebar.radio(
    "导航栏",
    options=["数据概览", "单产预测", "作物推荐", "管理方案推荐", "模型可解释性", "敏感性分析","模型测试与评估"]
)
# 根据导航选择功能
if menu == "数据概览":
    st.header("数据概览 📊")
    uploaded_file = st.file_uploader("上传 CSV 数据集", type=["csv"])
    if uploaded_file:
        # 读取数据
        data = pd.read_csv(uploaded_file)

        # 展示数据基本信息
        st.write("### 数据预览：", data.head())
        st.write(f"数据集包含 {data.shape[0]} 行，{data.shape[1]} 列")
        
        # 缺失值分析 & 描述性统计
        st.write("### 数据缺失值：")
        st.write(data.isnull().sum())
        st.write("### 数据描述性统计：")
        st.write(data.describe())

        # 数据可视化
        with st.expander("点击展开数据可视化部分"):
            col1, col2 = st.columns(2)
            with col1:
                fig1 = px.histogram(data, x="Crop", y="Yield_tons_per_hectare", color="Crop",
                                    title="作物产量分布")
                st.plotly_chart(fig1, use_container_width=True)
            with col2:
                fig2 = px.scatter(data, x="Temperature_Celsius", y="Yield_tons_per_hectare", color="Crop",
                                  title="温度与单产关系")
                st.plotly_chart(fig2, use_container_width=True)

            col3, col4 = st.columns(2)
            with col3:
                fig3 = px.scatter(data, x="Rainfall_mm", y="Yield_tons_per_hectare", color="Crop",
                                  title="降雨量与单产关系")
                st.plotly_chart(fig3, use_container_width=True)
            with col4:
                fig4 = px.box(data, x="Soil_Type", y="Yield_tons_per_hectare",
                              title="不同土壤类型的平均产量")
                st.plotly_chart(fig4, use_container_width=True)

elif menu == "单产预测":
    st.header("单产预测 🌾")

    # 输入表单 - 使用动态选项
    st.write("### 请输入条件：")
    crop = st.selectbox("作物 (Crop)", crop_options)
    region = st.selectbox("区域 (Region)", region_options)
    soil_type = st.selectbox("土壤 (Soil_Type)", soil_type_options)
    temperature = st.slider("温度 (Temperature)", min_value=temp_min, max_value=temp_max, step=0.5)
    rainfall = st.slider("降雨量 (Rainfall)", min_value=rainfall_min, max_value=rainfall_max, step=10.0)
    fertilizer_used = st.selectbox("是否施肥 (Fertilizer_Used)", [0, 1])
    irrigation_used = st.selectbox("是否灌溉 (Irrigation_Used)", [0, 1])
    weather_cond = st.selectbox("天气状况 (Weather_Condition)", weather_cond_options)

    # 构造输入
    user_input = {
        "Crop": crop,
        "Region": region,
        "Soil_Type": soil_type,
        "Temperature_Celsius": temperature,
        "Rainfall_mm": rainfall,
        "Fertilizer_Used": fertilizer_used,
        "Irrigation_Used": irrigation_used,
        "Weather_Condition": weather_cond
    }

    # 模型预测
    if st.button("预测单产"):
        prediction = predictor.predict_single(user_input)
        st.success(f"预测单产：{prediction:.2f} 吨/公顷")

elif menu == "作物推荐":
    st.header("作物推荐 🌾")

    # 用户输入地块条件 - 使用动态选项
    st.write("### 请输入地块条件：")
    region = st.selectbox("区域 (Region)", region_options)
    soil_type = st.selectbox("土壤 (Soil_Type)", soil_type_options)
    temperature = st.slider("温度 (Temperature)", min_value=temp_min, max_value=temp_max, step=0.5)
    rainfall = st.slider("降雨量 (Rainfall)", min_value=rainfall_min, max_value=rainfall_max, step=10.0)
    fertilizer_used = st.selectbox("是否施肥 (Fertilizer_Used)", [0, 1])
    irrigation_used = st.selectbox("是否灌溉 (Irrigation_Used)", [0, 1])
    weather_cond = st.selectbox("天气状况 (Weather_Condition)", weather_cond_options)

    # 可选作物列表 - 使用所有作物类型
    crops = crop_options

    if st.button("推荐最适合作物"):
        # 构造用户输入
        base_input = {
            "Region": region,
            "Soil_Type": soil_type,
            "Temperature_Celsius": temperature,
            "Rainfall_mm": rainfall,
            "Fertilizer_Used": fertilizer_used,
            "Irrigation_Used": irrigation_used,
            "Weather_Condition": weather_cond
        }

        # 调用推荐接口
        recommendations = predictor.predict_crops(crops, base_input)

        # 输出推荐结果
        st.write("### 作物推荐结果：")
        for i, rec in enumerate(recommendations[:3]):
            st.success(f"第 {i + 1} 名: {rec['Crop']} （预测单产: {rec['Yield']:.2f} 吨/公顷）")

        # 显示作物预测产量的表格和图表
        st.write("### 所有作物预测产量：")
        results_df = pd.DataFrame(recommendations)
        st.dataframe(results_df)

        # 条形图可视化
        fig = px.bar(results_df, x="Crop", y="Yield", title="各作物预测单产", text="Yield")
        st.plotly_chart(fig, use_container_width=True)

elif menu == "管理方案推荐":
    st.header("管理方案推荐 🚜")

    # 用户输入地块条件 - 使用动态选项
    st.write("### 请输入地块条件：")
    crop = st.selectbox("作物", crop_options)  # 使用动态作物选项
    region = st.selectbox("区域", region_options)
    soil_type = st.selectbox("土壤 (Soil_Type)", soil_type_options)
    temperature = st.slider("温度 (Temperature_Celsius)", min_value=temp_min, max_value=temp_max, step=0.5)
    rainfall = st.slider("降雨量 (Rainfall_mm)", min_value=rainfall_min, max_value=rainfall_max, step=10.0)
    weather_cond = st.selectbox("天气状况 (Weather_Condition)", weather_cond_options)

    # 构造用户基础输入
    base_input = {
        "Crop": crop,
        "Region": region,
        "Soil_Type": soil_type,
        "Temperature_Celsius": temperature,
        "Rainfall_mm": rainfall,
        "Weather_Condition": weather_cond
    }

    if st.button("推荐管理方案"):
        # 遍历 Fertilizer_Used 和 Irrigation_Used 的所有组合
        results = []
        for fertilizer in [0, 1]:
            for irrigation in [0, 1]:
                input_data = base_input.copy()
                input_data["Fertilizer_Used"] = fertilizer
                input_data["Irrigation_Used"] = irrigation

                # 调用模型预测单产
                predicted_yield = predictor.predict_single(input_data)
                results.append({
                    "Fertilizer_Used": fertilizer,
                    "Irrigation_Used": irrigation,
                    "Yield": predicted_yield
                })

        # 转换为 DataFrame，并排序
        results_df = pd.DataFrame(results)
        high_yield_df = results_df.sort_values(by="Yield", ascending=False)

        # 高产方案推荐
        st.write("### 🏆 高产方案推荐")
        best_high_yield = high_yield_df.iloc[0]
        st.success(
            f"最佳高产方案: 施肥 {int(best_high_yield['Fertilizer_Used'])}, "
            f"灌溉 {int(best_high_yield['Irrigation_Used'])} "
            f"(预测单产: {best_high_yield['Yield']:.2f})"
        )

        # 绿色低碳方案推荐
        st.write("### 🌱 绿色低碳方案推荐")
        low_resource_df = results_df[results_df["Yield"] >= 0.95 * best_high_yield["Yield"]]
        low_resource_df = low_resource_df.sort_values(by=["Fertilizer_Used", "Irrigation_Used"], ascending=True)
        
        if not low_resource_df.empty:
            best_low_resource = low_resource_df.iloc[0]
            st.info(
                f"最佳绿色方案: 施肥 {int(best_low_resource['Fertilizer_Used'])}, "
                f"灌溉 {int(best_low_resource['Irrigation_Used'])} "
                f"(预测单产: {best_low_resource['Yield']:.2f})"
            )
        else:
            st.info("当前条件下无满足低碳要求的替代方案，请以高产方案为准。")

        # 显示完整方案表格
        st.write("### 管理方案比较")
        st.dataframe(results_df)

        # 条形图可视化
        st.write("### 不同方案的单产对比")
        results_df["方案"] = results_df.apply(
            lambda x: f"施肥:{int(x['Fertilizer_Used'])}, 灌溉:{int(x['Irrigation_Used'])}", axis=1
        )
        fig = px.bar(results_df, x="方案", y="Yield", color="Yield", title="管理方案单产对比")
        st.plotly_chart(fig, use_container_width=True)
elif menu == "模型可解释性":
    st.header("模型可解释性 🧠")
    st.info("SHAP 计算量较大，已开启缓存与采样加速。首次加载需等待几秒，后续刷新秒开。")

    # 检查是否有数据用于解释（假设你在前面全局加载了 X_train，如果没有，需要加载）
    try:
        # 假设你的预处理后的特征数据存放在 X_train 中
        # 如果你没有 X_train，这里需要从原始数据读取并预处理，例如：
        # df = pd.read_csv("data/processed_data.csv")
        # X_train = df.drop(columns=["Yield"]) 
        
        with st.spinner("正在计算 SHAP 值，请勿刷新页面..."):
            # 调用带缓存的函数，传入模型和特征数据
            explainer, X_sample, shap_values = get_shap_values(predictor.model, X_train, sample_size=200)

        st.success("SHAP 计算完成！")

        # --- 图表 1：特征重要性柱状图 ---
        st.write("#### 1. 全局特征重要性 (哪些因素最影响产量？)")
        fig1 = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
        st.pyplot(fig1)
        plt.clf() # 关键：清空当前 figure 释放内存，否则多次绘图会内存泄漏

        # --- 图表 2：特征影响散点图 ---
        st.write("#### 2. 特征影响分布 (高/低值如何影响产量？)")
        fig2 = plt.figure(figsize=(10, 6))
        shap.summary_plot(shap_values, X_sample, show=False)
        st.pyplot(fig2)
        plt.clf()

    except NameError:
        st.error("未找到用于解释的特征数据 (X_train)。请确保在 app.py 开头加载了预处理后的数据集。")
    except Exception as e:
        st.error(f"SHAP 计算出错: {e}")


elif menu == "敏感性分析":
    st.header("敏感性分析 🌦️")

    # 用户选择作物与固定条件 - 使用动态选项
    st.write("### 选择作物与固定条件")
    crop = st.selectbox("作物 (Crop)", crop_options)  # 使用动态作物选项
    region = st.selectbox("区域 (Region)", region_options)  # 使用动态区域选项
    soil_type = st.selectbox("土壤类型 (Soil_Type)", soil_type_options)  # 使用动态土壤类型选项
    fertilizer_used = st.selectbox("是否施肥 (Fertilizer_Used)", [0, 1])
    irrigation_used = st.selectbox("是否灌溉 (Irrigation_Used)", [0, 1])
    weather_cond = st.selectbox("天气状况 (Weather_Condition)", weather_cond_options)  # 使用动态天气状况选项
    days_to_harvest = st.slider("生长周期 (Days_to_Harvest)", 60, 180, 120)

    # 用户选择敏感性变量
    sensitivity_var = st.radio(
        "分析变量",
        options=["Temperature (温度)", "Rainfall (降雨量)"]
    )

    # 敏感性变量范围选择 - 使用动态范围
    if sensitivity_var == "Temperature (温度)":
        sensitivity_range = st.slider("选择温度变化范围(°C)", temp_min, temp_max, (temp_min, temp_max))
    else:
        sensitivity_range = st.slider("选择降雨量变化范围", rainfall_min, rainfall_max, (rainfall_min, rainfall_max))

    # 确认输入
    st.write("### 固定条件为：")
    base_input = {
        "Crop": crop,
        "Region": region,
        "Soil_Type": soil_type,
        "Fertilizer_Used": fertilizer_used,
        "Irrigation_Used": irrigation_used,
        "Weather_Condition": weather_cond,
        "Days_to_Harvest": days_to_harvest
    }
    st.json(base_input)

    # 执行敏感性分析
    if st.button("运行敏感性分析", type="primary"):
        with st.spinner("正在计算敏感性..."):
            results = []

            # 遍历敏感性变量范围
            if sensitivity_var == "Temperature (温度)":
                temp_range = np.arange(sensitivity_range[0], sensitivity_range[1] + 0.5, 0.5)
                for temp in temp_range:
                    input_data = base_input.copy()
                    input_data["Temperature_Celsius"] = temp
                    input_data["Rainfall_mm"] = (sensitivity_range[1] + sensitivity_range[0]) / 2
                    yield_pred = predictor.predict_single(input_data)
                    results.append({"Variable": temp, "Yield": yield_pred})
            else:
                rain_range = np.arange(sensitivity_range[0], sensitivity_range[1] + 1, 1)
                for rain in rain_range:
                    input_data = base_input.copy()
                    input_data["Rainfall_mm"] = rain
                    input_data["Temperature_Celsius"] = (sensitivity_range[1] + sensitivity_range[0]) / 2
                    yield_pred = predictor.predict_single(input_data)
                    results.append({"Variable": rain, "Yield": yield_pred})

            # 转换为 DataFrame
            results_df = pd.DataFrame(results)
            variable_name = "Temperature_Celsius" if sensitivity_var == "Temperature (温度)" else "Rainfall_mm"

            # 显示敏感性结果表格
            st.write(f"### {variable_name} 与单产的关系：")
            st.dataframe(results_df)

            # 绘制敏感性趋势图
            st.write("### 敏感性分析趋势图")
            fig = px.line(results_df, x="Variable", y="Yield", 
                          title=f"{variable_name} 与单产变化趋势",
                          labels={"Variable": variable_name, "Yield": "预测单产 (吨/公顷)"})
            st.plotly_chart(fig, use_container_width=True)

            # 分析结论
            st.write("### 敏感性分析结论")
            max_yield = results_df["Yield"].max()
            min_yield = results_df["Yield"].min()
            optimal_value = results_df.loc[results_df["Yield"].idxmax(), "Variable"]
            worst_value = results_df.loc[results_df["Yield"].idxmin(), "Variable"]
            
            st.success(f"🏆 最优条件：在 {variable_name} 为 {optimal_value:.2f} 时，单产达到最大值 {max_yield:.2f} 吨/公顷。")
            st.warning(f"⚠️ 风险提示：在 {variable_name} 为 {worst_value:.2f} 时，单产降至最低 {min_yield:.2f} 吨/公顷，降幅达 {((max_yield-min_yield)/max_yield)*100:.1f}%。")


elif menu == "模型测试与评估":
    st.header("模型测试与评估 📊")
    
    try:
        # 加载数据（抽样10%加速测试）
        df = pd.read_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "example_data.csv"))
        test_data = df.sample(frac=0.1, random_state=42)
        
        # 检查列名（调试用，可注释）
        st.write("数据列名:", test_data.columns.tolist())
        
        # 预处理：正确接收 X 和 y
        X_test_processed, y_test_processed = processor.preprocess(test_data)
        
        # 模型预测
        y_pred = predictor.model.predict(X_test_processed)
        
        # 计算评估指标 - 修改：兼容不同版本的 scikit-learn
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import numpy as np
        
        mae = mean_absolute_error(y_test_processed, y_pred)
        
        # 兼容不同版本的 mean_squared_error
        try:
            # 新版本 scikit-learn 支持 squared 参数
            rmse = mean_squared_error(y_test_processed, y_pred, squared=False)
        except TypeError:
            # 旧版本 scikit-learn 不支持 squared 参数
            rmse = np.sqrt(mean_squared_error(y_test_processed, y_pred))
        
        r2 = r2_score(y_test_processed, y_pred)
        
        # 显示结果
        st.write("### 模型性能评估指标")
        col1, col2, col3 = st.columns(3)
        col1.metric("MAE", f"{mae:.4f} 吨/公顷")
        col2.metric("RMSE", f"{rmse:.4f} 吨/公顷")
        col3.metric("R²", f"{r2:.4f}")
        
        # 可视化（散点图+误差分布）
        st.write("### 预测值 vs 实际值")
        fig1 = px.scatter(x=y_test_processed, y=y_pred, 
                         labels={'x': '实际单产', 'y': '预测单产'},
                         title='模型预测效果')
        fig1.add_shape(type="line", x0=y_test_processed.min(), y0=y_test_processed.min(), 
                      x1=y_test_processed.max(), y1=y_test_processed.max(),
                      line=dict(color="Red", dash="dash"))
        st.plotly_chart(fig1)
        
        st.write("### 误差分布")
        errors = y_test_processed - y_pred
        fig2 = px.histogram(errors, nbins=50, 
                           labels={'value': '预测误差'},
                           title='误差分布')
        st.plotly_chart(fig2)
        
        # 下载测试结果
        test_results = pd.DataFrame({
            '实际单产': y_test_processed,
            '预测单产': y_pred,
            '误差': errors
        })
        st.download_button(
            label="下载测试结果",
            data=test_results.to_csv(index=False),
            file_name="test_results.csv",
            mime="text/csv"
        )
        
    except FileNotFoundError:
        st.error("数据文件未找到，请检查路径：C:\\Users\\UESTC\\Desktop\\agri_yield_forecast\\data\\example_data.csv")
    except Exception as e:
        st.error(f"测试出错: {e}")


else:
    st.write("请选择功能模块！")
