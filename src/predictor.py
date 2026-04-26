import pandas as pd
import joblib

class YieldPredictor:
    def __init__(self, model_path, feature_path):
        """
        初始化预测器
        :param model_path: 模型文件路径
        :param feature_path: 特征列名文件路径
        """
        self.model = joblib.load(model_path)
        self.feature_columns = joblib.load(feature_path)

    def predict_single(self, input_dict):
        """
        单条数据预测
        :param input_dict: 输入数据字典（如 {"Crop": "Wheat", "Temperature_Celsius": 25.0}）
        :return: 预测的单产值
        """
        # 1. 将字典转为 DataFrame
        df = pd.DataFrame([input_dict])
        
        # 2. 对分类变量进行独热编码（必须和 preprocess.py 里的列一致）
        categorical_cols = ['Crop', 'Region', 'Soil_Type', 'Weather_Condition']
        df = pd.get_dummies(df, columns=categorical_cols)
        
        # 3. 对齐特征列（确保和训练时的列顺序一致）
        for col in self.feature_columns:
            if col not in df.columns:
                df[col] = 0
        df = df[self.feature_columns]
        
        # 4. 预测并返回结果
        prediction = self.model.predict(df)[0]
        return prediction

    def predict_crops(self, crops, base_input):
        """
        预测多种作物的单产，返回按产量排序的推荐列表
        :param crops: 作物列表（如 ["Wheat", "Corn", "Rice"]）
        :param base_input: 基础输入字典（包含 Region、Soil_Type 等固定条件）
        :return: 按产量排序的作物推荐列表（字典列表）
        """
        recommendations = []
    
        for crop in crops:
            # 复制基础输入，添加当前作物
            input_data = base_input.copy()
            input_data["Crop"] = crop
        
            # 预测单产
            yield_pred = self.predict_single(input_data)
        
            # 添加到推荐列表
            recommendations.append({
                "Crop": crop,
                "Yield": yield_pred
            })
    
        # 按产量降序排序
        recommendations.sort(key=lambda x: x["Yield"], reverse=True)
        return recommendations