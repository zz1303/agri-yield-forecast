import pandas as pd
from sklearn.model_selection import train_test_split

# 数据预处理模块
class DataPreprocessor:
    def __init__(self, config):
        """
        初始化配置
        :param config: 数据列名配置 (字段映射)
        """
        self.config = config

    def load_data(self, filepath):
        """
        加载数据集
        :param filepath: 数据文件路径
        :return: DataFrame
        """
        data = pd.read_csv(filepath)
        print(f"数据加载成功，数据集包含 {data.shape[0]} 行，{data.shape[1]} 列")
        return data

    def preprocess(self, df):
        # 映射字段
        df = df.rename(columns=self.config)

        # 检查原始数据中的所有类别（新增）
        print("=== 原始数据探索 ===")
        print("作物类型:", df['Crop'].unique())
        print("区域类型:", df['Region'].unique())
        print("土壤类型:", df['Soil_Type'].unique())
        print("天气状况:", df['Weather_Condition'].unique())
        print("降雨量范围:", df['Rainfall_mm'].min(), "到", df['Rainfall_mm'].max())
        print("温度范围:", df['Temperature_Celsius'].min(), "到", df['Temperature_Celsius'].max())
        
        # 填充缺失值
        df.fillna({
            'Temperature_Celsius': df['Temperature_Celsius'].mean(),
            'Rainfall_mm': df['Rainfall_mm'].mean(),
            'Fertilizer_Used': df['Fertilizer_Used'].mean(),
            'Irrigation_Used': df['Irrigation_Used'].mean(),
            'Weather_Condition': df['Weather_Condition'].mode()[0]  # 用众数填充分类变量
        }, inplace=True)
        
        # 检查填充后的数据
        print("填充后缺失值数量:", df.isnull().sum().sum())
        
        # 编码分类变量
        categorical_cols = ['Crop', 'Region', 'Soil_Type', 'Weather_Condition']
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
        
        # 检查编码后的数据
        print("编码后数据形状:", df.shape)
        print("编码后列名:", df.columns.tolist())

        # 分离特征和目标
        X = df.drop('Yield_tons_per_hectare', axis=1)
        y = df['Yield_tons_per_hectare']
        return X, y

    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        划分训练集与测试集
        :param X: 特征矩阵
        :param y: 目标值
        :return: 训练集和测试集
        """
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
        print("数据集划分完成")
        return X_train, X_test, y_train, y_test


COLUMN_CONFIG = {
    "Crop": "Crop",
    "Region": "Region",
    "Soil_Type": "Soil_Type",
    "Temperature": "Temperature_Celsius",
    "Rainfall": "Rainfall_mm",
    "Fertilizer_Used": "Fertilizer_Used",
    "Irrigation_Used": "Irrigation_Used",
    "Weather_Condition": "Weather_Condition",  # 新增：添加Weather_Condition映射
    "Yield": "Yield_tons_per_hectare"
}