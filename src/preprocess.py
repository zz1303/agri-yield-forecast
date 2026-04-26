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
        # 映射字段（缩进）
        df = df.rename(columns=self.config)

        # 填充缺失值（缩进）
        df.fillna({
            'Temperature_Celsius': df['Temperature_Celsius'].mean(),
            'Rainfall_mm': df['Rainfall_mm'].mean(),
            'Fertilizer_Used': df['Fertilizer_Used'].mean(),
            'Irrigation_Used': df['Irrigation_Used'].mean()
        }, inplace=True)
        df.dropna(inplace=True)

        # 编码分类变量（缩进）
        categorical_cols = ['Crop', 'Region', 'Soil_Type', 'Weather_Condition']  # 添加 Weather_Condition
        df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

        # 分离特征和目标（缩进）
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
    "Yield": "Yield_tons_per_hectare"
}
