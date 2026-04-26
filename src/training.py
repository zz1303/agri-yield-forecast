from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib
import os
import sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from preprocess import DataPreprocessor

class ModelTrainer:
    def __init__(self):
        self.models = {
            "LinearRegression": LinearRegression(),
            "RandomForest": RandomForestRegressor(random_state=42),
            "XGBoost": XGBRegressor(objective='reg:squarederror', random_state=42)
        }

    def train(self, model_name, X_train, y_train):
        """
        训练指定模型
        :param model_name: 模型名称
        :param X_train: 训练集特征
        :param y_train: 训练集目标
        :return: 训练好的模型
        """
        print(f"正在训练模型: {model_name}")
        model = self.models[model_name]
        model.fit(X_train, y_train)
        return model

    def evaluate(self, model, X_test, y_test):
        """
        评估模型性能
        :param model: 训练好的模型
        :param X_test: 测试集特征
        :param y_test: 测试集目标
        :return: MAE, RMSE, R²
        """
        y_pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = mean_squared_error(y_test, y_pred) ** 0.5
        r2 = r2_score(y_test, y_pred)
        print(f"评估结果 -> MAE: {mae}, RMSE: {rmse}, R²: {r2}")
        return mae, rmse, r2

    def save_model(self, model, filepath):
        """
        保存模型到本地
        :param model: 训练好的模型
        :param filepath: 模型保存路径
        """
        joblib.dump(model, filepath)
        print(f"模型已保存至 {filepath}")


if __name__ == "__main__":
    # 1. 配置路径
    DATA_PATH = "../data/example_data.csv"
    MODEL_PATH = "../models/yield_model.pkl"
    
    # 2. 确保模型保存的文件夹存在
    os.makedirs("../models", exist_ok=True)

    # 3. 字段配置
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

    # 4. 实例化
    processor = DataPreprocessor(COLUMN_CONFIG)
    trainer = ModelTrainer()

    # 5. 加载并预处理数据
    print("正在加载数据...")
    df = processor.load_data(DATA_PATH)
    X, y = processor.preprocess(df)
    
    # 6. 调整数据划分比例：90%训练集，10%测试集
    X_train, X_test, y_train, y_test = processor.split_data(X, y, test_size=0.1, random_state=42)
    print(f"数据集划分完成 - 训练集: {len(X_train)} 样本, 测试集: {len(X_test)} 样本")

    # 7. 训练模型 (使用XGBoost)
    best_model_name = "XGBoost"  # 使用XGBoost模型
    model = trainer.train(best_model_name, X_train, y_train)

    # 8. 评估模型
    trainer.evaluate(model, X_test, y_test)

    # 9. 保存模型
    trainer.save_model(model, MODEL_PATH)

    # 10. 保存特征列名
    joblib.dump(X_train.columns, '../models/feature_columns.pkl')