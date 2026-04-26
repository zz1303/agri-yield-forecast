import joblib
import pandas as pd


class YieldPredictor:
    def __init__(self, model_path, config):
        """
        初始化推理器
        :param model_path: 模型文件路径
        :param config: 数据列名配置
        """
        self.model = joblib.load(model_path)
        self.config = config
        print("模型加载成功")

    def predict_single(self, user_input):
        """
        接收单条用户输入，预测单产
        :param user_input: 字典形式的输入数据
        :return: 预测产量
        """
        # 格式化输入为 DataFrame
        df = pd.DataFrame([user_input])
        processor = DataPreprocessor(self.config)
        
        # 使用预处理器进行数据准备
        df_preprocessed, _ = processor.preprocess(df)
        
        # 模型预测
        prediction = self.model.predict(df_preprocessed)
        return prediction[0]

    def predict_crops(self, crops, base_input):
        """
        遍历所有作物，预测单产，用于作物推荐模块
        :param crops: 待遍历的作物列表
        :param base_input: 基础输入条件（不包含作物）
        :return: 各作物的预测产量，按降序排列
        """
        results = []
        for crop in crops:
            input_data = base_input.copy()
            input_data['Crop'] = crop
            yield_prediction = self.predict_single(input_data)
            results.append({"Crop": crop, "Yield": yield_prediction})
        
        # 按预测产量排序
        results = sorted(results, key=lambda x: x['Yield'], reverse=True)
        return results
