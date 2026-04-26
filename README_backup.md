agri_yield_forecast/ # 项目根目录
├── data/ # 数据目录
├── models/ # 保存模型
├── notebooks/ # 数据分析及建模 notebook
├── src/ # 核心代码模块
│ ├── preprocess.py # 数据预处理模块
│ ├── training.py # 模型训练模块
│ ├── predictor.py # 推理模块（需创建）
│ └── utils.py # 通用工具模块（可选）
├── app.py # Streamlit 应用主入口
├── requirements.txt # 依赖声明文件
└── README.md # 项目说明文档


## 本地运行
### 1. 克隆项目
bash
git clone https://github.com/your-username/agri_yield_forecast.git
cd agri_yield_forecast


### 2. 安装依赖
确保 Python 版本 ≥ 3.8，并安装必要依赖：
bash
pip install -r requirements.txt


### 3. 运行应用
运行 Streamlit 主程序：
bash
streamlit run app.py


## 部署方法
- **Streamlit Community Cloud:** 将项目上传至 GitHub，绑定到 Streamlit Cloud 提供的服务，快速部署；
- **Hugging Face Spaces:** 将项目上传至 Hugging Face，使用 Streamlit 模板运行。