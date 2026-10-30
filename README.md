# 🩺 Sarcopenia Risk Analytics (SRA) | 肌少症风险分析平台

[![Streamlit App](https://img.shields.io/badge/Streamlit-Open%20App-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://sarcopenia-risk-analytics-j5mybxvzszqvdazhbxtjrf.streamlit.app/)
[![Institution: WCH](https://img.shields.io/badge/Institution-West%20China%20Hospital-blue)](http://www.wchscu.cn/)

[English](#english) | [中文说明](#中文说明) | [Online Access | 在线访问](#online-access)

---

<a id="english"></a>

## 🌍 English

### 📖 Overview

**Sarcopenia Risk Analytics (SRA)** is a Streamlit application that estimates the **1–7-year cumulative risk of incident sarcopenia** from five baseline measurements. The prediction engine uses **ridge-penalized Fine-Gray regression** to account for **death as a competing event**. Sarcopenia was defined according to the **Asian Working Group for Sarcopenia (AWGS) 2025** criteria.

The tool was developed using data from the **West China Health and Aging Trend (WCHAT)** longitudinal cohort. It is intended for community-dwelling adults aged **50 years or older who are free of sarcopenia at baseline**. Predictions for individuals outside the study population should be interpreted in their clinical context.

This project is supported by the Department of Geriatrics and the National Clinical Research Center for Geriatrics, West China Hospital, Sichuan University, China.

### ✨ Features

- **Five-input assessment:** age, sex, mid-upper arm circumference, hip circumference and calf circumference.
- **Annual prediction horizons:** select one or more years from 1 to 7, or select all years at once.
- **Absolute risk display:** risk percentages and progress bars on a consistent 0%–100% scale.
- **Interactive cumulative-risk curve:** a step curve with probability tooltips, zooming and markers at the selected years.
- **Downloadable PDF report:** baseline measurements, risk percentages and progress bars for the selected years, a cumulative-risk curve, the generation date and model information.
- **Plain-language model information:** an explanation of the model and its five inputs, with measurement guidance available in the input tooltips.

### ⚙️ Prediction Workflow

The saved bundle contains **five imputation-specific models refitted on the full development cohort**, each paired with its corresponding fitted preprocessor.

1. Collect the five raw baseline measurements.
2. Apply each model's saved scaling parameters. The four continuous predictors are standardized; sex remains coded as male = 1 and female = 2.
3. Obtain the predicted cumulative incidence function (CIF) from each Fine-Gray model.
4. Average the five **predicted probabilities at each time point**.
5. Use the resulting mean curve for both the website and the PDF report.

The displayed risk is the estimated probability of developing sarcopenia within the specified horizon, accounting for competing death. It is not an overall survival probability. The curve starts at zero at baseline and increases in steps. In the cohort used to fit the current models, no new sarcopenia events were recorded at year 2; the predicted risks at years 1 and 2 are therefore identical.

---

<a id="中文说明"></a>

## 🌏 中文说明

### 📖 项目简介

**肌少症风险分析平台（SRA）** 是一款基于 Streamlit 的应用程序，利用5项基线测量指标估计 **1–7年内新发肌少症的累计风险**。预测引擎采用**带Ridge惩罚的Fine-Gray回归**，将**死亡作为竞争事件**纳入分析。肌少症依据**亚洲肌少症工作组（AWGS）2025标准**定义。

本工具基于 **West China Health and Aging Trend（WCHAT）**纵向队列数据开发，面向**年龄≥50岁且基线无肌少症的社区居民**。对于研究人群之外的个体，应结合其临床情况解释预测结果。

本项目受到四川大学华西医院老年医学科及国家老年疾病临床医学研究中心的支持。

### ✨ 主要功能

- **五项指标评估：**年龄、性别、上臂围、臀围和小腿围。
- **年度预测时点：**可选择1–7年中的一个或多个年份，也可一键选择全部年份。
- **绝对风险展示：**以百分比和进度条展示风险，统一采用0%–100%的尺度。
- **交互式累计风险曲线：**采用阶梯式曲线，支持风险数值提示、缩放及所选年份标记。
- **可下载的PDF报告：**包含基线测量指标、所选年份的风险百分比与进度条、累计风险曲线、生成日期和模型信息。
- **通俗的模型说明：**介绍模型及其5项输入指标，并在输入框提示中提供测量指导。

### ⚙️ 预测流程

保存的模型包包含**在完整开发集上重新拟合的5个插补模型**，每个模型均配有对应的已拟合预处理器。

1. 收集5项原始基线测量指标。
2. 应用各模型对应的已保存标准化参数。4项连续预测指标进行标准化；性别保留男 = 1、女 = 2的编码。
3. 分别获得各Fine-Gray模型预测的累计发生函数（CIF）。
4. 在每个时间点对5个模型的**预测概率取平均**。
5. 网页与PDF报告使用由此得到的同一条平均预测曲线。

所展示的风险是在考虑死亡竞争事件后，个体于指定预测期限内发生肌少症的估计概率，而非总体生存概率。曲线在基线时从0开始，随后呈阶梯式上升。用于拟合当前模型的队列在第2年未记录到新发肌少症事件，因此第1年与第2年的预测风险相同。

---

## 📋 Required Inputs | 必需输入

| Predictor / 指标 | Internal name / 代码名称 | Unit / 单位 |
| --- | --- | --- |
| Age / 年龄 | `age` | Years; age ≥50 / 岁，年龄≥50岁 |
| Sex / 性别 | `sex` | Select Male or Female / 选择男或女 |
| Mid-upper arm circumference / 上臂围 | `arm_circumference` | cm |
| Hip circumference / 臀围 | `hip_circumference` | cm |
| Calf circumference / 小腿围 | `calf_circumference` | cm |

Select at least one prediction year. An optional record identifier can be included in the report; it is not used for prediction.

请至少选择一个预测年份。可选填写记录编号以便在报告中识别，该编号不参与预测。

---

## 📂 Project Structure | 项目结构

```text
Sarcopenia-Risk-Predictor/
├── streamlit_app.py                        # Streamlit application entry
├── README.md                               # Project documentation
├── requirements.txt                        # Application dependencies
├── logo.ico                                # Institutional logo
├── models/
│   ├── final_fine_gray_prediction_bundle.joblib
│   ├── feature_bounds.json
│   └── scalers/
│       ├── scalers_1.joblib
│       ├── scalers_2.joblib
│       ├── scalers_3.joblib
│       ├── scalers_4.joblib
│       └── scalers_5.joblib
└── src/
    ├── __init__.py
    ├── prediction.py                       # Model loading, scaling and mean CIF
    ├── inputs.py                           # Five-input sidebar and year selection
    ├── outputs.py                          # Risk cards, progress bars and Altair curve
    ├── report_generator.py                 # PDF chart and report generation
    └── layouts.py                          # Header, styling and usage information
```

## 🚀 Quick Start | 快速开始

### 1. Prepare the environment | 准备环境

Open a terminal in the project folder and activate your Python environment.

在项目文件夹中打开终端，并激活准备使用的Python环境。

```bash
python -m pip install -r requirements.txt
```

Use an environment compatible with the saved models and preprocessors. The application loads existing models and does not require the original cohort data to run.

请使用与已保存模型及预处理器兼容的运行环境。应用加载已有模型，运行时无需原始队列数据。

### 2. Check model assets | 检查模型文件

Confirm that the final model bundle and all five `models/scalers/scalers_N.joblib` files (`N = 1, …, 5`) are present. The loader verifies the five-predictor schema, the number of models and the full-development-cohort model designation.

确认最终模型包及5个 `models/scalers/scalers_N.joblib` 文件（`N = 1, …, 5`）齐全。加载器会核验5项指标、模型数量及完整开发集模型标记。

### 3. Start the application | 启动应用

```bash
python -m streamlit run streamlit_app.py
```

Open the local URL displayed by Streamlit. Enter the five measurements, select prediction years, and click **Run Assessment**. Results appear as risk percentages, progress bars and an interactive curve. Click **Download Risk Report (PDF)** to save the corresponding report. If you change the inputs, run the assessment again to refresh the results; a reminder appears while previous results are displayed.

打开Streamlit显示的本地地址，填写5项指标并选择预测年份，点击 **Run Assessment** 查看风险百分比、进度条及交互式曲线，再点击 **Download Risk Report (PDF)** 保存对应报告。修改输入后，请重新运行评估以刷新结果；仍显示旧结果时，页面会给出提醒。

---

<a id="online-access"></a>

## 🔗 Online Access | 在线访问

[Open Sarcopenia Risk Analytics / 打开在线工具](https://sarcopenia-risk-analytics-j5mybxvzszqvdazhbxtjrf.streamlit.app/)

---

## 🧾 Data Handling | 数据处理

Inputs are processed by the Python process running the Streamlit application: on your computer when run locally, or on the hosting server when deployed online. Assessment results are held in session state, and PDFs are generated in memory. The application does not persist individual inputs to a database or local data file. Downloaded PDFs are saved by your browser.

输入由运行Streamlit的Python进程处理：本地运行时在本机计算，在线部署时在托管服务器计算。评估结果保留在会话状态中，PDF在内存中生成。应用不会将个体输入持久化保存到数据库或本地数据文件；下载的PDF由浏览器保存。

---

## ℹ️ Intended Use | 使用说明

This tool is intended for **non-commercial research and education**, and as an aid to clinical evaluation. It does not establish a diagnosis, determine treatment or replace professional clinical judgment.

本工具用于**非商业性科研、教育及辅助临床评估**，不用于确立诊断或决定治疗，也不能替代专业临床判断。

© 2026 West China Hospital, Sichuan University, China.
