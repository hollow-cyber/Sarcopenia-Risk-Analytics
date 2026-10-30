# Sarcopenia Risk Analytics (SRA) | 肌少症风险分析平台

[![Streamlit App](https://img.shields.io/badge/Streamlit-Open%20App-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://sarcopenia-risk-analytics-j5mybxvzszqvdazhbxtjrf.streamlit.app/)
[![Institution: WCH](https://img.shields.io/badge/Institution-West%20China%20Hospital-blue)](http://www.wchscu.cn/)

[English](#english) | [中文说明](#中文说明) | [Quick Start / 快速开始](#quick-start)

---

<a id="english"></a>

## English

### Overview

**Sarcopenia Risk Analytics (SRA)** is a Streamlit application for estimating the **1-7-year cumulative risk of incident sarcopenia** using five baseline measurements. The prediction engine uses **Ridge-penalized Fine-Gray regression**, accounting for **death as a competing event**. Sarcopenia was defined using the **Asian Working Group for Sarcopenia (AWGS) 2025** criteria.

The tool was developed from the **West China Health and Aging Trend (WCHAT)** longitudinal cohort. It is intended for community-dwelling adults aged **50 years or older who are free of sarcopenia at baseline**. Predictions require clinical interpretation when used outside the study population.

This project is supported by the Department of Geriatrics and the National Clinical Research Center for Geriatrics, West China Hospital, Sichuan University, China.

### Features

- **Five-input assessment:** age, sex, mid-upper arm circumference, hip circumference and calf circumference.
- **Annual prediction horizons:** select one or more years from 1 to 7, or select all years together.
- **Absolute risk display:** percentages and progress bars on a common 0%-100% scale.
- **Interactive cumulative-risk curve:** step-function display with probability tooltips, zooming and markers for selected years.
- **Downloadable PDF report:** baseline measurements, selected-year percentages and progress bars, the cumulative-risk curve, generation date and model information.
- **Plain-language model information:** explanations of the model and the five measurements, with measurement guidance in the input tooltips.

### Prediction workflow

The saved bundle contains **five imputation-specific models refitted on the full development cohort**, with five matching fitted preprocessors.

1. Collect the five raw baseline measurements.
2. Apply the saved scaling parameters corresponding to each model. The four continuous predictors are standardized; sex remains coded as male=1 and female=2.
3. Use each Fine-Gray model to predict its cumulative incidence function (CIF).
4. Average the five **predicted probabilities at each time point**.
5. Use the same averaged curve for the website and PDF report.

The displayed risk is the estimated probability of developing sarcopenia within the stated horizon, accounting for competing death. It is not an overall survival probability. The curve starts at zero at baseline and increases in steps. In the currently supplied models, years 1 and 2 have equal predictions because there is no additional recorded sarcopenia event in using cohort at year 2.

---

<a id="中文说明"></a>

## 中文说明

### 项目简介

**肌少症风险分析平台（SRA）** 是一款基于 Streamlit 的应用程序，使用5项基线测量指标估计 **1-7年内新发肌少症的累计风险**。预测引擎采用**带Ridge惩罚的Fine-Gray回归**，并将**死亡作为竞争事件**纳入分析。肌少症依据**亚洲肌少症工作组（AWGS）2025标准**定义。

本工具基于 **West China Health and Aging Trend（WCHAT）**纵向队列开发，面向**年龄≥50岁且基线无肌少症的社区居民**。用于研究人群之外的个体时，应结合临床情况解释预测结果。

本项目受到四川大学华西医院老年医学科及国家老年疾病临床医学研究中心的支持。

### 主要特色

- **五项指标评估：**年龄、性别、上臂围、臀围和小腿围。
- **年度预测时点：**可选择1-7年中的一个或多个年份，也可一次选择全部年份。
- **绝对风险展示：**以百分比及进度条展示风险，统一采用0%-100%的尺度。
- **交互式累计风险曲线：**采用阶梯式曲线，支持风险数值提示、缩放及所选年份标记。
- **可下载的PDF报告：**包含基线测量指标、所选年份的风险百分比及进度条、累计风险曲线、生成日期和模型信息。
- **通俗的模型说明：**介绍模型及5项测量指标，并在输入框提示中提供测量指导。

### 预测流程

保存的模型包包含**在完整开发集上重新拟合、分别对应5个插补数据集的模型**，并配有5个匹配的已拟合预处理器。

1. 收集5项原始基线测量指标。
2. 应用与各模型对应的已保存标准化参数。4项连续预测指标进行标准化；性别保留男=1、女=2的编码。
3. 使用各Fine-Gray模型预测其累计发生函数（CIF）。
4. 在每个时间点对5个模型的**预测概率取平均**。
5. 网页与PDF报告使用同一条平均预测曲线。

所展示的风险是在考虑死亡竞争事件后，个体于指定预测期限内发生肌少症的估计概率，而非总体生存概率。曲线在基线时从0开始，随后呈阶梯式上升。在当前提供的模型中，由于所用队列第2年没有记录新增的肌少症事件，第1年与第2年的预测风险相同。

---

## Required Inputs | 必需输入

| Predictor / 指标 | Internal name / 代码名称 | Unit or coding / 单位或编码 |
| --- | --- | --- |
| Age / 年龄 | `age` | Years; age ≥50 / 岁，年龄≥50岁 |
| Sex / 性别 | `sex` | Select Male or Female; internally 1 or 2 / 界面选择男或女，内部编码为1或2 |
| Mid-upper arm circumference / 上臂围 | `arm_circumference` | cm |
| Hip circumference / 臀围 | `hip_circumference` | cm |
| Calf circumference / 小腿围 | `calf_circumference` | cm |

An optional record identifier can be included in the report. It is not a model predictor. 

至少选择一个预测年份；可选填写记录编号，该编号不参与预测。

---

## Project Structure | 项目结构

```text
Sarcopenia-Risk-Predictor/
├── app.py                                  # Streamlit application entry
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

<a id="quick-start"></a>

## Quick Start | 快速开始

### 1. Prepare the environment | 准备环境

Open a terminal in the project folder and activate the intended Python environment. 在项目文件夹中打开终端，并激活准备使用的Python环境。

```bash
python -m pip install -r requirements.txt
```

Use an environment compatible with the saved model artifacts. The application loads existing models; running it does not require the original cohort data. 请保持运行环境与保存的模型兼容；启动网站无需原始队列数据。

### 2. Check model assets | 检查模型文件

Confirm that the bundle and all five `models/scalers/scalers_N.joblib` files are present. The loader checks the five-predictor schema and requires the final full-development-cohort model bundle. 确认最终模型包及5个标准化器齐全；加载器会核验5项指标、模型数量及完整开发集标记。

### 3. Start the application | 启动应用

```bash
python -m streamlit run app.py
```

Open the local URL printed by Streamlit. Enter the five measurements, select prediction years, and click **Run Assessment**. Results appear as risk percentages, progress bars and an interactive curve. Use **Download Risk Report (PDF)** to export the corresponding report.

打开Streamlit输出的本地地址，填写5项指标并选择预测年份，点击 **Run Assessment** 查看结果，再通过 **Download Risk Report (PDF)** 下载报告。修改输入后，需要重新点击评估按钮；旧结果保留时页面会显示提醒。

---

## Online Access | 在线访问

[Open Sarcopenia Risk Analytics / 打开在线工具](https://sarcopenia-risk-analytics-j5mybxvzszqvdazhbxtjrf.streamlit.app/)

## Data Handling | 数据处理

Inputs are processed by the Python process running the Streamlit application: on the local machine for a local run, or on the hosting server for an online deployment. The application keeps assessment results in session state and generates PDFs in memory; it does not implement persistent storage of individual inputs in a database or local data file. A downloaded PDF is saved by the user's browser.

输入由运行Streamlit的Python进程处理：本地运行时在本机计算，在线部署时在托管服务器计算。应用在会话状态中保留评估结果，并在内存中生成PDF；代码不会将个体输入持久化保存到数据库或本地数据文件。下载后的PDF由用户浏览器保存。

## Intended Use | 使用说明

This tool is provided for **non-commercial research and educational use** and to support clinical evaluation. It does not establish a diagnosis, determine treatment or replace professional clinical judgment. 

本工具用于**非商业性科研、教育及辅助临床评估**，不用于确立诊断或决定治疗，也不能替代专业临床判断。

© 2026 West China Hospital, Sichuan University, China.
