# Sarcopenia Risk Analytics (SRA) | 肌少症风险分析平台

[![Streamlit App](https://static.streamlit.io/badge_streamlit.svg)](https://sarcopenia-risk-analytics.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Institution: WCH](https://img.shields.io/badge/Institution-West%20China%20Hospital-blue)](http://www.wchscu.cn/)

[English](#english) | [中文说明](#中文说明)

---

<a name="english"></a>
## 🌍 English Description

### 🏛️ Clinical Background
**Sarcopenia Risk Analytics (SRA)** is a professional-grade clinical decision support tool designed for individualized prediction of sarcopenia onset risk. Powered by a **Cox Proportional Hazards Ensemble Model**, it transforms baseline clinical metrics into longitudinal prognostic trajectories.

This project is supported by:
* **Department of Geriatrics**, West China Hospital (WCH), Sichuan University.
* **National Clinical Research Center for Geriatrics**, China.

### ✨ Key Features
* **Ensemble Prognostic Engine**: Multi-fold cross-validated Cox models for stable consensus risk estimation.
* **Dynamic Survival Trajectories**: Interactive 7-year survival curves powered by **Altair**.
* **Clinical Reporting**: High-resolution, institutional-branded PDF reports with dynamic risk-level styling.
* **OOD Detection**: Proactive warnings for inputs outside the model's validated training distribution.



---

<a name="中文说明"></a>
## 🇨🇳 中文说明

### 🏛️ 临床背景与支持
**肌少症风险分析平台 (SRA)** 是一款专为老年人群设计的专业级临床决策支持工具。该平台基于 **Cox 比例风险集成模型**，能够将患者的基础临床指标转化为长期的个体化肌少症发生风险轨迹图。

技术支持单位：
* **四川大学华西医院** 老年医学科。
* **国家老年疾病临床医学研究中心**。

### ✨ 核心功能
* **集成预后引擎**：采用多折交叉验证的 Cox 集成模型，提供稳健的风险共识评估。
* **动态生存轨迹**：基于 **Altair** 实现 7 年期交互式生存概率曲线展示，支持缩放与悬停。
* **专业临床报告**：一键生成带有华西医院标识的高分辨率 PDF 评估报告。
* **分布外检测 (OOD)**：自动识别并警示超出模型验证范围的异常输入，确保预测可靠性。

---

## 📂 Project Structure | 项目结构

```text
Sarcopenia-Risk-Analytics/
├── app.py                # Main app entry | 主程序入口
├── requirements.txt      # Dependencies | 依赖库列表
├── logo.ico              # Institutional Logo | 机构图标
├── feature_mapping.txt   # Feature labels mapping | 特征标签映射表
├── src/                  # Source code | 核心代码
│   ├── prediction.py     # Inference & OOD logic | 风险推断与计算
│   ├── report_generator.py# PDF reporting engine | PDF报告生成引擎
│   ├── outputs.py        # Visualizations | 结果可视化组件
│   ├── inputs.py         # UI input components | 用户输入组件
│   └── layouts.py        # Custom CSS & Headers | 页面布局与样式
├── models/               # Model weights | 训练完毕的模型信息
└── config/               # Thresholds & bounds | 临床阈值与分布边界配置
```

## 🚀 Quick Start | 快速开始

1. **Clone & Install | 克隆与安装**:
   ```bash
   git clone [https://github.com/your-username/Sarcopenia-Risk-Analytics.git](https://github.com/your-username/Sarcopenia-Risk-Analytics.git)
   cd Sarcopenia-Risk-Analytics
   pip install -r requirements.txt

2. **Run Application | 运行应用**:
   ```bash
   streamlit run app.py
---

## ⚖️ Disclaimer | 免责声明 ##
This tool is for clinical decision support only and does not constitute a formal medical diagnosis. Final diagnostic responsibility remains exclusively with the presiding physician. Provided for non-commercial research and educational use only.

本工具仅用于临床决策支持，不构成正式医学诊断。最终诊断责任由主治医师承担。本软件仅供非商业性科研及教育使用。
---
© 2025 West China Hospital, Sichuan University.   
