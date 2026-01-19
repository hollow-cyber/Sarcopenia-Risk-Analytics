# Sarcopenia Risk Analytics (SRA) | 肌少症风险分析平台

[![Streamlit App](https://img.shields.io/badge/Streamlit-Open%20App-FF4B4B?style=flat&logo=Streamlit&logoColor=white)](https://sarcopenia-risk-analytics-j5mybxvzszqvdazhbxtjrf.streamlit.app/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Institution: WCH](https://img.shields.io/badge/Institution-West%20China%20Hospital-blue)](http://www.wchscu.cn/)

[English](#english) | [中文说明](#中文说明)

---

<a name="english"></a>
## 🌍 English Description

### 🏛️ Clinical Background
**Sarcopenia Risk Analytics (SRA)** is a professional-grade clinical decision support tool **based on Asian Working Group for Sarcopenia (AWGS) 2025 consensus**, designing for individualized prediction of sarcopenia onset risk. Powered by a **Cox Proportional Hazards Ensemble Model**, it transforms baseline clinical metrics into longitudinal prognostic trajectories.

This project is supported by:
* Department of Geriatrics, West China Hospital, Sichuan University.
* National Clinical Research Center for Geriatrics, China.

### ✨ Key Features
* **Explainable Prognostic Engine**: Built upon the West China Health and Aging Trend (WCHAT) longitudinal cohort, our system utilizes a Cox Proportional Hazards (CPH) model validated through multi-fold cross-validation. It delivers **highly robust and clinically interpretable risk assessments for sarcopenia**, bridging the gap between machine learning and bedside decision-making.
* **Long-term Survival Trajectory**: Featuring an Altair-based interactive visualization suite, the platform renders **individualized survival probability curves over a 7-year horizon**. It supports high-precision hover-querying and dynamic scaling, intuitively capturing the non-linear evolution of patient risks over time.
* **Clinical-Grade Assessment Reports**: A built-in professional PDF generation module enables one-click exportation of comprehensive clinical reports. These reports automatically synthesize baseline patient metrics, multi-year risk projections, and visual diagnostics, facilitating standardized documentation and clinical decision support.



---

<a name="中文说明"></a>
## 中文说明

### 🏛️ 临床背景与支持
**肌少症风险分析平台 (SRA)** 是一款基于**亚洲肌少症工作组2025年共识**，专为老年人群设计的专业级临床决策支持工具。该平台基于 **Cox 比例风险集成模型**，能够将患者的基础临床指标转化为长期的个体化肌少症发生风险轨迹图。

技术支持单位：
* 四川大学华西医院 老年医学科。
* 国家老年疾病临床医学研究中心。

### ✨ 核心功能
* **可解释预测引擎**：基于 West China Health and Aging Trend (WCHAT) 纵向队列数据构建，采用多折交叉验证的 Cox 比例风险模型，提供具备**高度稳健性与可解释的肌少症风险评估**。
* **长周期生存轨迹**：集成 Altair 交互式可视化方案，**实时渲染 1-7 年期的个体化生存概率曲线**。支持高精度悬停检索与动态缩放，直观呈现患者风险随时间演变的非线性趋势。
* **临床级评估报告**：内置专业的 PDF 生成模块，可一键导出报告。报告自动整合患者基线参数、多年期预测风险及可视化图表，满足临床决策支持与标准化存档需求。

---

## 📂 Project Structure | 项目结构

```text
Sarcopenia-Risk-Analytics/
├── app.py                # Main app entry | 主程序入口
├── requirements.txt      # Dependencies | 依赖库列表
├── logo.ico              # Institutional Logo | 机构图标
├── feature-mapping.txt   # Feature labels mapping | 特征标签映射表
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
   git clone [https://github.com/hollow-cyber/Sarcopenia-Risk-Analytics.git](https://github.com/hollow-cyber/Sarcopenia-Risk-Analytics.git)
   cd Sarcopenia-Risk-Analytics
   pip install -r requirements.txt

2. **Run Application | 运行应用**:
   ```bash
   streamlit run streamlit_app.py
---

## ⚖️ Disclaimer | 免责声明 ##
This tool is for clinical decision support only and does not constitute a formal medical diagnosis. Final diagnostic responsibility remains exclusively with the presiding physician. Provided for non-commercial research and educational use only.

本工具仅用于临床决策支持，不构成正式医学诊断。最终诊断责任由主治医师承担。本软件仅供非商业性科研及教育使用。

---
© 2026 West China Hospital, Sichuan University, China.   
