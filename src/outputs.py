"""
This project is supported by the Department of Geriatrics and the National Clinical Research Center for Geriatrics,
West China Hospital, Sichuan University, China.
"""

import altair as alt
import pandas as pd
import streamlit as st
from src.prediction import load_thresholds, cal_probability_at_time, ensure_survival_func_0_time


def show_risk_metrics(survival_func: pd.Series, eval_times: list[int] | None) -> None:
	"""Calculates and displays risk metrics for specific time horizons.

	This function categorizes prediction timepoints into Short, Mid, or Long-term
	prognosis and renders them as interactive metric cards in the Streamlit UI.

	Args:
		survival_func (pd.Series): Individualized survival function.
		eval_times (list[int] | None): Timepoints (in years) to evaluate risk.
	"""
	if survival_func.empty or not eval_times:
		return
	
	# Filter out evaluation times that exceed the model's maximum follow-up duration
	max_time = survival_func.index.max()
	valid_times = sorted([t for t in eval_times if t <= max_time])
	
	st.markdown("### 📊 Sarcopenia Risk Assessment")
	
	# Create dynamic columns based on the number of selected timepoints
	cols = st.columns(len(valid_times))
	
	# Iterate through columns and timepoints to display metric cards
	for col, t in zip(cols, valid_times):
		with col:
			# Categorize the prediction horizon based on the year
			if t <= 2:
				horizon_label = "Short-term"
			elif 3 <= t <= 5:
				horizon_label = "Mid-term"
			else:
				horizon_label = "Long-term"
			
			st.caption(f"⏱️ {horizon_label} Forecast")
			
			# Extract risk probability (1 - Survival Probability)
			# Assuming cal_probability_at_time returns (prob_surv, prob_risk)
			_, prob_risk = cal_probability_at_time(survival_func, t)
			# Render the metric card
			st.metric(
				label=f"{t}-Year Cumulative Risk",
				value=f"{prob_risk:.2%}",
			)


def show_risk_stratification(relative_risk: float, status_text: str, method_name: str = "Cox") -> None:
	"""Renders a professional clinical summary and visual risk dashboard.

    This function serves as the visual output layer for risk analysis. It consumes
    pre-calculated risk metrics and categories to:
    1.  Render a dynamic HTML/CSS gradient progress bar showing the patient's position.
    2.  Provide a color-coded status indicator aligned with clinical risk levels.
    3.  Display domain-specific management recommendations (Monitoring, Exercise, Nutrition).

    Args:
        relative_risk (float): The pre-calculated Relative Risk score (exp(βx)).
        status_text (str): The clinical risk category derived from model assets
            (e.g., "Low Risk", "Moderate Risk", "High Risk").
        method_name (str): Identifier to fetch visual constants (like MAX_BAR)
            from local config. Defaults to "Cox".
    """
	
	# 1. Configuration & Threshold Retrieval
	thresholds = load_thresholds(method_name)
	LOW_RISK = thresholds.get("low_risk", 0.6)
	HIGH_RISK = thresholds.get("high_risk", 1.6)
	MAX_BAR = thresholds.get("max_display_rr", 3.0)
	
	# 2. Visual Logic for Progress Bar (Ensuring Robustness for Outliers)
	# The progress bar is capped at 100% width even if RR exceeds MAX_BAR
	progress_width = min((relative_risk / MAX_BAR) * 100, 100)
	
	if relative_risk < LOW_RISK:
		bar_color = "#28a745"  # Clinical Green
	elif relative_risk > HIGH_RISK:
		bar_color = "#dc3545"  # Clinical Red
	else:
		bar_color = "#ffc107"  # Clinical Amber/Yellow
	
	st.markdown("### 🚥 Risk Stratification Analysis")
	
	# 3. HTML/CSS Dashboard Component (Centered Visual)
	# This provides a more intuitive sense of risk location than a simple number.
	bar_html = f"""
	<div style="margin: 20px 0; font-family: sans-serif;">
		<div style="display: flex; justify-content: space-between; margin-bottom: 8px; font-size: 20px;">
			<span style="color: #495057;">Relative Risk Scale (RR)</span>
			<span style="font-weight: bold; color: {bar_color};">{status_text}</span>
		</div>
		<div style="background-color: #e9ecef; border-radius: 12px; height: 22px; width: 100%; border: 1px solid #dee2e6;">
			<div style="background-color: {bar_color}; width: {progress_width}%; height: 100%; border-radius: 12px; transition: width 0.8s ease-in-out;"></div>
		</div>
		<div style="display: flex; justify-content: space-between; margin-top: 10px; font-size: 18px; color: #adb5bd;">
			<span style="color: #000000;">0.00 (Baseline)</span>
			<span style="color: #000000;">{LOW_RISK:.2f} (Low)</span>
			<span style="color: #000000;">{HIGH_RISK:.2f} (High)</span>
			<span style="color: #000000;">{MAX_BAR:.2f}+ (Extreme)</span>
		</div>
	</div>
	"""
	st.markdown(bar_html, unsafe_allow_html=True)
	
	# 4. Professional Clinical Management Strategies
	st.markdown("#### 📋 Management Recommendations")
	
	# Summary Card
	st.markdown(f"""
	<div style="padding:18px; border-radius:10px; border-left: 6px solid {bar_color}; background-color: #f8f9fa; margin-bottom: 20px;">
		<strong style="color:{bar_color}; font-size:20px;">Assessment: {status_text}</strong><br>
		<span style="color:#6c757d; font-size:20px;">Individual Relative Risk (RR) calculated at <strong>{relative_risk:.2f}</strong>.</span>
	</div>
	""", unsafe_allow_html=True)
	
	# Detailed Domain-Specific Recommendations
	tab1, tab2, tab3 = st.tabs(["🔍 Monitoring", "🏋️ Exercise", "🥗 Nutrition"])
	
	with tab1:
		if status_text == "High Risk":
			st.write("**Frequency:** Re-assessment of physical performance every 3 months.")
			st.write("**Action:** Comprehensive Geriatric Assessment (CGA) to identify reversible contributors.")
		else:
			st.write("**Frequency:** Annual sarcopenia screening.")
			st.write("**Action:** Routine monitoring of calf circumference and SARC-F scores.")
	
	with tab2:
		if status_text == "High Risk":
			st.write("**Prescription:** Supervised High-Intensity Resistance Training (HIRT).")
			st.write("**Focus:** Power and strength exercises targeted at major lower-limb muscle groups.")
		else:
			st.write("**Prescription:** Community-based or home resistance exercises 2–3 times per week.")
	
	with tab3:
		if status_text == "High Risk":
			st.write("**Supplementation:** Increase protein intake to 1.2–1.5 g/kg/day.")
			st.write("**Optimization:** Consider leucine-enriched amino acids and Vitamin D (if <30 ng/mL).")
		else:
			st.write("**Target:** Maintain a balanced diet with protein-rich meals (≥25g protein/meal).")


def show_altair_survival_chart(survival_func: pd.Series, highlight_times: list | None = None) -> None:
	"""Renders a step-function survival curve with shaded area using Altair.

	This chart provides an intuitive visual representation of the individualized
	sarcopenia-free trajectory over time. It features interactive tooltips,
	shaded probability areas, and optional markers for key clinical timepoints.

	Args:
		survival_func (pd.Series): The individualized survival function.
		highlight_times (list | None): Specific timepoints (e.g., [1, 3, 5])
			to highlight with markers on the curve.
	"""
	
	# 1. Data Preparation
	# Ensure t=0 is included and convert Series to a long-format DataFrame
	data = ensure_survival_func_0_time(survival_func).reset_index()
	data.columns = ['Time', 'Survival Probability']
	
	# 2. Base Chart Definition
	# Define common encoding and axes for all layers
	base = alt.Chart(data).encode(
		x=alt.X(
			'Time:Q',
			title='Time (Years)',
			axis=alt.Axis(tickMinStep=1, format='d', grid=False)
		),
		y=alt.Y(
			'Survival Probability:Q',
			title='Sarcopenia-Free Probability',
			scale=alt.Scale(domain=[0, 1.05]),
			axis=alt.Axis(format='.2f')
		),
		tooltip=[
			alt.Tooltip('Time:Q', title='Time (Year)', format='d'),
			alt.Tooltip('Survival Probability:Q', title='Survival Prob.', format='.2%')
		]
	)
	
	# Layer 1: Shaded Area (Step-function interpolation)
	area = base.mark_area(
		opacity=0.25,
		color='#D6EAF8',
		interpolate='step-after'  # Clinical standard for survival curves
	)
	
	# Layer 2: Survival Line
	line = base.mark_line(
		color='#2E86C1',
		strokeWidth=2.5,
		interpolate='step-after'
	)
	
	layers = [area, line]
	
	# Optional Layer 3: Highlight Key Clinical Timepoints
	if highlight_times:
		max_time = data["Time"].max()
		valid_highlights = [t for t in highlight_times if t <= max_time]
		
		points = base.mark_circle(
			size=150,
			color='#C0392B',  # Red markers for emphasis
			opacity=1
		).transform_filter(
			alt.FieldOneOfPredicate(field='Time', oneOf=valid_highlights)
		)
		layers.append(points)
	
	# 3. Chart Configuration & Display
	# Combine layers and apply professional styling
	chart = alt.layer(*layers).properties(
		# title='Individualized Prognostic Survival Trajectory',
		height=500,
		width='container'
	).configure_axis(
		labelFontSize=16,
		titleFontSize=18,
		labelColor='#333333',
		titleColor='#000000',
		grid=True,
		gridColor='#D0D0D0',
		gridOpacity=0.6,
		gridDash=[2, 2],
		domain=True,
		domainColor='#000000',
		domainWidth=1.5,
	).interactive()  # Allows zooming and panning
	
	st.markdown("### 📈 Prognostic Survival Curve")
	st.caption(
		"💡 **Instructions:** Hover over the curve to view precise probabilities. "
		"Use the mouse wheel to zoom and drag to pan."
	)
	st.altair_chart(chart)
