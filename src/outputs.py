"""
This project is supported by the Department of Geriatrics and the National Clinical Research Center for Geriatrics,
West China Hospital, Sichuan University, China.
"""

import altair as alt
import pandas as pd
import streamlit as st

from src.prediction import ensure_cumulative_risk_0_time, risk_at_time


def show_model_information() -> None:
	"""Explains the model and its five inputs without coefficient plots."""
	with st.expander("ℹ️ About the Model and the Five Measurements"):
		st.markdown("""
		This tool estimates the probability of developing **sarcopenia within
		1-7 years** from baseline measurements. It uses a **Fine-Gray model**
		that accounts for death as a competing event: a person who dies can no
		longer develop sarcopenia afterwards.

		**Only five measurements are needed:**

		- **Age:** age at the baseline assessment, in years.
		- **Sex:** male or female.
		- **Mid-upper arm circumference:** the upper-arm measurement, in centimeters.
		- **Hip circumference:** the measurement around the widest part of the buttocks, in centimeters.
		- **Calf circumference:** the maximum calf measurement, in centimeters.

		The five measurements are considered together. The displayed percentage
		is an estimated cumulative probability. For example, a 5-year estimate
		of 10% means an estimated 10 in 100 people with this profile would
		develop sarcopenia within five years, accounting for competing death.
		""")


def show_risk_metrics(
		cumulative_risk: pd.Series,
		eval_times: list[int] | None,
) -> None:
	"""Displays absolute risk percentages and 0-100% progress bars.

	Args:
		cumulative_risk: Averaged imputation-specific cumulative incidences.
		eval_times: Selected prediction horizons in years.
	"""
	if cumulative_risk.empty or not eval_times:
		return
	st.markdown("### 📊 Sarcopenia Risk Assessment")
	st.caption("Each bar represents cumulative risk on a 0%-100% scale.")
	years = sorted(set(eval_times))
	# Use the same column count in every row; leave unused final-row cells empty.
	columns_per_row = min(4, len(years))
	for offset in range(0, len(years), columns_per_row):
		row_years = years[offset:offset + columns_per_row]
		for column, year in zip(st.columns(columns_per_row), row_years):
			with column:
				probability = risk_at_time(cumulative_risk, year)
				st.metric(label=f"{year}-Year Cumulative Risk", value=f"{probability:.2%}")
				st.progress(probability)


def show_altair_risk_chart(
		cumulative_risk: pd.Series,
		highlight_times: list | None = None,
) -> None:
	"""Renders the cumulative-risk step curve with interactive probability tooltips.

	Args:
		cumulative_risk: Averaged Fine-Gray cumulative-incidence curve.
		highlight_times: Prediction years to mark explicitly on the curve.
	"""
	curve = ensure_cumulative_risk_0_time(cumulative_risk)
	data = pd.DataFrame({"Time": curve.index, "Cumulative Risk": curve.to_numpy()})
	encoding = {
		"x": alt.X("Time:Q", title="Time from Baseline (Years)",
			axis=alt.Axis(tickMinStep=1, format=".0f", grid=False)),
		"y": alt.Y("Cumulative Risk:Q", title="Cumulative Risk of Sarcopenia",
			scale=alt.Scale(domain=[0, 1]), axis=alt.Axis(format=".0%")),
		"tooltip": [
			alt.Tooltip("Time:Q", title="Time (Years)", format=".1f"),
			alt.Tooltip("Cumulative Risk:Q", title="Cumulative Risk", format=".2%"),
		],
	}
	base = alt.Chart(data).encode(**encoding)
	area = base.mark_area(opacity=0.20, color="#D6EAF8", interpolate="step-after")
	line = base.mark_line(color="#2E86C1", strokeWidth=2.5, interpolate="step-after")
	layers = [area, line]
	if highlight_times:
		# Explicit point data also marks years without an event-time jump, e.g. year 2.
		points = pd.DataFrame({
			"Time": sorted(set(highlight_times)),
			"Cumulative Risk": [risk_at_time(curve, year) for year in sorted(set(highlight_times))],
		})
		layers.append(alt.Chart(points).mark_circle(size=130, color="#C0392B", opacity=1).encode(**encoding))
	chart = alt.layer(*layers).properties(height=500, width="container").configure_axis(
		labelFontSize=16, titleFontSize=18, labelColor="#333333", titleColor="#000000",
		grid=True, gridColor="#D0D0D0", gridOpacity=0.6, gridDash=[2, 2],
		domain=True, domainColor="#000000", domainWidth=1.5,
	).interactive()
	st.markdown("### 📈 Cumulative Risk of Sarcopenia")
	st.caption("Hover to view probabilities; scroll to zoom and drag to pan. Shading indicates the area under the curve.")
	st.altair_chart(chart)
