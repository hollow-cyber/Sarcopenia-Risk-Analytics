"""
This project is supported by the Department of Geriatrics and the National Clinical Research Center for Geriatrics,
West China Hospital, Sichuan University, China.
"""

import datetime
import logging
import re
import streamlit as st
from typing import cast

from src.inputs import get_user_input_sidebar
from src.layouts import set_st_header
from src.outputs import show_model_information, show_risk_metrics, show_altair_risk_chart
from src.prediction import PROJECT_ROOT, check_data_plausibility, load_model_assets, predict_individual_risk
from src.report_generator import plot_cumulative_risk_curve, generate_report_pdf


def run_st_app() -> None:
	"""Runs the five-input Fine-Gray assessment, visualizations and PDF export."""
	set_st_header(
		main_title="Sarcopenia Risk Analytics (SRA)",
		image_path=PROJECT_ROOT / "logo.ico",
		sidebar_title="📋 Individual Measurements",
	)
	show_model_information()

	# Invalidate results from the previous Cox implementation during a hot reload.
	if st.session_state.get("prediction_schema") != "fine_gray_cif_v1":
		st.session_state.inference_results = None
		st.session_state.frozen_params = None
		st.session_state["prediction_schema"] = "fine_gray_cif_v1"
	current_data, is_complete = get_user_input_sidebar()

	if st.sidebar.button("🩺 Run Assessment", type="primary"):
		if not is_complete:
			st.error("❌ Please enter all five measurements and select at least one prediction year.")
		else:
			with st.spinner("Calculating cumulative risk...", show_time=True):
				try:
					assets = load_model_assets()
					outside = check_data_plausibility(current_data, assets["feature_bounds"])
					cumulative_risk = predict_individual_risk(current_data, assets)
					highlight_times = cast(list[int], current_data["eval_times"])
					chart_buffer = plot_cumulative_risk_curve(
						cumulative_risk, highlight_times=highlight_times,
					)
					pdf_bytes = generate_report_pdf(current_data, cumulative_risk, chart_buffer, out_of_bounds=outside)
				except Exception as error:
					logging.exception("Fine-Gray assessment failed")
					st.session_state.inference_results = None
					st.session_state.frozen_params = None
					st.error(f"❌ Unable to complete the assessment: {error}")
					return
				st.session_state.inference_results = {
					"cumulative_risk": cumulative_risk,
					"pdf_bytes": pdf_bytes,
					"outside_bounds": outside,
				}
				st.session_state.frozen_params = current_data.copy()

	if st.session_state.inference_results is not None:
		results = st.session_state.inference_results
		cached_data = st.session_state.frozen_params
		if current_data != cached_data:
			st.warning("⚠️ Inputs have changed. The results below use the previous inputs; click 'Run Assessment' to refresh.")
		if results["outside_bounds"]:
			st.warning(
				"⚠️ These measurements are outside the development-cohort range: "
				+ ", ".join(results["outside_bounds"])
				+ ". Predictions for these measurements may be less reliable."
			)
		st.divider()
		show_risk_metrics(results["cumulative_risk"], cached_data["eval_times"])
		st.divider()
		show_altair_risk_chart(results["cumulative_risk"], highlight_times=cached_data["eval_times"])

		# Keep filenames portable even when a record identifier contains punctuation.
		identifier = re.sub(r"[^\w.-]+", "_", cached_data.get("user_id", "")).strip("._")
		today = datetime.date.today()
		filename = f"Sarcopenia_Risk_Report_{identifier}_{today}.pdf" if identifier else f"Sarcopenia_Risk_Report_{today}.pdf"
		if st.download_button(
			label="📥 Download Risk Report (PDF)", data=results["pdf_bytes"],
			file_name=filename, mime="application/pdf", type="primary",
		):
			st.toast("Your report download has started.", icon="📄")


if __name__ == "__main__":
	run_st_app()
