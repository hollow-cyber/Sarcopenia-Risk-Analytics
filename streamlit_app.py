"""
This project is supported by the Department of Geriatrics and the National Clinical Research Center for Geriatrics,
West China Hospital, Sichuan University, China.
"""

import datetime
import streamlit as st
from src.layouts import set_st_header
from src.prediction import load_model_assets, cal_single_person_surv_func
from src.inputs import get_user_input_sidebar
from src.outputs import show_risk_metrics, show_risk_stratification, show_altair_survival_chart
from src.report_generator import plot_survival_curve, generate_report_pdf


def run_st_app() -> None:
	"""
	Orchestrates the Streamlit application lifecycle for Sarcopenia Risk Prediction.

	This function handles:
	1. UI initialization and responsive design alerts.
	2. Model interpretability via forest plot visualization.
	3. Session state management for caching inference results.
	4. Synchronous prediction pipeline triggered by user submission.
	5. Dynamic rendering of clinical metrics and PDF report generation.
	"""
	
	# --- 1. UI Configuration & Header ---
	set_st_header(
		main_title="Sarcopenia Risk Analytics (SRA)",
		image_path="logo.ico",
		sidebar_title="📋 Patient Parameters"
	)
	
	# --- 2. Model Interpretability Section ---
	# Providing transparency on feature weights via an expandable Forest Plot
	with st.expander("🔍 Model Interpretation (Forest Plot) — Learn how the model calculates risk"):
		st.markdown("""
            **About this visualization:** This forest plot illustrates the contribution of
            each variable to the final prediction. Factors on the right increase
            risk (HR > 1), while those on the left are protective (HR < 1).
        """)
		# Display the static SVG forest plot from the model directory
		st.image("models/Cox/forest_plot.svg", width='stretch')
	
	# --- 3. Session State Initialization ---
	# Prevents unnecessary re-computation during Streamlit reruns
	if "inference_results" not in st.session_state:
		# Stores: survival_func, avg_rr, risk_status, and pdf_bytes
		st.session_state.inference_results = None
	
	if "frozen_params" not in st.session_state:
		# Stores a snapshot of the parameters used for the current cached result
		st.session_state.frozen_params = None
	
	# --- 4. Sidebar Input Handling ---
	# Real-time retrieval of user inputs from the sidebar
	current_input_data, is_form_complete = get_user_input_sidebar()
	
	# --- 5. Assessment Execution Logic ---
	if st.sidebar.button("🩺 Run Assessment", type="primary"):
		if not is_form_complete:
			st.error("❌ Incomplete Data: Please fill in all clinical parameters before proceeding.")
			st.stop()
		else:
			with st.spinner("Executing consensus inference...", show_time=True):
				# Load model assets (cached via @st.cache_resource)
				model_assets = load_model_assets()
				
				# Perform ensemble prediction
				survival_func, avg_rr, risk_status = cal_single_person_surv_func(
					current_input_data,
					model_assets
				)
				
				# Generate high-resolution chart for the PDF report
				chart_buffer = plot_survival_curve(
					survival_func,
					highlight_times=current_input_data.get('eval_times', [])
				)
				
				# Compose the formal clinical PDF report
				pdf_bytes = generate_report_pdf(
					current_input_data,
					survival_func,
					avg_rr,
					risk_status,
					chart_buffer
				)
				
				# Update Session State with current inference outputs
				st.session_state.inference_results = {
					"survival_func": survival_func,
					"avg_rr": avg_rr,
					"risk_status": risk_status,
					"pdf_bytes": pdf_bytes,
				}
				# Lock current inputs to sync UI with the generated report
				st.session_state.frozen_params = current_input_data.copy()
	
	# --- 6. Result Visualization & Export ---
	if st.session_state.inference_results is not None:
		results = st.session_state.inference_results
		cached_data = st.session_state.frozen_params
		
		# OOD (Out-of-Sync) Warning: Detected if user modifies inputs after assessment
		if current_input_data != cached_data:
			st.warning(
				"⚠️ **Parameters Changed:** The inputs below do not match the current prediction. Click 'Run Assessment' to refresh.")
		
		st.divider()
		
		# Render interactive clinical metrics cards
		show_risk_metrics(
			results.get("survival_func"),
			eval_times=cached_data.get('eval_times')
		)
		
		st.divider()
		
		# Render the risk stratification dashboard (Progress Bar & Tabs)
		show_risk_stratification(
			results.get("avg_rr"),
			results.get("risk_status")
		)
		
		st.divider()
		
		# Render the interactive Altair survival trajectory
		show_altair_survival_chart(
			results.get("survival_func"),
			highlight_times=cached_data.get('eval_times')
		)
		
		# Dynamic filename generation for the PDF export
		patient_id = cached_data.get('user_id')
		file_timestamp = datetime.date.today()
		if patient_id:
			export_filename = f"Sarcopenia_Risk_Report_{patient_id}_{file_timestamp}.pdf"
		else:
			export_filename = f"Sarcopenia_Risk_Report_{file_timestamp}.pdf"
		
		# Provide PDF download capability without triggering a full script rerun
		if st.download_button(
				label="📥 Download Clinical Report (PDF)",
				data=results["pdf_bytes"],
				file_name=export_filename,
				mime="application/pdf",
				type="primary"
		):
			st.toast("**Clinical Report generation: Success.** Your download has started.", icon='📄')


if __name__ == "__main__":
	run_st_app()
