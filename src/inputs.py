"""
This project is supported by the Department of Geriatrics and the National Clinical Research Center for Geriatrics,
West China Hospital, Sichuan University, China.
"""

import streamlit as st


def _select_all_years() -> None:
	"""Synchronizes the prediction-year selector when the all-years box changes."""
	st.session_state["eval_times"] = list(range(1, 8)) if st.session_state["select_all_years"] else []


def get_user_input_sidebar() -> tuple[dict[str, str | int | float | list[int] | None], bool]:
	"""Renders the five baseline predictors and prediction-year settings.

	Returns:
		The current input dictionary and whether all five measurements and at
		least one prediction year have been supplied.
	"""
	# ================= Part 1: Basic Demographics =================
	st.sidebar.subheader("👤 Demographics")
	user_id = st.sidebar.text_input(
		"User ID / Medical Record No. (Optional):",
		value="", placeholder="e.g., P2026103001",
		help="Enter an identifier to distinguish this report from others.",
	)
	age = st.sidebar.number_input(
		"Age (Years):", min_value=50, max_value=120, value=None,
		placeholder="e.g., 72", help="Age at the baseline assessment.",
	)
	sex_label = st.sidebar.radio("Sex:", ["Male", "Female"], index=None, horizontal=True)
	sex = 1 if sex_label == "Male" else (2 if sex_label == "Female" else None)

	# ================= Part 2: Circumference Measurements =================
	st.sidebar.divider()
	st.sidebar.subheader("📐 Circumferences")
	arm_circumference = st.sidebar.number_input(
		"Mid-upper Arm Circ. (cm):",
		min_value=10.0, max_value=100.0, step=0.1, value=None, format="%.1f",
		placeholder="please enter a number",
		help="Measure the circumference midway between the acromion (shoulder tip) and olecranon (elbow tip), with the arm hanging naturally, in cm.",
	)
	hip_circumference = st.sidebar.number_input(
		"Hip Circ. (cm):",
		min_value=30.0, max_value=300.0, step=0.1, value=None, format="%.1f",
		placeholder="please enter a number",
		help="Stand upright with legs together and arms relaxed; measure horizontally around the hips at the pubic symphysis in front and the fullest part of the buttocks behind, in cm."
	)
	calf_circumference = st.sidebar.number_input(
		"Calf Circ. (cm):",
		min_value=10.0, max_value=100.0, step=0.1, value=None, format="%.1f",
		placeholder="please enter a number",
		help="Measure the widest part of the calf with a flexible tape while standing balanced with knees straight, in cm.",
	)

	# ================= Part 3: Prediction Settings =================
	st.sidebar.divider()
	st.sidebar.subheader("⏳ Prediction Settings")
	st.sidebar.checkbox(
		"Evaluate all available years (1-7)", value=False,
		key="select_all_years", on_change=_select_all_years,
		help="Select all years, then untick individual years if needed.",
	)
	eval_times = st.sidebar.multiselect(
		"Prediction Horizon (Years):", list(range(1, 8)),
		key="eval_times", placeholder="Select years ahead",
		help="Select at least one year for the cumulative risk assessment.",
	)

	# Check both missing measurements and an empty year selection.
	required_values = [age, sex, arm_circumference, hip_circumference, calf_circumference]
	all_filled = all(value is not None for value in required_values) and bool(eval_times)
	user_data = {
		"user_id": user_id,
		"age": age,
		"sex": sex,
		"arm_circumference": arm_circumference,
		"hip_circumference": hip_circumference,
		"calf_circumference": calf_circumference,
		"eval_times": eval_times,
	}
	return user_data, all_filled
