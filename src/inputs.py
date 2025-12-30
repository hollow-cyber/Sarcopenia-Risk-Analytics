"""
This project is supported by the Department of Geriatrics and the National Clinical Research Center for Geriatrics,
West China Hospital, Sichuan University, China.
"""

import streamlit as st


def get_user_input_sidebar() -> tuple[dict[str, str | int | float | list[int]], bool]:
	"""Renders the sidebar for clinical parameter input and returns the collected data.

	This function organizes inputs into logical sections (Demographics, Anthropometrics,
	and Circumferences) and performs real-time BMI calculation.

	Returns:
		tuple: A tuple containing:
			- user_data (dict): Dictionary of clinical features for the model.
			- all_filled (bool): True if all required fields are completed.
	"""
	
	# ================= Part 1: Basic Demographics =================
	st.sidebar.subheader("👤 Demographics")
	
	user_id = st.sidebar.text_input(
		"User ID / Medical Record No. (Optional):",
		value="",
		placeholder="e.g., P2025122901",
		help="Enter a unique identifier to distinguish this report from others."
	)
	
	age = st.sidebar.number_input(
		"Age (Years):",
		min_value=50, max_value=120, value=None,
		placeholder="e.g., 72",
		help="Please enter the patient's chronological age."
	)
	
	col1, col2 = st.sidebar.columns(2)
	with col1:
		sex_label = st.radio(
			"Sex:",
			["Male", "Female"],
			index=None,
			horizontal=True
		)
		# Mapping logic
		sex = 1 if sex_label == "Male" else (2 if sex_label == "Female" else None)
	
	with col2:
		smoker_label = st.radio(
			"Smoking Status:",
			["Yes", "No"],
			index=None,
			horizontal=True,
			help="Current smoking habit."
		)
		current_smoker = 1 if smoker_label == "Yes" else (0 if smoker_label == "No" else None)
	
	st.sidebar.divider()
	
	# ================= Part 2: Core Anthropometrics =================
	st.sidebar.subheader("📏 Anthropometrics")
	
	# Height and Weight side-by-side
	c1, c2 = st.sidebar.columns(2)
	with c1:
		height = st.number_input(
			"Height (cm)",
			min_value=100.0, max_value=250.0, step=0.1, value=None,
			format="%.1f"
		)
	with c2:
		weight = st.number_input(
			"Weight (kg)",
			min_value=20.0, max_value=300.0, step=0.01, value=None,
			format="%.2f"
		)
	
	# --- Real-time BMI Calculation ---
	bmi = None
	if height is not None and weight is not None:
		# Note: Convert height from cm to meters for BMI formula
		bmi = weight / ((height / 100) ** 2)
		
		# Display feedback to user
		if 10 <= bmi <= 60:
			st.sidebar.info(f"📊 Calculated BMI: **{bmi:.2f}**")
		else:
			st.sidebar.warning(f"⚠️ Calculated BMI (**{bmi:.2f}**) appears unusual. Please check inputs.")
	else:
		st.sidebar.caption("👉 BMI will be calculated automatically.")
	
	st.sidebar.divider()
	
	# ================= Part 3: Circumference Metrics =================
	st.sidebar.subheader("📐 Circumferences")
	
	c3, c4 = st.sidebar.columns(2)
	with c3:
		arm_circumference = st.number_input(
			"Arm Circ. (cm)",
			min_value=10.0, max_value=60.0, step=0.1, value=None,
			format="%.1f",
			help="Mid-upper arm circumference of the dominant hand."
		)
		hip_circumference = st.number_input(
			"Hip Circ. (cm)",
			min_value=30.0, max_value=300.0, step=0.1, value=None,
			format="%.1f",
			help="Circumference at the widest part of the buttocks."
		)
	with c4:
		waist_circumference = st.number_input(
			"Waist Circ. (cm)",
			min_value=30.0, max_value=300.0, step=0.1, value=None,
			format="%.1f",
			help="Circumference at the level of the umbilicus after expiration."
		)
		# Calf circumference is a key predictor for Sarcopenia
		calf_circumference = st.number_input(
			"Calf Circ. (cm)",
			min_value=10.0, max_value=70.0, step=0.1, value=None,
			format="%.1f",
			help="Maximum circumference of the dominant calf."
		)
	
	# ================= Part 4: Prediction Settings =================
	st.sidebar.divider()
	st.sidebar.subheader("⏳ Prediction Settings")
	
	select_all = st.sidebar.checkbox(
		"Evaluate all available years (1-7)",
		value=False,
		help="Toggle this to quickly select or deselect the entire 7-year follow-up period."
	)
	
	if not select_all:
		st.sidebar.caption("Or customize specific years below:")
	else:
		st.sidebar.caption("All years selected. You can still untick specific years below.")
	
	default_selection = list(range(1, 8)) if select_all else None
	
	eval_times = st.sidebar.multiselect(
		"Prediction Horizon (Years):",
		range(1, 8),
		default=default_selection,
		placeholder="Select years ahead",
		help="Select the specific years for which you want to predict the cumulative risk of sarcopenia."
	)
	
	# ================= Data Packaging =================
	# Check if all required fields are filled
	required_vals = [
		age, sex, bmi, current_smoker,
		arm_circumference, waist_circumference,
		hip_circumference, calf_circumference,
		eval_times
	]
	all_filled = all(v is not None for v in required_vals)
	
	user_data = {
		'user_id': user_id,
		'age': age,
		'sex': sex,
		'bmi': bmi,
		'current_smoker': current_smoker,
		'arm_circumference': arm_circumference,
		'waist_circumference': waist_circumference,
		'hip_circumference': hip_circumference,
		'calf_circumference': calf_circumference,
		'eval_times': eval_times,
	}
	
	return user_data, all_filled
