"""
This project is supported by the Department of Geriatrics and the National Clinical Research Center for Geriatrics,
West China Hospital, Sichuan University, China.
"""

import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from typing import Any
from pathlib import Path


def check_data_plausibility(personal_data: dict, feature_bounds: dict) -> list[str]:
	"""Identifies input features that fall outside the model's training distribution.

	In clinical predictive modeling, out-of-distribution (OOD) inputs can lead to
	unreliable 'extrapolated' predictions. This utility performs a plausibility
	check by comparing current patient metrics against the union of boundaries
	derived from the imputed training cohorts.

	Args:
		personal_data (dict): Dictionary containing the patient's clinical
			input features (e.g., age, BMI, circumferences).
		feature_bounds (dict): Metadata containing the 'min' and 'max' values
			for each feature as observed in the training dataset.
			Structure: {feature_name: {'min': float, 'max': float}}.

	Returns:
		list[str]: A list of formatted strings identifying features that exceed
			training bounds, e.g., ["Age (115)", "BMI (14.2)"]. Returns an
			empty list if all inputs are within the valid range.
	"""
	warnings = []
	
	for feature, val in personal_data.items():
		# Retrieve the valid range for the current feature
		val_range = feature_bounds.get(feature)
		
		if val_range is not None and val is not None:
			# Check if the clinical metric is an outlier relative to training data
			if val < val_range['min'] or val > val_range['max']:
				warnings.append(f"{feature} ({val})")
	
	return warnings


@st.cache_resource
def load_model_assets(method_name: str = "Cox") -> dict[str, Any]:
	"""Loads trained models, preprocessors, and feature lists from the local directory.

	This function utilizes Streamlit's cache mechanism to ensure that heavy model
	assets are only loaded into memory once during the application lifecycle,
	significantly improving performance for concurrent users.

	Args:
		method_name: The subdirectory name within 'models' containing the
			specific model assets. Defaults to "Cox".

	Returns:
		A dictionary containing three keys:
			- "features": List of feature names required by the model.
			- "preprocessors": List of fitted ColumnTransformer objects.
			- "models": List of trained survival analysis models (e.g., CoxPHFitter).
	"""
	# Obtain the absolute path of the current script
	current_file = Path(__file__).resolve()
	
	# Locate the project root directory (assuming this script is inside 'src/')
	project_root = current_file.parent.parent
	base_path = project_root / 'models' / method_name
	
	# Safety check: Verify if the model directory exists to prevent FileNotFoundError
	if not base_path.exists():
		st.error(f"❌ Model path not found: {base_path}")
		st.stop()
	
	# Load feature names from the text file (tab-separated)
	with open(base_path / 'final_model_features.txt', 'r', encoding='utf-8') as f:
		features = f.read().strip().split('\t')
	
	# Load serialized assets using joblib
	assets = {
		"features": features,
		"preprocessors": joblib.load(base_path / 'final_feature_preprocessors.joblib'),
		"models": joblib.load(base_path / 'final_models.joblib')
	}
	return assets


def load_thresholds(method_name: str = "Cox") -> dict[str, float]:
	"""Loads clinical risk thresholds from a local configuration file.

	This utility function retrieves pre-defined cut-off values (e.g., quintiles or
	clinically validated thresholds) from a JSON file. This ensures that
	stratification logic remains decoupled from the core UI code for easier
	maintenance and updates.

	Args:
		method_name: The name of the specific model or algorithm whose
			thresholds are being requested. Defaults to "Cox".

	Returns:
		A dictionary containing threshold constants such as:
			- "low_risk": The upper bound for the low-risk category.
			- "high_risk": The lower bound for the high-risk category.
			- "max_display_rr": The maximum value for the visual progress bar.
	"""
	# Resolve the path to the config directory relative to this script
	config_path = Path(__file__).resolve().parent.parent / 'config' / 'thresholds.json'
	
	try:
		with open(config_path, 'r', encoding='utf-8') as f:
			config_data = json.load(f)
			return config_data.get(method_name, {})
	except FileNotFoundError:
		st.error(f"❌ Configuration file not found at {config_path}. Using default empty thresholds.")
		return {}


def cal_single_person_surv_func(
		personal_data_dict: dict,
		assets: dict
) -> tuple[pd.Series, float, str]:
	"""Calculates individualized survival probability and clinical risk stratification.

	This function executes a multi-stage ensemble prediction pipeline designed for
	robustness in clinical settings. It integrates multiple cross-validated Cox-based
	models to mitigate bias and provide a consensus survival estimate.

	The pipeline involves:
	1.  **Plausibility Verification**: Validates if input characteristics are within the
		distribution of the training cohort to ensure predictive reliability.
	2.  **Schema Alignment**: Maps raw clinical inputs to standardized feature vectors.
	3.  **Ensemble Inference**: Iteratively applies fold-specific preprocessors and
		survival estimators to generate a distribution of results.
	4.  **Consensus Aggregation**: Computes arithmetic means for survival functions
		S(t) and Partial Hazards (RR) to produce a unified clinical indicator.
	5.  **Risk Categorization**: Stratifies patients based on clinically validated
		relative risk (RR) thresholds.

	Args:
		personal_data_dict (dict): A collection of patient baseline characteristics
			(e.g., age, BMI, circumferences) obtained from clinical assessment.
		assets (dict): A structured container of trained model components including:
			- "features" (list): The deterministic order of predictors.
			- "preprocessors" (list): Fitted ColumnTransformers for normalization.
			- "models" (list): Trained survival estimators (e.g., CoxPHFitter).

	Returns:
		tuple: A tri-element tuple containing:
			- avg_survival_func (pd.Series): The ensemble-averaged survival function.
			  Index represents time points (Years); values represent survival probability [0, 1].
			- relative_risk (float): The consensus Partial Hazard (exp(βx)), representing
			  the individual's risk multiplier relative to the population mean.
			- status_text (str): Qualitative clinical risk category ("Low", "Moderate", or "High Risk").
	"""
	
	# ================= 0. Data Plausibility Check (Out-of-Distribution Detection) =================
	try:
		# Define path to the pre-calculated training data distribution boundaries
		config_path = Path(__file__).resolve().parent.parent / 'config' / 'feature_bounds.json'
		
		with open(config_path, 'r', encoding='utf-8') as f:
			feature_bounds = json.load(f)
			# Identify metrics exceeding the model's validated range to warn users of uncertainty
			out_of_bounds_features = check_data_plausibility(personal_data_dict, feature_bounds)
			if out_of_bounds_features:
				st.warning(
					f"⚠️ **Caution:** The following inputs are outside the model's core validation range: "
					f"{', '.join(out_of_bounds_features)}. "
					"The prediction results may have increased uncertainty."
				)
	except FileNotFoundError:
		# Fallback if configuration is missing; proceed with prediction but without OOD warning
		pass
	
	# ================= 1. Resource Initialization =================
	# Extract structural assets required for the ensemble pipeline
	features = assets["features"]
	preprocessors = assets["preprocessors"]
	models = assets["models"]
	
	# ================= 2. Data Preprocessing =================
	# Construct a single-row DataFrame ensured to match the model's feature schema
	person_data_df = pd.DataFrame([personal_data_dict])[features]
	
	all_survival_funcs: list[pd.DataFrame] = []
	risk_scores: list[float] = []
	
	# ================= 3. Iterative Prediction (Ensemble) =================
	# Iterate through cross-validation folds to derive a robust consensus estimate
	for model, preprocessor in zip(models, preprocessors):
		
		# A. Feature Transformation: Scaling and categorical encoding
		X_processed_values = preprocessor.transform(person_data_df)
		
		# B. Schema Reconstruction: Mapping NumPy arrays back to labeled DataFrames
		cols = preprocessor.get_feature_names_out()
		X_processed = pd.DataFrame(X_processed_values, columns=cols)
		
		# C. Feature Re-alignment: Ensuring exact column ordering for position-dependent models
		try:
			X_final = X_processed[features]
		except KeyError as e:
			st.error(f"❌ Schema Mismatch: Preprocessor output does not match required features. {e}")
			st.stop()
		
		# D. Survival Function Prediction: Estimating individualized survival curves
		all_survival_funcs.append(model.predict_survival_function(X_final))
		
		# E. Partial Hazard Calculation: Obtaining the exponentiated risk score (RR)
		risk_scores.append(model.predict_partial_hazard(X_final).item())
	
	# ================= 4. Result Ensemble & Aggregation =================
	# Perform arithmetic averaging across all folds to stabilize predictive variance
	avg_survival_func: pd.Series = pd.concat(all_survival_funcs, axis=1).mean(axis=1)
	
	# Aggregate relative risk scores into a single clinical metric
	relative_risk = float(np.mean(risk_scores))
	
	# ================= 5. Clinical Risk Stratification =================
	# Categorize the patient based on pre-defined clinical hazard thresholds
	thresholds = load_thresholds()
	LOW_RISK_VAL = thresholds.get("low_risk", 0.6)
	HIGH_RISK_VAL = thresholds.get("high_risk", 1.6)
	
	if relative_risk < LOW_RISK_VAL:
		status_text = "Low Risk"
	elif relative_risk > HIGH_RISK_VAL:
		status_text = "High Risk"
	else:
		status_text = "Moderate Risk"
	
	return avg_survival_func, relative_risk, status_text


def cal_probability_at_time(
		survival_func: pd.Series,
		time: int | float
) -> tuple[float, float]:
	"""Safely extracts health and disease probabilities at a specific time point.

	This function calculates the survival probability (health) and the complement
	cumulative incidence (disease risk) from the survival curve. It handles
	potential missing time points by using a 'look-back' approach.
	
	Calculates:
    - Survival Probability S(t): Prob(T > t)
    - Cumulative Incidence F(t): Prob(T <= t) = 1 - S(t)

	Args:
		survival_func: The individualized survival function where
			the index is time and values are survival probabilities.
		time: The specific time point (e.g., years) to evaluate.

	Returns:
		tuple: A tuple containing:
			- prob_surv: Survival probability at the given time.
			- prob_disease: Probability of disease occurrence (1 - survival).
	"""
	
	# Obtain survival probability; disease probability is defined as 1 - S(t).
	# .asof(time) retrieves the last available value at or before the specified time.
	# If the requested time is before the first recorded data point (e.g., Day 0),
	# the survival probability defaults to 1.0 (100%).
	prob_surv = survival_func.asof(time)
	if pd.isna(prob_surv):
		prob_surv = 1.0
	
	return prob_surv, 1 - prob_surv


def ensure_survival_func_0_time(survival_func: pd.Series) -> pd.Series:
	"""Ensures the survival curve includes the baseline origin (t=0, p=1.0).

	In clinical survival analysis, the logical starting point for all subjects
	is a 100% survival probability. This function checks for the presence of
	the t=0 index and prepends it if missing to ensure proper visualization
	and calculation.

	Args:
		survival_func: The individualized survival function.

	Returns:
		The survival function with the baseline origin ensured.
	"""
	
	# Check if the baseline time point (0.0) exists in the index.
	if 0 not in survival_func.index:
		# Prepend a survival probability of 1.0 at t=0 and re-sort the index.
		survival_func = pd.concat([
			pd.Series([1.0], index=[0.0]),
			survival_func
		]).sort_index()
	
	return survival_func
