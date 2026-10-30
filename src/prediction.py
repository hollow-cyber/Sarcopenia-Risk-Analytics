"""
This project is supported by the Department of Geriatrics and the National Clinical Research Center for Geriatrics,
West China Hospital, Sichuan University, China.
"""

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st

# One shared definition keeps the sidebar, model input and PDF labels consistent.
# The numeric order used for inference always comes from the saved model bundle.
FEATURE_LABELS = {
	"age": "Age",
	"sex": "Sex",
	"arm_circumference": "Mid-upper arm circumference",
	"hip_circumference": "Hip circumference",
	"calf_circumference": "Calf circumference",
}
CONTINUOUS_FEATURES = ("age", "arm_circumference", "hip_circumference", "calf_circumference")
MAX_PREDICTION_YEAR = 7.0
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def extract_scaling_parameters(
		preprocessor: Any,
		features: list[str],
) -> dict[str, tuple[float, float]]:
	"""Extracts the fitted scaling parameters for the four continuous predictors.

	The saved ColumnTransformer expects all original development columns. Its
	StandardScaler transforms each column independently, so the selected columns
	can be transformed exactly using their saved mean_ and scale_. Sex is passed
	through unchanged: male=1 and female=2. No parameters are fitted here.

	Args:
		preprocessor: A fitted development-cohort ColumnTransformer.
		features: Feature names in the model bundle's required order.

	Returns:
		A mapping from continuous feature to its fitted (mean, scale).

	Raises:
		ValueError: The preprocessor does not match the supported five-input model.
	"""
	if not hasattr(preprocessor, "transformers_"):
		raise ValueError("The saved preprocessor is not a fitted ColumnTransformer.")
	entries = [item for item in preprocessor.transformers_ if item[0] == "scaler"]
	if len(entries) != 1:
		raise ValueError("Each preprocessor must contain one fitted 'scaler' transformer.")
	_, scaler, selector = entries[0]
	input_names = list(preprocessor.feature_names_in_)
	if isinstance(selector, slice):
		columns = input_names[selector]
	else:
		columns = [value if isinstance(value, str) else input_names[int(value)] for value in selector]
	if not hasattr(scaler, "mean_") or not hasattr(scaler, "scale_"):
		raise ValueError("The scaler must expose fitted mean_ and scale_ arrays.")
	if not scaler.with_mean or not scaler.with_std:
		raise ValueError("The current model requires centering and standard-deviation scaling.")
	if "sex" in columns or not set(CONTINUOUS_FEATURES).issubset(columns):
		raise ValueError("The scaler must scale the four continuous predictors and leave sex unchanged.")
	if set(features) != set(FEATURE_LABELS):
		raise ValueError("The bundle must contain the five supported predictors.")
	output_names = list(preprocessor.get_feature_names_out())
	if len(output_names) != len(set(output_names)) or not set(features).issubset(output_names):
		raise ValueError("Preprocessor output does not contain unique, unprefixed model feature names.")
	
	# Reject a changed categorical transformation rather than silently treating
	# encoded sex values as the original study codes.
	for name, transformer, selected in preprocessor.transformers_:
		if name == "scaler" or isinstance(transformer, str):
			continue
		if isinstance(selected, slice):
			selected_names = input_names[selected]
		else:
			selected_names = [value if isinstance(value, str) else input_names[int(value)] for value in selected]
		if "sex" in selected_names and name != "remainder":
			raise ValueError("Sex must be passed through without categorical encoding.")
	
	means = np.asarray(scaler.mean_, dtype=float)
	scales = np.asarray(scaler.scale_, dtype=float)
	if len(columns) != means.size or means.shape != scales.shape:
		raise ValueError("Saved scaler columns, means and standard deviations do not match.")
	parameters = {}
	for feature in CONTINUOUS_FEATURES:
		index = columns.index(feature)
		mean, scale = float(means[index]), float(scales[index])
		if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
			raise ValueError(f"Invalid saved scaling parameters for {feature}.")
		parameters[feature] = (mean, scale)
	return parameters


@st.cache_resource
def load_model_assets(model_dir: str | Path | None = None) -> dict[str, Any]:
	"""Loads the five final Fine-Gray models and their matching preprocessors.

	Args:
		model_dir: Model asset directory; defaults to the project's models folder.

	Returns:
		Models, ordered features, fitted scaling parameters and predictor bounds.

	Raises:
		ValueError: The model role, feature schema or fitted model state is invalid.
		FileNotFoundError: A required bundle or matching preprocessor is absent.
	"""
	base_path = PROJECT_ROOT / "models" if model_dir is None else Path(model_dir)
	bundle = joblib.load(base_path / "final_fine_gray_prediction_bundle.joblib")
	if not isinstance(bundle, dict):
		raise ValueError("The Fine-Gray prediction bundle must be a dictionary.")
	if bundle.get("model_role") != "final model refitted on the full development cohort":
		raise ValueError("Please use the final models fitted on the full development cohort.")
	if bundle.get("training_population") != "full development cohort":
		raise ValueError("The bundle's training population is not the full development cohort.")
	models = list(bundle["models"])
	features = list(bundle["features"])
	if len(models) != 5 or bundle.get("n_imputations") != 5:
		raise ValueError("Exactly five imputation-specific final models are required.")
	if len(features) != 5 or set(features) != set(FEATURE_LABELS):
		raise ValueError("The model feature list does not match the five-input interface.")
	
	scaling_parameters = []
	for number, model in enumerate(models, start=1):
		if not callable(getattr(model, "predict_cumulative_incidence", None)):
			raise ValueError(f"Model {number} does not support cumulative-incidence prediction.")
		if int(model.n_features_in_) != len(features) or int(model.cause) != 1:
			raise ValueError(f"Model {number} has an incompatible input schema or event of interest.")
		if not bool(model.converged_) or not np.isfinite(model.coef_).all():
			raise ValueError(f"Model {number} did not converge or contains invalid coefficients.")
		path = base_path / "scalers" / f"scalers_{number}.joblib"
		preprocessor = joblib.load(path)
		scaling_parameters.append(extract_scaling_parameters(preprocessor, features))
	
	bounds_path = base_path / "feature_bounds.json"
	if bounds_path.is_file():
		with bounds_path.open("r", encoding="utf-8-sig") as file:
			bounds = json.load(file)
		bounds = {feature: bounds[feature] for feature in features if feature in bounds}
	else:
		bounds = {}
	return {
		"models": models,
		"features": features,
		"scaling_parameters": scaling_parameters,
		"feature_bounds": bounds,
	}


def check_data_plausibility(
		personal_data: dict,
		feature_bounds: dict,
) -> list[str]:
	"""Lists selected predictors outside their development-cohort ranges.

	Args:
		personal_data: Raw, unscaled patient measurements.
		feature_bounds: Saved lower and upper bounds indexed by feature name.

	Returns:
		Human-readable descriptions of out-of-range measurements.
	"""
	outside = []
	for feature, label in FEATURE_LABELS.items():
		value = personal_data.get(feature)
		bounds = feature_bounds.get(feature)
		if value is not None and bounds is not None:
			if value < bounds["min"] or value > bounds["max"]:
				outside.append(f"{label} ({value:g})")
	return outside


def transform_selected_features(
		personal_data: dict[str, Any],
		features: list[str],
		scaling_parameters: dict[str, tuple[float, float]],
) -> np.ndarray:
	"""Transforms the five raw inputs using one saved imputation's parameters.

	Args:
		personal_data: Five raw measurements using male=1 and female=2.
		features: Required numeric column order from the bundle.
		scaling_parameters: Fitted means and standard deviations for one model.

	Returns:
		A finite float matrix with shape (1, 5).
	"""
	values = []
	for feature in features:
		value = personal_data.get(feature)
		if value is None:
			raise ValueError(f"Please provide {FEATURE_LABELS[feature]}.")
		value = float(value)
		if not np.isfinite(value):
			raise ValueError(f"{FEATURE_LABELS[feature]} must be a finite number.")
		if feature == "sex":
			if value not in (1.0, 2.0):
				raise ValueError("Sex must be coded as male=1 or female=2.")
		else:
			if value <= 0:
				raise ValueError("Measurements must be positive.")
			if feature == "age" and value < 50:
				raise ValueError("Age must be at least 50 years.")
			mean, scale = scaling_parameters[feature]
			value = (value - mean) / scale
		values.append(value)
	return np.asarray([values], dtype=float)


def predict_individual_risk(
		personal_data: dict[str, Any],
		assets: dict[str, Any],
) -> pd.Series:
	"""Averages the five imputation-specific cumulative-incidence predictions.

	Each model receives inputs transformed using its matching development-cohort
	scaler. Averaging is performed on probabilities, after model prediction.

	Args:
		personal_data: Baseline measurements and optional selected evaluation years.
		assets: The validated dictionary returned by load_model_assets().

	Returns:
		A series of cumulative sarcopenia risks indexed by years, including 0 and 7.

	Raises:
		ValueError: A requested year or predicted probability is invalid.
	"""
	models, features = assets["models"], assets["features"]
	parameters = assets["scaling_parameters"]
	if len(models) != 5 or len(parameters) != 5:
		raise ValueError("Five models and five matching sets of scaling parameters are required.")
	requested_times = np.asarray(personal_data.get("eval_times", []), dtype=float)
	if requested_times.ndim != 1 or not np.isfinite(requested_times).all():
		raise ValueError("Prediction years must be a finite one-dimensional list.")
	if np.any((requested_times < 0) | (requested_times > MAX_PREDICTION_YEAR)):
		raise ValueError("Prediction is supported only from baseline through year 7.")
	
	# All annual times are present, including years with no baseline-hazard jump.
	# Preserve any additional saved event times so the displayed step curve is exact.
	grid_parts = [np.arange(0.0, MAX_PREDICTION_YEAR + 1.0), requested_times]
	for model in models:
		event_times = np.asarray(model._state.baseline_times, dtype=float)
		if not np.isfinite(event_times).all():
			raise ValueError("A model contains invalid baseline event times.")
		grid_parts.append(event_times[(event_times >= 0) & (event_times <= MAX_PREDICTION_YEAR)])
	times = np.unique(np.concatenate(grid_parts))
	
	predictions = []
	for number, (model, scaling) in enumerate(zip(models, parameters), start=1):
		x = transform_selected_features(personal_data, features, scaling)
		probabilities = np.asarray(model.predict_cumulative_incidence(x, times=times), dtype=float)
		if probabilities.shape != (1, len(times)) or not np.isfinite(probabilities).all():
			raise ValueError(f"Model {number} returned an invalid probability array.")
		if np.any((probabilities < -1e-10) | (probabilities > 1.0 + 1e-10)):
			raise ValueError(f"Model {number} returned probabilities outside [0, 1].")
		if np.any(np.diff(probabilities[0]) < -1e-10) or abs(probabilities[0, 0]) > 1e-10:
			raise ValueError(f"Model {number} returned an invalid cumulative-incidence curve.")
		predictions.append(np.clip(probabilities[0], 0.0, 1.0))
	pooled = np.mean(np.stack(predictions, axis=0), axis=0)
	return pd.Series(pooled, index=times, name="Cumulative Risk")


def ensure_cumulative_risk_0_time(cumulative_risk: pd.Series) -> pd.Series:
	"""Validates a cumulative-incidence curve and ensures its origin is (0, 0)."""
	curve = cumulative_risk.astype(float).copy().sort_index()
	curve.index = curve.index.astype(float)
	if curve.empty or curve.index.has_duplicates:
		raise ValueError("The cumulative-risk curve is empty or contains duplicate times.")
	if not np.isfinite(curve.index).all() or not np.isfinite(curve.to_numpy()).all():
		raise ValueError("The cumulative-risk curve contains invalid times or probabilities.")
	if curve.index.min() < 0 or curve.index.max() > MAX_PREDICTION_YEAR:
		raise ValueError("The cumulative-risk curve must be within 0-7 years.")
	if (curve < 0).any() or (curve > 1).any() or np.any(np.diff(curve.to_numpy()) < -1e-10):
		raise ValueError("Cumulative risks must increase monotonically within [0, 1].")
	if 0.0 in curve.index and abs(float(curve.loc[0.0])) > 1e-10:
		raise ValueError("Cumulative risk must be zero at baseline.")
	curve.loc[0.0] = 0.0
	return curve.sort_index()


def risk_at_time(
		cumulative_risk: pd.Series,
		time: int | float,
) -> float:
	"""Reads cumulative risk using the latest step at or before a supported time.

	Args:
		cumulative_risk: Model curve including baseline and the supported end time.
		time: Prediction horizon, in years.

	Returns:
		The absolute cumulative risk as a number between 0 and 1.
	"""
	query = float(time)
	curve = ensure_cumulative_risk_0_time(cumulative_risk)
	if not np.isfinite(query) or query < 0 or query > float(curve.index.max()):
		raise ValueError("The requested prediction year lies outside the available curve.")
	return float(curve.asof(query))
