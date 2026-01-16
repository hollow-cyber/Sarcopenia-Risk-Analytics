"""
This Script is Supported by Department of Geriatrics and National Clinical Research Center for Geriatrics,
West China Hospital, Sichuan University, China.
"""

import io
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from fpdf import FPDF
from PIL import Image
from pathlib import Path
from typing import Any
from src.prediction import ensure_survival_func_0_time, cal_probability_at_time


def plot_survival_curve(
		survival_func: pd.Series,
		line_style: str = 'step',
		highlight_times: list[int] | None = None,
		font_family: str | list[str] | None = None,
) -> io.BytesIO:
	"""Plots a high-resolution, individualized sarcopenia-free probability curve.

	This function generates a publication-quality visualization of the survival
	function $S(t)$, typically representing the probability of remaining event-free
	(e.g., Sarcopenia-free) over a longitudinal follow-up period.

	The visualization adheres to clinical standards:
	- **Step-function representation**: Reflects the discrete nature of survival
	  estimates.
	- **Point-of-care annotations**: Highlights specific clinical horizons (e.g.,
	  1, 3, or 5-year risk) with precise probability markers.
	- **Academic Aesthetics**: Optimized for high-DPI inclusion in clinical reports
	  or research manuscripts.

	Args:
		survival_func: The individualized survival function where the
			index is time (Years) and values are probabilities [0, 1].
		line_style: The geometric interpretation of the curve. 'step' (post-step)
			is recommended for rigorous survival analysis. 'smooth' provides a
			fluid trend visualization. Defaults to 'step'.
		highlight_times: Specific years to emphasize with
			vertical guidelines and percentage annotations. Defaults to None.
		font_family: Typographic configuration.
			Supports fallback lists for cross-platform compatibility (e.g.,
			['Arial', 'SimSun']). Defaults to None (Arial/SimSun).

	Returns:
		io.BytesIO: A memory buffer containing the generated PNG image (300 DPI).
	"""
	
	# --- 1. Data & Environment Preparation ---
	# Ensure the curve starts at (t=0, S(t)=1.0) for biological plausibility
	curve_plot = ensure_survival_func_0_time(survival_func)
	
	# Handle mutable default argument for fonts safely
	if font_family is None:
		font_family = ['Arial', 'SimSun']
	
	# Configure global Matplotlib parameters
	plt.rcParams['font.family'] = font_family
	# Critical for proper hyphen/minus rendering
	plt.rcParams['axes.unicode_minus'] = False
	
	# Initialize figure and axes
	fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
	
	# Palette definition for a professional medical aesthetic
	line_color = '#2E86C1'  # Professional Blue
	fill_color = '#D6EAF8'  # Light Blue for shaded area
	dot_color = '#C0392B'  # Contrast Red for markers
	
	# 1. Plot the Main Curve and Shaded Area
	if line_style == 'step':
		# Step-post: Rigorous survival analysis representation
		ax.step(curve_plot.index, curve_plot.values, where='post',
		        color=line_color, linewidth=3, label='Survival Probability')
		
		# Fill the area under the curve
		ax.fill_between(curve_plot.index, curve_plot.values, step='post',
		                alpha=0.2, color=fill_color)
	else:
		# Smooth line: Visually fluid for simplified trends
		ax.plot(curve_plot.index, curve_plot.values,
		        color=line_color, linewidth=3, label='Survival Probability',
		        marker='o', markersize=4)
		
		ax.fill_between(curve_plot.index, curve_plot.values,
		                alpha=0.2, color=fill_color)
	
	# 2. Dynamic Axis Configuration (Adding padding for readability)
	max_time = curve_plot.index.max()
	
	# X-Axis: Start at 0, extend 10% beyond max_time for padding
	ax.set_xlim(0, max_time * 1.1)
	# Set tick intervals to every 1 year
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	# Format ticks as integers
	ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%d'))
	
	# Y-Axis: Domain [0, 1.05] for visibility
	ax.set_ylim(0, 1.05)
	# Set tick intervals to 0.1 probability
	ax.yaxis.set_major_locator(ticker.MultipleLocator(0.1))
	# Format ticks to one decimal place
	ax.yaxis.set_major_formatter(ticker.FormatStrFormatter('%.2f'))
	
	# 3. Annotate Critical Time Points
	if highlight_times is not None:
		# Filter time points that exist within the model's projection range
		valid_highlights = [t for t in highlight_times if t <= max_time]
		
		for t in valid_highlights:
			# cal_probability_at_time is an external utility function
			prob_surv, _ = cal_probability_at_time(survival_func, t)
			
			# Draw marker points
			ax.scatter(t, prob_surv, color=dot_color, s=80, zorder=5, linewidth=2)
			
			# Add text annotation (Probability in percentage)
			ax.annotate(f'{prob_surv:.2%}', xy=(t, prob_surv), xytext=(10, 10),
			            textcoords='offset points', fontsize=13,
			            fontweight='bold', color=dot_color)
			
			# Draw vertical and horizontal dashed guidelines
			ax.vlines(t, 0, prob_surv, linestyles=':', colors='gray',
			          alpha=0.6, linewidth=1.5)
			ax.hlines(prob_surv, 0, t, linestyles=':', colors='gray',
			          alpha=0.6, linewidth=1.5)
	
	# 4. Aesthetic Refinement of Axes and Labels
	ax.set_xlabel('Time (Years)', fontsize=14, fontweight='bold', labelpad=10)
	ax.set_ylabel('Sarcopenia-Free Probability', fontsize=14, fontweight='bold', labelpad=10)
	
	# Adjust tick font sizes
	ax.tick_params(axis='both', which='major', labelsize=12)
	
	# Remove top and right spines for a clean publication-ready look
	ax.spines['top'].set_visible(False)
	ax.spines['right'].set_visible(False)
	
	# Enhance the visibility of left and bottom spines
	ax.spines['left'].set_linewidth(1.5)
	ax.spines['bottom'].set_linewidth(1.5)
	
	# 5. Add Axis Indicators (Arrows)
	# X-axis arrow
	ax.plot(1, 0, ">", transform=ax.transAxes, clip_on=False,
	        markersize=8, color='black', markeredgewidth=0)
	# Y-axis arrow
	ax.plot(0, 1, "^", transform=ax.transAxes, clip_on=False,
	        markersize=8, color='black', markeredgewidth=0)
	
	# Chart Title
	# ax.set_title('Sarcopenia-Free Probability Curve', fontsize=16, fontweight='bold', pad=20)
	
	# Adjust layout to prevent overlapping elements
	plt.tight_layout()
	
	# 6. Save Plot to Memory Buffer
	img_buf = io.BytesIO()
	fig.savefig(img_buf, format='png', dpi=300)
	# Reset buffer position to start
	img_buf.seek(0)
	# Close plot to release memory resources
	plt.close(fig)
	
	return img_buf


def format_output_value(
		value: float | int | Any
) -> str:
	"""Standardizes numerical precision for professional clinical reporting.

	This function ensures that clinical metrics are presented with appropriate
	precision:
	1. If the input is a float with more than one decimal place (e.g., BMI),
	   it is rounded to exactly two decimal places.
	2. If the input has zero or one decimal place (e.g., Age or Circumference),
	   the original precision is preserved to maintain clinical accuracy.
	3. Non-numeric types are returned as their standard string representation.

	Args:
		value: The numerical value or raw data to format.

	Returns:
		A formatted string tailored for medical documentation.
	"""
	# Pre-convert to string for type-agnostic handling
	val_str = str(value)
	
	# Process only float types to handle precision logic
	if isinstance(value, float):
		# Split to isolate the fractional part
		decimal_part = val_str.split('.')[1]
		
		# If precision exceeds one decimal place, format to two decimal places
		# f-string formatting handles standard rounding automatically
		if len(decimal_part) > 1:
			return f"{value:.2f}"
		
		# If the float is single-precision (e.g., 35.0), return as is
		return val_str
	
	# Return the original string for integers (e.g., 70) or other types
	return val_str


def read_feature_mapping(
		file_path: str | Path,
		key_col: int = 1,
		value_col: int = 2,
		sep: str = "\t"
) -> dict[str, str]:
	"""Reads a feature name mapping table from a text file.

	Used to map technical feature keys (e.g., 'bmi') to human-readable
	labels (e.g., 'Body Mass Index, kg/m²').

	Args:
		file_path: Path to the mapping file.
		key_col: Index of the column to be used as dictionary keys.
		value_col: Index of the column to be used as dictionary values.
		sep: Delimiter used in the file. Defaults to tab ("\t").

	Returns:
		A dictionary mapping technical keys to descriptive labels.
	"""
	# Read the mapping file; assumes the first row contains valid column data
	df = pd.read_csv(file_path, sep=sep)
	
	return dict(zip(df.iloc[:, key_col], df.iloc[:, value_col]))


def format_user_data_for_report(
		user_data: dict[str, Any]
) -> dict[str, str]:
	"""Normalizes raw user data into a format suitable for PDF reporting.

	This function performs three main tasks:
	1. Maps internal keys to clinical labels using an external mapping file.
	2. Converts binary categorical values (e.g., 0/1) into descriptive strings.
	3. Applies standardized numerical rounding via `format_output_value`.

	Args:
		user_data: Dictionary containing raw patient metrics.

	Returns:
		A dictionary of formatted strings with clinical labels.
	"""
	# Load mapping: Column 1 (internal key) to Column 2 (clinical label)
	feature_mapping = read_feature_mapping("feature_mapping.txt")
	
	# Initialize formatted dictionary
	formatted_data = {}
	if user_data.get('user_id'):
		formatted_data = {"User ID / Record No.": user_data.get('user_id')}
	
	for key, raw_val in user_data.items():
		# Only process keys defined in the mapping file (excluding ID which is handled)
		if key in feature_mapping and key != "user_id":
			clinical_label = feature_mapping[key]
			
			# Specialized transformation for categorical variables
			if key == "sex":
				formatted_data[clinical_label] = "Male" if raw_val == 1 else "Female"
			elif key == "current_smoker":
				formatted_data[clinical_label] = "Yes" if raw_val == 1 else "No"
			else:
				# Apply the decimal rounding logic for all other numerical metrics
				if "circumference" in clinical_label:
					formatted_data[clinical_label] = format_output_value(raw_val) + " cm"
				else:
					formatted_data[clinical_label] = format_output_value(raw_val)
	
	return formatted_data


class ReportPDF(FPDF):
	"""Custom FPDF class for standardized clinical geriatric assessment reports.

	This class extends FPDF to ensure institutional visual identity and legal
	compliance for medical reports. It automatically
	manages headers with West China Hospital branding and footers containing
	mandatory clinical disclaimers.

	Attributes:
		font (str): The primary font family used for consistent typography
			across the document.
	"""
	
	def __init__(self,
	             font_family: str = "Arial",
	             *args: Any,
	             **kwargs: Any
	             ) -> None:
		"""Initializes the PDF document with a unified typographic configuration.

		Args:
			font_family: The font name (e.g., 'Arial', 'SimSun') used as
				 the document's baseline style.
			*args, **kwargs: Standard FPDF parameters (e.g., orientation, unit, format).
		"""
		super().__init__(*args, **kwargs)
		# Unified font parameter to ensure style consistency
		self.font: str = font_family
	
	def header(self) -> None:
		"""Renders the institutional header with logo and clinical center branding.

		Executed automatically at the start of each page. It positions the
		National Clinical Research Center logo and right-aligns the center's
		full official title.
		"""
		# --- Institutional Identity ---
		# Logo: Placed at the top-left margin
		self.image(name="logo.ico", x=10, y=7, w=15)
		
		# Typography: Institutional blue with bold weight
		self.set_font(family=self.font, style="B", size=10)
		self.set_text_color(31, 119, 180)  # Clinical Blue (Primary Branding)
		
		header_text = (
			"National Clinical Research Center for Geriatrics\n"
			"West China Hospital, Sichuan University, China"
		)
		# Right-aligned multi-line text for professional header layout
		self.multi_cell(w=0, h=5, text=header_text, align="R")
		
		# Visual spacing before document content
		self.ln(5)
	
	def footer(self) -> None:
		"""Renders the medical disclaimer and dynamic page numbering.

		Executed automatically at the end of each page. This section contains
		the mandatory Cox model validation disclaimer, ensuring that predictions
		are used solely for clinical decision support.
		"""
		# Position at 25 mm from bottom for the disclaimer
		self.set_y(-25)
		
		# Italicized subtle gray font for secondary legal information
		self.set_font(family=self.font, style="I", size=8)
		self.set_text_color(150, 150, 150)
		
		disclaimer = (
			"Disclaimer: This assessment is powered by a Cox Proportional Hazards Ensemble Model "
			"validated on a longitudinal West China Hospital cohort and is for clinical decision support only. "
			"It does not constitute a formal diagnosis. Final clinical judgment remains with the physician."
		)
		self.multi_cell(w=0, h=4, text=disclaimer)
		
		# --- Page Navigation ---
		# Position at 10 mm from bottom for page numbering
		self.set_y(-10)
		self.set_font(family=self.font, size=8)
		self.cell(w=0, h=10, text=f"Page {self.page_no()}/{{nb}}", align="C")


def add_patient_metadata_table(
		pdf: ReportPDF,
		user_data: dict[str, str]
) -> None:
	"""Renders patient baseline metrics in a structured, multi-column clinical grid.

	This function transforms raw clinical input into a formatted table within the PDF
	report. It features a prioritized display for patient identifiers followed by
	a balanced two-column layout for anthropometric and demographic metrics
   .

	Args:
		pdf: The custom FPDF instance used for report generation.
		user_data: A dictionary of formatted clinical predictors
			(e.g., Age, BMI, Circumferences) where keys are human-readable labels
			and values are pre-formatted strings.
	"""
	# --- Configuration: Grid Geometry ---
	# Widths are calibrated for standard A4 portrait orientation
	label_width = 65  # Space allocated for the metric name (e.g., "Calf circumference")
	value_width = 30  # Space allocated for the metric value (e.g., "34.5 cm")
	
	# --- Data Sorting Logic ---
	# Separating Identifiers from regular metrics for visual hierarchy
	metadata_items = []
	id_label, id_value = None, None
	
	for label, val in user_data.items():
		# Institutional requirement: Unique identifiers (ID/MRN) should be prominent
		if "ID" in label:
			id_label, id_value = label, val
		else:
			metadata_items.append((label, val))
	
	# --- Section 1: Primary Identifier (Standalone Row) ---
	if id_label:
		# Bold label for emphasis
		pdf.set_font(family=pdf.font, style="B", size=10)
		pdf.cell(w=37, h=10, text=f"{id_label}:")
		
		# Regular weight for the actual ID value
		pdf.set_font(family=pdf.font, style="", size=10)
		pdf.cell(w=0, h=10, text=str(id_value), ln=1)
		
		# Aesthetic Horizontal Divider: Separates ID from baseline metrics
		pdf.set_draw_color(220, 220, 220)  # Light gray for subtle separation
		pdf.line(10, pdf.get_y(), 200, pdf.get_y())
		pdf.ln(2)
	
	# --- Section 2: Clinical Metrics (Balanced Two-Column Grid) ---
	# Iterating through items in pairs to create a dense yet readable data block
	for i in range(0, len(metadata_items), 2):
		# --- Column A Rendering ---
		pdf.set_x(10)
		pdf.set_font(family=pdf.font, style="B", size=10)
		pdf.cell(w=label_width, h=8, text=f"{metadata_items[i][0]}:")
		
		pdf.set_font(family=pdf.font, style="", size=10)
		pdf.cell(w=value_width, h=8, text=str(metadata_items[i][1]))
		
		# --- Column B Rendering (Condition-based) ---
		if i + 1 < len(metadata_items):
			# Calculate offset to maintain consistent vertical alignment
			pdf.set_x(10 + label_width + value_width)
			
			pdf.set_font(family=pdf.font, style="B", size=10)
			pdf.cell(w=label_width, h=8, text=f"{metadata_items[i + 1][0]}:")
			
			pdf.set_font(family=pdf.font, style="", size=10)
			pdf.cell(w=value_width, h=8, text=str(metadata_items[i + 1][1]))
		
		# Move to the next row of the grid
		pdf.ln(8)
	
	# Add vertical spacing before the next report section
	pdf.ln(5)


def add_pdf_chart(
		pdf: ReportPDF,
		plot_buffer: io.BytesIO,
		chart_title: str,
		width: int | float = 160,
		x_pos: int | float = 25,
		bottom_threshold: int | float = 260,
		bottom_padding: int | float = 5
) -> None:
	"""
	Integrates a high-resolution prognostic chart and its corresponding section header into the PDF
	document with automated spatial layout management.

	This utility ensures clinical-grade typesetting by executing the following logic:
	1. Performs geometric scaling of the graphical asset to maintain aspect ratio within specified
	   document constraints.
	2. Implements a look-ahead page-break validation that treats the section header and the image
	   as an atomic unit, preventing orphaned titles at page terminals.
	3. Triggers automated page instantiation and invokes institutional header/footer templates
	   as required.
	4. Executes immediate memory buffer disposal to optimize resource utilization in concurrent
	   execution environments.

	Args:
	   pdf: The active FPDF instance for report generation.
	   plot_buffer: Memory buffer containing the encoded PNG graphical asset.
	   chart_title: Descriptive nomenclature for the clinical section (e.g., "4. Predicted Survival Trajectory").
	   width: Horizontal dimension of the image in millimeters. Defaults to 160 (A4 centered).
	   x_pos: Lateral start coordinate for image rendering. Defaults to 25.
	   bottom_threshold: Vertical coordinate limit (mm) utilized as a safety boundary to
		  prevent collision with the institutional footer. Defaults to 260.
	   bottom_padding: Vertical clearance (mm) appended post-rendering to separate
		  sequential content. Defaults to 5.
	"""
	
	# --- 1. Geometric Scaling and Pre-computation ---
	# Extract native resolution to compute the aspect ratio for vertical scaling
	img = Image.open(plot_buffer)
	orig_w, orig_h = img.size
	aspect_ratio = orig_h / orig_w
	display_h = width * aspect_ratio
	
	# Define vertical height for the section header (10mm cell + 5mm spacing)
	header_unit_height = 15
	# Determine the cumulative vertical displacement required for the section
	total_required_height = display_h + header_unit_height
	
	# --- 2. Intelligent Pagination and Layout Validation ---
	# Retrieve current cursor position on the vertical axis
	current_y = pdf.get_y()
	
	# Logic: If the cumulative height of the header and image exceeds the threshold,
	# relocate the entire section to the subsequent page to ensure structural integrity.
	if current_y + total_required_height > bottom_threshold:
		# Instantiate a new page and reset the vertical origin
		pdf.add_page()
	
	# --- 3. Section Header Rendering ---
	# Render the section title with an institutional gray background for visual stratification
	pdf.set_font(family=pdf.font, style="B", size=13)
	pdf.set_fill_color(245, 245, 245)
	pdf.cell(w=0, h=10, text=f" {chart_title}", ln=True, fill=True)
	pdf.ln(5)  # Inter-element spacing between header and chart
	
	# Re-calculate the vertical coordinate for image placement post-header rendering
	render_y = pdf.get_y()
	
	# --- 4. Graphical Asset Rendering ---
	# Render the chart asset at the calculated coordinates
	pdf.image(name=plot_buffer, x=x_pos, y=render_y, w=width)
	
	# --- 5. Memory Management and Cursor Synchronization ---
	# Dispose of the image buffer immediately to optimize RAM availability
	plot_buffer.close()
	
	# Synchronize the PDF cursor to the base of the image plus the specified padding
	pdf.set_y(render_y + display_h + bottom_padding)


def generate_report_pdf(
		user_data: dict[str, Any],
		survival_func: pd.Series,
		rr_score: float,
		status_text: str,
		plot_buffer: io.BytesIO | None,
		font_name: str = "Arial"
) -> bytes:
	"""
	Orchestrates the generation of a formal, publication-quality Clinical Assessment Report.

	This high-level controller integrates multidimensional patient data, quantitative
	ensemble model outputs, and longitudinal sarcopenia-free probability visualizations
	into a standardized PDF document.
	It utilizes dynamic visual styling—calibrated to calculated risk tiers—to facilitate
	rapid clinical diagnostic interpretation.

	Args:
	   user_data: Raw clinical predictors and demographic attributes
		  retrieved from the patient interface.
	   survival_func: The individualized survival function estimated by the
		  ensemble engine.
	   rr_score: The consensus Relative Risk (Partial Hazard) score.
	   status_text: Qualitative clinical risk stratification (e.g., "High Risk").
	   plot_buffer: Memory buffer containing the high-resolution
		  Sarcopenia-Free Probability Curve. If null, the graphical section is omitted.
	   font_name: Baseline typeface utilized for institutional document consistency.
		  Defaults to "Arial".

	Returns:
	   bytes: The serialized PDF document as a byte stream, optimized for
		  web-based transmission or cloud archiving.
	"""
	# Initialize the PDF engine with institutional branding templates
	pdf = ReportPDF(font_family=font_name)
	pdf.alias_nb_pages()  # Enable dynamic page total placeholders {nb}
	pdf.add_page()
	
	# --- Section 0: Document Identification and Temporal Traceability ---
	pdf.set_font(family=pdf.font, style='B', size=18)
	pdf.cell(w=0, h=15, text="Sarcopenia Risk Assessment Report", ln=True, align='C')
	
	# Render document generation timestamp for clinical record-keeping
	pdf.set_font(family=pdf.font, size=9)
	timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
	pdf.cell(w=0, h=5, text=f"Generated on: {timestamp}", ln=True, align='C')
	pdf.ln(8)
	
	# --- Section 1: Baseline Clinical Profile ---
	# Render section header with standardized clinical gray background
	pdf.set_font(family=pdf.font, style="B", size=13)
	pdf.set_fill_color(245, 245, 245)
	pdf.cell(w=0, h=10, text=" 1. Patient Baseline Information", ln=True, fill=True)
	pdf.ln(2)
	
	# Data Normalization: Transform raw clinical keys into human-readable labels
	formatted_data = format_user_data_for_report(user_data)
	add_patient_metadata_table(pdf, formatted_data)
	
	# --- Section 2: Longitudinal Risk Projection Table ---
	pdf.set_font(family=pdf.font, style="B", size=13)
	pdf.cell(w=0, h=10, text=" 2. Sarcopenia Risk Assessment", ln=True, fill=True)
	pdf.ln(5)
	
	# Spatial Alignment: Calculate horizontal origin for table centering
	page_width = 210  # A4 standard width (mm)
	table_width = 120  # Combined width of the bi-columnar grid
	start_x = (page_width - table_width) / 2
	
	# Render Table Header with institutional styling
	pdf.set_x(start_x)
	pdf.set_font(family=pdf.font, style="B", size=11)
	pdf.set_fill_color(230, 230, 230)
	pdf.cell(60, 8, "Prediction Horizon", border=1, align='C', fill=True)
	pdf.cell(60, 8, "Cumulative Risk (%)", border=1, align='C', ln=True, fill=True)
	
	# Populate Table Rows: Iteratively compute risk probabilities for specified horizons
	pdf.set_font(family=pdf.font, style="", size=11)
	for t in sorted(user_data.get('eval_times', [])):
		# Retrieve discrete probability from the survival function
		pdf.set_x(start_x)
		_, risk_prob = cal_probability_at_time(survival_func, t)
		pdf.cell(60, 8, f"{t} Year(s)", border=1, align='C')
		pdf.cell(60, 8, f"{risk_prob:.2%}", border=1, align='C', ln=True)
	pdf.ln(5)
	
	# --- Section 3: Risk Stratification and Clinical Dashboard ---
	pdf.set_font(family=pdf.font, style="B", size=13)
	pdf.cell(w=0, h=10, text=" 3. Risk Stratification Analysis", ln=True, fill=True)
	pdf.ln(5)
	
	# Dynamic Alert Logic: Calibrate color schemes to diagnostic severity
	# Red: Critical Alarm; Green: Baseline Safety; Amber: Elevated Monitoring
	if "high" in status_text.lower():
		bg, text_color = (255, 235, 235), (200, 0, 0)
	elif "low" in status_text.lower():
		bg, text_color = (235, 255, 235), (0, 128, 0)
	else:
		bg, text_color = (255, 250, 230), (180, 120, 0)
	
	# Render Summary Block: Visual emphasis on qualitative risk status
	pdf.set_fill_color(*bg)
	pdf.rect(x=10, y=pdf.get_y(), w=190, h=22, style="F")
	
	pdf.set_text_color(*text_color)
	pdf.set_font(family=pdf.font, style="B", size=15)
	pdf.cell(w=0, h=12, text=f"Assessment Conclusion: {status_text.upper()}", ln=True, align="C")
	
	pdf.set_font(family=pdf.font, style="B", size=11)
	pdf.cell(w=0, h=8, text=f"Individual Relative Risk (RR): {rr_score:.2f} Fold", ln=True, align="C")
	
	# Revert to standard palette for subsequent documentation
	pdf.set_text_color(0, 0, 0)
	pdf.ln(5)
	
	# --- Section 4: Graphical Prognostic Trajectory ---
	# Embed high-resolution visualization with atomic pagination handling
	if plot_buffer:
		add_pdf_chart(pdf, plot_buffer, " 4. Sarcopenia-Free Probability Curve")
	
	# --- Section 5: Professional Authentication and Accountability ---
	pdf.ln(5)
	pdf.set_font(family=pdf.font, style="I", size=10)
	# Placeholder for physician verification and clinical authentication
	pdf.cell(w=0, h=10, text="Physician Signature: ______________", ln=True, align="R")
	
	# Serialize the document object to a byte stream for delivery
	return bytes(pdf.output())
