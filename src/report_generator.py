"""
This project is supported by the Department of Geriatrics and the National Clinical Research Center for Geriatrics,
West China Hospital, Sichuan University, China.
"""

import datetime
import io

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from fpdf import FPDF
from PIL import Image
from typing import Any

from src.prediction import FEATURE_LABELS, PROJECT_ROOT, ensure_cumulative_risk_0_time, risk_at_time


def plot_cumulative_risk_curve(
		cumulative_risk: pd.Series,
		highlight_times: list[int] | None = None,
		font_family: str | list[str] | None = None,
) -> io.BytesIO:
	"""Plots the individual cumulative-incidence curve for the PDF report.

	Args:
		cumulative_risk: Averaged Fine-Gray probabilities indexed by years.
		highlight_times: Selected prediction years to mark on the curve.
		font_family: Matplotlib font family, with a portable fallback by default.

	Returns:
		An in-memory PNG at 300 dpi. The caller owns this buffer.
	"""
	curve = ensure_cumulative_risk_0_time(cumulative_risk)
	fonts = ["Arial", "DejaVu Sans"] if font_family is None else font_family
	# A local style context prevents PDF plotting from changing other app charts.
	with plt.rc_context({"font.family": fonts, "axes.unicode_minus": False}):
		fig, ax = plt.subplots(figsize=(10, 6), dpi=300)
		axis_label_size = 14
		risk_label_offset = 12  # Fixed physical distance above every red point, in points.
		ax.step(curve.index, curve.values, where="post", color="#2E86C1", linewidth=3)
		ax.fill_between(curve.index, curve.values, step="post", alpha=0.20, color="#D6EAF8")
		ax.set_xlim(0, float(curve.index.max()) + 0.4)
		# Leave room above 100% for the larger labels; probability ticks stop at 100%.
		ax.set_ylim(0, 1.10)
		ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
		ax.xaxis.set_major_formatter(ticker.FormatStrFormatter("%d"))
		ax.set_yticks([index / 10 for index in range(11)])
		ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1.0, decimals=0))

		# Keep all labels centered above their points at the same physical distance.
		for year in sorted(set(highlight_times or [])):
			probability = risk_at_time(curve, year)
			ax.scatter(year, probability, color="#C0392B", s=65, zorder=5)
			ax.vlines(year, 0, probability, linestyles=":", colors="gray", alpha=0.5)
			ax.annotate(
				f"{probability:.2%}", xy=(year, probability),
				xytext=(0, risk_label_offset), textcoords="offset points",
				fontsize=axis_label_size, color="#C0392B", ha="center", va="bottom",
			)
		ax.set_xlabel("Time from Baseline (Years)", fontsize=axis_label_size, fontweight="bold", labelpad=10)
		ax.set_ylabel("Cumulative Risk of Sarcopenia", fontsize=axis_label_size, fontweight="bold", labelpad=10)
		ax.tick_params(axis="both", labelsize=12)
		ax.spines["top"].set_visible(False)
		ax.spines["right"].set_visible(False)
		ax.spines["left"].set_linewidth(1.2)
		ax.spines["bottom"].set_linewidth(1.2)
		fig.tight_layout()
		buffer = io.BytesIO()
		try:
			fig.savefig(buffer, format="png", dpi=300)
		finally:
			plt.close(fig)
		buffer.seek(0)
	return buffer


def format_output_value(value: Any) -> str:
	"""Formats numeric patient measurements without exposing long float strings."""
	if isinstance(value, (float, int)):
		return f"{value:.2f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
	return str(value)


def format_user_data_for_report(user_data: dict[str, Any]) -> dict[str, str]:
	"""Maps only the five supported inputs to human-readable PDF labels.

	Args:
		user_data: Patient inputs, optional identifier and selected prediction years.

	Returns:
		Patient labels and values. No external feature-mapping file is needed.
	"""
	formatted = {}
	if user_data.get("user_id"):
		formatted["User ID / Record No."] = str(user_data["user_id"])
	for feature, label in FEATURE_LABELS.items():
		value = user_data[feature]
		if feature == "sex":
			if value not in (1, 2):
				raise ValueError("Sex must be coded as male=1 or female=2.")
			formatted[label] = "Male" if value == 1 else "Female"
		elif feature == "age":
			formatted[label] = format_output_value(value) + " years"
		else:
			formatted[label] = format_output_value(value) + " cm"
	return formatted


class ReportPDF(FPDF):
	"""Keeps the institutional header and footer consistent across report pages."""

	def __init__(self, font_family: str = "Arial", *args: Any, **kwargs: Any) -> None:
		"""Initializes the report with a shared font and reserved footer space."""
		super().__init__(*args, **kwargs)
		self.report_font = font_family
		self.set_auto_page_break(auto=True, margin=30)

	def header(self) -> None:
		"""Renders the existing logo and institutional heading."""
		self.image(name=str(PROJECT_ROOT / "logo.ico"), x=10, y=7, w=15)
		self.set_font(family=self.report_font, style="B", size=10)
		self.set_text_color(31, 119, 180)
		self.multi_cell(
			w=0, h=5,
			text="National Clinical Research Center for Geriatrics\nWest China Hospital, Sichuan University, China",
			align="R", new_x="LMARGIN", new_y="NEXT",
		)
		self.set_text_color(0, 0, 0)
		self.ln(8)

	def footer(self) -> None:
		"""Adds the Fine-Gray model statement and dynamic page numbering."""
		self.set_y(-25)
		self.set_font(family=self.report_font, style="I", size=8)
		self.set_text_color(130, 130, 130)
		self.multi_cell(
			w=0, h=4,
			text=("This research tool estimates cumulative sarcopenia risk using a Fine-Gray model "
				  "that accounts for competing death. Estimates support clinical evaluation and do not "
				  "constitute a diagnosis."),
			new_x="LMARGIN", new_y="NEXT",
		)
		self.set_y(-10)
		self.cell(w=0, h=5, text=f"Page {self.page_no()}/{{nb}}", align="C")


def _section_header(
		pdf: ReportPDF,
		title: str,
) -> None:
	"""Adds a shaded section heading while reserving room for its first row."""
	if pdf.will_page_break(22):
		pdf.add_page()
	pdf.set_font(family=pdf.report_font, style="B", size=13)
	pdf.set_text_color(0, 0, 0)
	pdf.set_fill_color(245, 245, 245)
	pdf.cell(w=0, h=10, text=f" {title}", fill=True, new_x="LMARGIN", new_y="NEXT")
	pdf.ln(3)


def add_patient_metadata_table(
		pdf: ReportPDF,
		user_data: dict[str, str],
) -> None:
	"""Renders the identifier and five measurements in a readable baseline table.

	Args:
		pdf: Active report document.
		user_data: Already formatted patient labels and values.
	"""
	for label, value in user_data.items():
		if pdf.will_page_break(8):
			pdf.add_page()
		pdf.set_font(family=pdf.report_font, style="B", size=10)
		pdf.cell(w=80, h=8, text=f"{label}:")
		pdf.set_font(family=pdf.report_font, size=10)
		pdf.multi_cell(w=0, h=8, text=value, new_x="LMARGIN", new_y="NEXT")
	pdf.ln(3)


def add_risk_progress_table(
		pdf: ReportPDF,
		cumulative_risk: pd.Series,
		years: list[int],
) -> None:
	"""Adds selected-year probabilities and neutral 0-100% progress bars.

	Args:
		pdf: Active report document.
		cumulative_risk: The same cumulative incidences displayed on the website.
		years: Selected prediction horizons.
	"""
	def table_header() -> None:
		pdf.set_font(family=pdf.report_font, style="B", size=10)
		pdf.set_fill_color(230, 230, 230)
		pdf.cell(45, 9, "Prediction Horizon", border=1, align="C", fill=True)
		pdf.cell(45, 9, "Cumulative Risk", border=1, align="C", fill=True)
		pdf.cell(100, 9, "Risk Scale: 0%-100%", border=1, align="C", fill=True,
			new_x="LMARGIN", new_y="NEXT")

	table_header()
	for year in sorted(set(years)):
		if pdf.will_page_break(10):
			pdf.add_page()
			table_header()
		probability = risk_at_time(cumulative_risk, year)
		x, y = pdf.l_margin, pdf.get_y()
		pdf.set_font(family=pdf.report_font, size=10)
		pdf.cell(45, 10, f"{year} Year(s)", border=1, align="C")
		pdf.cell(45, 10, f"{probability:.2%}", border=1, align="C")
		pdf.cell(100, 10, "", border=1)
		bar_x, bar_y, bar_width = x + 96, y + 3, 88
		pdf.set_fill_color(232, 236, 240)
		pdf.rect(bar_x, bar_y, bar_width, 4, style="F")
		if probability > 0:
			pdf.set_fill_color(46, 134, 193)
			pdf.rect(bar_x, bar_y, bar_width * probability, 4, style="F")
		pdf.set_xy(pdf.l_margin, y + 10)
	pdf.ln(4)


def add_pdf_chart(
		pdf: ReportPDF,
		plot_buffer: io.BytesIO,
		chart_title: str,
		width: int | float = 160,
		x_pos: int | float = 25,
) -> None:
	"""Places a chart and its heading together, preserving the image aspect ratio.

	Args:
		pdf: Active report document.
		plot_buffer: PNG buffer owned and closed by the report generator.
		chart_title: Heading for this report section.
		width: Displayed image width in millimeters.
		x_pos: Image left position in millimeters.
	"""
	plot_buffer.seek(0)
	with Image.open(io.BytesIO(plot_buffer.getvalue())) as image:
		image_height = width * image.height / image.width
	# Inspect a separate buffer so PIL cannot close the original before PDF embedding.
	if pdf.will_page_break(image_height + 18):
		pdf.add_page()
	_section_header(pdf, chart_title)
	y = pdf.get_y()
	pdf.image(name=plot_buffer, x=x_pos, y=y, w=width)
	pdf.set_y(y + image_height + 4)


def generate_report_pdf(
		user_data: dict[str, Any],
		cumulative_risk: pd.Series,
		plot_buffer: io.BytesIO | None,
		font_name: str = "Arial",
		out_of_bounds: list[str] | None = None,
) -> bytes:
	"""Generates the baseline profile, risk percentages, bars and risk-curve PDF.

	Args:
		user_data: Five raw predictors, optional identifier and evaluation years.
		cumulative_risk: Averaged imputation-specific Fine-Gray probabilities.
		plot_buffer: Optional PNG chart; closed after successful or failed export.
		font_name: Font family used throughout the report.
		out_of_bounds: Optional messages for measurements beyond development bounds.

	Returns:
		PDF bytes ready for Streamlit's download button.
	"""
	years = list(user_data.get("eval_times", []))
	if not years:
		raise ValueError("Select at least one prediction year before generating a report.")
	# Validate all selected years before writing a partially completed document.
	for year in years:
		risk_at_time(cumulative_risk, year)
	pdf = ReportPDF(font_family=font_name)
	pdf.alias_nb_pages()
	pdf.add_page()
	pdf.set_font(family=pdf.report_font, style="B", size=18)
	pdf.cell(0, 13, "Sarcopenia Risk Assessment Report", align="C", new_x="LMARGIN", new_y="NEXT")
	pdf.set_font(family=pdf.report_font, size=9)
	timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	pdf.cell(0, 5, f"Generated on: {timestamp}", align="L", new_x="LMARGIN", new_y="NEXT")
	pdf.ln(6)

	_section_header(pdf, "1. Individual Baseline Information")
	add_patient_metadata_table(pdf, format_user_data_for_report(user_data))
	_section_header(pdf, "2. Sarcopenia Risk Assessment")
	add_risk_progress_table(pdf, cumulative_risk, years)
	pdf.set_font(family=pdf.report_font, size=9)
	pdf.multi_cell(
		0, 5,
		"Percentages represent the estimated cumulative probability of developing sarcopenia "
		"within each stated year, accounting for death as a competing event.",
		new_x="LMARGIN", new_y="NEXT",
	)
	if out_of_bounds:
		pdf.ln(2)
		pdf.multi_cell(0, 5, "Outside development-cohort ranges: " + "; ".join(out_of_bounds)
			+ ". Estimates may be less reliable.", new_x="LMARGIN", new_y="NEXT")
	pdf.ln(4)

	try:
		if plot_buffer is not None:
			add_pdf_chart(pdf, plot_buffer, "3. Cumulative Risk of Sarcopenia")
		pdf.set_font(family=pdf.report_font, style="I", size=9)
		pdf.multi_cell(
			0, 5,
			"Model: Ridge Fine-Gray regression, using age, sex, mid-upper arm circumference, "
			"hip circumference and calf circumference. Predictions are averaged across five "
			"models fitted to multiply imputed development data. "
			"Sarcopenia was defined using AWGS 2025 criteria.",
			new_x="LMARGIN", new_y="NEXT",
		)
		pdf.ln(4)
		pdf.cell(0, 8, "Physician Signature: ______________", align="R", new_x="LMARGIN", new_y="NEXT")
		return bytes(pdf.output())
	finally:
		if plot_buffer is not None:
			plot_buffer.close()
