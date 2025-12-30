"""
This project is supported by the Department of Geriatrics and the National Clinical Research Center for Geriatrics,
West China Hospital, Sichuan University, China.
"""

import base64
import streamlit as st
from pathlib import Path


def get_image_base64(image_path: str | Path) -> str:
	"""Converts a local image file to a base64 encoded string.

	This function reads an image from the local file system, ensures its
	existence, and encodes it for embedding in HTML/CSS contexts.

	Args:
		image_path (str | Path): The full filesystem path to the image file.

	Returns:
		str: The base64 encoded string representing the image data.

	Raises:
		FileNotFoundError: If the specified image_path does not exist or is not a file.
	"""
	path = Path(image_path)
	
	# Ensure the path points to a valid file before reading
	if not path.is_file():
		raise FileNotFoundError(f"Logo image not found at: {path.absolute()}")
	
	# Read and encode the file bytes
	img_bytes = path.read_bytes()
	return base64.b64encode(img_bytes).decode()


def set_st_header(
		main_title: str,
		image_path: str | Path,
		sidebar_title: str
) -> None:
	"""Sets the header configuration for the Streamlit application.

	This includes configuring the page layout, rendering the sidebar title,
	and displaying a centered main title with an accompanying logo.

	Args:
		main_title: The text to be displayed as the primary heading.
		image_path: The path to the logo image file.
		sidebar_title: The title to be displayed in the sidebar.
	"""
	
	# Configure the page to use the wide-screen layout
	st.set_page_config(layout="wide")
	
	# --- Sidebar Configuration ---
	# Inject custom CSS to center the sidebar title
	st.markdown(
		"""
		<style>
		[data-testid="stSidebarNav"] + div h1 {
		   text-align: center;
		}
		/* Target the sidebar title specifically */
		[data-testid="stSidebar"] h1 {
		   text-align: center;
		}
		</style>
		""",
		unsafe_allow_html=True
	)
	st.sidebar.title(sidebar_title)
	st.sidebar.divider()
	
	# --- Main Page Header Configuration ---
	# Inject custom CSS for a flexbox-based header (Logo + Text)
	st.markdown(
		"""
		<style>
		.main-title {
		   display: flex;
		   align-items: center; /* Vertical alignment */
		   justify-content: center; /* Horizontal alignment */
		   gap: 20px; /* Spacing between logo and text */
		   margin-bottom: 25px;
		}
		.main-title img {
		   width: 70px; /* Set fixed logo width */
		   height: auto; /* Maintain aspect ratio */
		}
		</style>
		""",
		unsafe_allow_html=True
	)
	
	# Convert the logo to base64 for HTML embedding
	try:
		image_base64 = get_image_base64(image_path)
		# Render the logo and title using HTML
		st.markdown(
			f"""
            <div class="main-title">
               <img src="data:image/png;base64,{image_base64}" alt="logo">
               <h1>{main_title}</h1>
            </div>
            """,
			unsafe_allow_html=True
		)
	except Exception:
		# Fallback if image loading fails
		st.title(main_title)
	
	# --- Notice / Disclaimer Banner with Gradient ---
	st.markdown(
		"""
		<div style="
		   background: linear-gradient(90deg, #4A90E2 0%, #9013FE 100%);
		   color: #FFFFFF;
		   padding: 12px;
		   border-radius: 5px;
		   text-align: center;
		   font-weight: bold;
		   font-size: 16px;
		   margin-bottom: 30px;
		   line-height: 1.5;
		   box-shadow: 0 4px 6px rgba(0,0,0,0.1);
		   ">
		   Supported by the National Clinical Research Center for Geriatrics & West China Hospital, Sichuan University, China.
		   For non-commercial use only. <br>
		</div>
		""",
		unsafe_allow_html=True
	)
	
	with st.expander("📜 Read Before Use", expanded=True):
		st.warning("""
		**Clinical Disclaimer**:
		- This risk assessment is powered by a **Cox Proportional Hazards Ensemble Model**, which has been rigorously validated using longitudinal cohorts from **West China Hospital**.
		- This tool is intended strictly for **clinical decision support** and does not constitute a formal medical diagnosis or a professional medical opinion.
		- All predictive results should be interpreted within the context of a comprehensive clinical evaluation. **Final clinical judgment and diagnostic responsibility remain exclusively with the presiding physician**.
		""")


def show_responsive_warning(breakpoint_px: int = 768) -> None:
	"""Renders a conditional warning message tailored for mobile/narrow screens.

	This function utilizes CSS Media Queries to inject a custom HTML alert box
	that remains hidden on desktops but becomes visible when the viewport width
	drops below the specified threshold.

	Unlike standard st.warning(), this approach bypasses Streamlit's component
	isolation, allowing for precise responsive layout control without affecting
	other UI elements.

	Args:
		breakpoint_px (int): The maximum screen width (in pixels) at which the
			warning should be displayed. Defaults to 768 (standard tablet/mobile breakpoint).
	"""
	
	# Define the styling and behavior using an internal CSS block
	# We use a unique class name to avoid conflicts with Streamlit's native styles
	st.markdown(f"""
        <style>
        .mobile-only-alert {{
            display: none; /* Default hidden state for desktop/wide screens */
            background-color: #fff3cd;
            color: #856404;
            padding: 16px;
            border-radius: 8px;
            border: 1px solid #ffeeba;
            margin-bottom: 24px;
            font-family: 'Source Sans Pro', sans-serif;
            font-size: 14px;
            line-height: 1.6;
            box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        }}

        /* Activation logic based on the passed breakpoint parameter */
        @media (max-width: {breakpoint_px}px) {{
            .mobile-only-alert {{
                display: block; /* Visible on mobile devices */
            }}
        }}
        </style>

        <div class="mobile-only-alert">
            <span style="font-size: 18px; margin-right: 8px;">⚠️</span>
            <strong>Note for Mobile Users:</strong> For the optimal visual experience
            with detailed prognostic charts and clinical reports, using a desktop
            or tablet browser is highly recommended.
        </div>
    """, unsafe_allow_html=True)
