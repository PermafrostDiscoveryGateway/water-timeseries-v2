"""Tutorial popup helper for map_viewer dashboard."""

from typing import Optional

import streamlit as st


# =============================================================================
# INTERNAL DIALOG - DO NOT CALL DIRECTLY
# =============================================================================
@st.dialog("Welcome to Lost Lakes", width="large")
def _show_tutorial_dialog(sections: dict[str, str]) -> None:
    """Internal dialog function for the tutorial popup."""
    # Header
    st.markdown("### Lost Lakes: Near Real-Time Lake Drainage")
    st.markdown("Discover Arctic Lakes disappearing right after it happened")

    # Main content (2/3) and info panel (1/3) with vertical divider
    col_content, col_divider, col_sidebar = st.columns([2, 0.02, 1])

    with col_content:
        st.markdown("---")
        for header, content in sections.items():
            st.markdown(f"**{header}**")
            st.markdown(content)

    with col_divider:
        st.markdown(
            """
            <style>
                .vertical-divider {
                    border-left: 1px solid #e0e0e0;
                    height: 100%;
                }
            </style>
            <div class="vertical-divider"></div>
            """,
            unsafe_allow_html=True,
        )

    with col_sidebar:
        # Info box with logos and acknowledgment
        st.markdown(
            """
        <style>
            .info-box {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                padding: 10px;
                margin-bottom: 10px;
                margin-top: -8px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            }
            .info-box a {
                color: #1a73e8;
                text-decoration: none;
            }
            .info-box a:hover {
                text-decoration: underline;
            }
            .logo-row {
                display: flex;
                flex-wrap: wrap;
                align-items: center;
                gap: 6px;
                margin-bottom: 8px;
                padding-bottom: 8px;
                border-bottom: 1px solid #e0e0e0;
            }
            .logo-row img {
                height: 24px;
                width: auto;
            }
            .text-row {
                font-size: 10px;
                line-height: 1.4;
            }
            [data-testid="stHorizontalBlock"] [data-testid="stVerticalBlock"]:nth-child(3) > div {
                margin-top: 0 !important;
                padding-top: 0 !important;
            }
        </style>
        <div class="info-box">
            <div class="logo-row">
                <img src="https://www.awi.de/_assets/978631966794c5093250775de182779d/Images/AWI/awi_logo.svg" alt="AWI">
                <img src="https://pdg.open.uaf.edu/wp-content/uploads/sites/46/2023/06/PDG_logo_compact_transparent_bkg.png" alt="PDG">
                <img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcQ5o8cBYEkcRNtWKws2y7tZPZvy2OeEC5pp-S-13KPd_g&s=10" alt="Arctic ODS">
                <img src="https://images.seeklogo.com/logo-png/63/1/national-center-for-supercomputing-applications-logo-png_seeklogo-637758.png" alt="NCSA">
                <img src="https://arcticdata.io/wp-content/themes/aurora/library/images/logo_.png" alt="Arctic Data">
                <img src="https://www.google.com/s2/favicons?domain=woodwellclimate.org&sz=64" alt="Woodwell">
            </div>
            <div class="text-row">
                <b>Lost Lakes</b> is part of the <i>Permafrost Discovery Gateway</i> Project. The Permafrost Discovery Gateway (PDG) is an online platform for creating, analyzing, and visualizing big geospatial permafrost products funded through the <i>Google.org Impact Challenge</i>.<br><br>
                <b>Team:</b> Created by Ingmar Nitze (AWI), Todd Nicholson (NCSA), Drshika Asher (NCSA), Kayla Hardie (Google), Matt Jones (NCEAS/ADC). With support from many PDG Team members.<br><br>
                <b>Contact:</b> <a href="mailto:ingmar.nitze@awi.de?subject=Lost%20Lakes%20Viewer">ingmar.nitze@awi.de</a>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Photos stacked vertically, matching info box width
        photos_html = """
        <style>
            .photo-box {
                background-color: #ffffff;
                border: 1px solid #e0e0e0;
                border-radius: 8px;
                overflow: hidden;
                margin-bottom: 10px;
                box-shadow: 0 1px 3px rgba(0, 0, 0, 0.08);
            }
            .photo-box img {
                width: 100%;
                height: auto;
                display: block;
            }
            .photo-caption {
                font-size: 10px;
                color: #666;
                padding: 6px 8px;
                background-color: #f9f9f9;
                line-height: 1.3;
            }
        </style>
        """
        st.markdown(photos_html, unsafe_allow_html=True)

        try:
            from pathlib import Path

            img_path = Path(__file__).parent / ".." / "images" / "P1010258.JPG"
            import base64

            with open(img_path.resolve(), "rb") as f:
                img_data = f.read()
            b64 = base64.b64encode(img_data).decode()
            st.markdown(
                f"""
            <div class="photo-box">
                <img src="data:image/jpeg;base64,{b64}" />
                <div class="photo-caption">Drained Lake at Cape Halkett, Alaska North Slope, July 2015. Photo I.Nitze (AWI)</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        except Exception:
            pass

        try:
            from pathlib import Path

            img_path = Path(__file__).parent / ".." / "images" / "20240701_110133.jpg"
            import base64

            with open(img_path.resolve(), "rb") as f:
                img_data = f.read()
            b64 = base64.b64encode(img_data).decode()
            st.markdown(
                f"""
            <div class="photo-box">
                <img src="data:image/jpeg;base64,{b64}" />
                <div class="photo-caption">Lake-rich permafrost landscape on the Seward Peninsula in Alaska, July 2024. Photo I.Nitze (AWI)</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
        except Exception:
            pass

    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Close!", key="tutorial_close", use_container_width=True):
            st.session_state["tutorial_visible"] = False
            st.rerun()


# =============================================================================
# STANDARD TUTORIAL CONTENT (shown for ALL configurations)
# =============================================================================
STANDARD_TUTORIAL = {
    "Getting Started": (
        "This dashboard lets you explore spatial data through interactive maps "
        "and visualizations. Use the sidebar to navigate and configure views."
    ),
    "Interactive Features": (
        "- **Click** on a lake to plot monthly area information\n"
        "- **Zoom & pan** using mouse or touch controls\n"
        "- **Toggle layers** to show/hide different data sets\n"
        "- **Export** data using download buttons"
    ),
}


# =============================================================================
# CONFIG-SPECIFIC TUTORIAL CONTENT (extend as needed)
# =============================================================================
CONFIG_TUTORIALS: dict[str, dict[str, str]] = {
    "nrt_drainage": {
        "Near Real-Time Drainage Detection": (
            "This view shows near real-time drainage events detected from "
            "satellite imagery. Drains appear as rapid transitions from water "
            "to land in the time series data."
        ),
        "Detection Methodology": (
            "We used time-series monthly aggregated Dynamic World (Land Cover layer base on Sentinel-2 imagery). "
            "Drainage events are identified when the water area of the latest "
            "observation is much lower than the expected area. We used ARIMA forcasting to estimate the expected water area for the analyzed month. "
            "Drainage confidence is a score that increases when: \n"
            "1. Observed lake area is 25% below expected lake area \n"
            "2. Observed lake area is below th expected 90% confidence interval \n"
            "3. Observed lake area is the absolute historical minimum since 2017 \n"
            " \n"
            "A drainage confidence score of 3 means that this is a likely lake drainage.\n"
            "To validate the change you can scroll down to visually inspect the latest satellite image and compare it to a historical image."
        ),
    },
    "drainage_year": {
        "Near Real-Time Drainage Detection": (
            "This view shows drained lakes between 2017 and 2025. Drained lakes are highlighted."
        ),
        "Detection Methodology": (
            "We used time-series monthly aggregated Dynamic World (Land Cover layer base on Sentinel-2 imagery). We used a basic thresholding where lake area dropped by 25% or more with a sustained lake area loss."
        ),
    },
}


def _get_tutorial_sections(config_name: Optional[str] = None) -> dict[str, str]:
    """Get tutorial sections for a configuration."""
    sections = dict(STANDARD_TUTORIAL)
    if config_name and config_name in CONFIG_TUTORIALS:
        sections.update(CONFIG_TUTORIALS[config_name])
    return sections


def show_tutorial_popup(config_name: Optional[str] = None, auto_show: bool = True) -> None:
    """Display a tutorial popup.

    Args:
        config_name: Optional configuration name to show config-specific content.
        auto_show: If True, shows on startup (respects dismissal). If False, forces show.

    Usage:
        Add at the top of your create_app() function in map_viewer.py:

        show_tutorial_popup(config_name="your_config_name")

    Or call with auto_show=False to force display:

        show_tutorial_popup(config_name="your_config_name", auto_show=False)
    """
    # Initialize session state
    if "tutorial_visible" not in st.session_state:
        st.session_state["tutorial_visible"] = True
    if "tutorial_dismissed" not in st.session_state:
        st.session_state["tutorial_dismissed"] = False
    if "tutorial_dialog_shown" not in st.session_state:
        st.session_state["tutorial_dialog_shown"] = False

    # Check if tutorial should be shown
    if auto_show:
        if st.session_state.get("tutorial_dismissed", False):
            return
        if st.session_state.get("tutorial_dialog_shown", False):
            return
        if not st.session_state.get("tutorial_visible", True):
            return

    # Mark dialog as shown this run to prevent reopening on reruns
    st.session_state["tutorial_dialog_shown"] = True

    # Get tutorial content and show dialog
    sections = _get_tutorial_sections(config_name)
    _show_tutorial_dialog(sections)


def show_help_button(config_name: Optional[str] = None, label: str = "Open Help") -> None:
    """Add a help button to the sidebar that opens the tutorial popup.

    Args:
        config_name: Configuration name for config-specific content.
        label: Button label text.

    Usage:
        In your sidebar:

        from tutorial_popup import show_help_button
        show_help_button(config_name=viz_configuration_name)
    """
    if st.button(label, key="tutorial_help_btn", use_container_width=True):
        # Reset the flag so the dialog shows
        st.session_state["tutorial_dialog_shown"] = False
        show_tutorial_popup(config_name=config_name, auto_show=False)


def register_config_tutorial(config_name: str, content: dict[str, str]) -> None:
    """Register custom tutorial sections for a specific configuration.

    Args:
        config_name: Name matching your viz_configuration key.
        content: Dict of {header: content_text} pairs.

    Example:
        register_config_tutorial("flood_analysis", {
            "Flood Risk Levels": "Red zones indicate high flood risk...",
            "Evacuation Routes": "Dashed lines show evacuation paths...",
        })
    """
    if config_name not in CONFIG_TUTORIALS:
        CONFIG_TUTORIALS[config_name] = {}
    CONFIG_TUTORIALS[config_name].update(content)


def get_config_tutorial(config_name: str) -> dict[str, str]:
    """Retrieve tutorial content for a specific configuration.

    Returns config-specific content merged with standard content.
    """
    sections = dict(STANDARD_TUTORIAL)
    if config_name in CONFIG_TUTORIALS:
        sections.update(CONFIG_TUTORIALS[config_name])
    return sections


def clear_config_tutorial(config_name: str) -> None:
    """Remove custom tutorial content for a configuration."""
    if config_name in CONFIG_TUTORIALS:
        del CONFIG_TUTORIALS[config_name]


def reset_all_tutorials() -> None:
    """Clear all config-specific tutorials and reset to defaults."""
    CONFIG_TUTORIALS.clear()
