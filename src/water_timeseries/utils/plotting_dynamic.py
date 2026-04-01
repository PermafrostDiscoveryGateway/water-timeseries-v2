"""Dynamic plotting utilities using Plotly for interactive visualizations."""

from typing import List, Optional

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Color mapping for Dynamic World land cover classes
DW_COLOR_MAP = {
    "water": "#4a90e2",  # Blue
    "bare": "#8B4513",  # Brown
    "vegetation": "#228B22",  # Green
    "snow_and_ice": "#000000",  # Black
    "flooded_vegetation": "#31a354",  # Green shade
    "grass": "#addd8e",  # Light green
    "shrub_and_scrub": "#78c679",  # Medium green
    "trees": "#238b45",  # Dark green
    "crops": "#ffffcc",  # Light yellow
}


# Color mapping for JRC water classes
JRC_COLOR_MAP = {
    "area_water_permanent": "#4a90e2",  # Blue
    "area_water_seasonal": "#a6bddb",  # Light blue
    "area_land": "#8B4513",  # Brown
}


def plot_water_time_series_dw_interactive(
    df: pd.DataFrame,
    first_break: pd.Timestamp | None = None,
    normalization_factor: Optional[float] = None,
    lake_id: Optional[str] = None,
    height: int = 500,
    width: Optional[int] = None,
) -> go.Figure:
    """
    Create an interactive time series plot for Dynamic World data using Plotly.

    This function creates an interactive visualization similar to the static
    plot_water_time_series_dw but with Plotly for interactivity.

    Args:
        df: DataFrame with columns 'date', 'variable', and 'value'.
            The 'variable' column should contain land cover types.
        first_break: Optional timestamp for breakpoint vertical line.
        normalization_factor: Optional factor for secondary y-axis (normalized %).
        lake_id: Optional lake identifier for the title.
        height: Plot height in pixels.
        width: Plot width in pixels (optional).

    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure.

    Example:
        >>> fig = plot_water_time_series_dw_interactive(df, lake_id="abc123")
        >>> st.plotly_chart(fig)
    """
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Define which variables to plot (in order for legend)
    plot_order = ["water", "bare", "vegetation", "snow_and_ice"]

    # Plot each variable
    for variable in plot_order:
        if variable not in df["variable"].values:
            continue

        data = df[df["variable"] == variable]
        color = DW_COLOR_MAP.get(variable, "#888888")

        # Thicker line and markers for water
        line_width = 2.5 if variable == "water" else 1.5
        marker_size = 8 if variable == "water" else 6

        # Calculate relative values if normalization factor is provided
        customdata = None
        if normalization_factor is not None and normalization_factor > 0:
            customdata = (data["value"] / normalization_factor * 100).values

        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["value"],
                name=variable.replace("_", " ").title(),
                mode="lines+markers",
                line=dict(color=color, width=line_width),
                marker=dict(size=marker_size, symbol="circle"),
                connectgaps=True,
                customdata=customdata,
                hovertemplate="%{y:.2f} ha | %{customdata:.1f}%<extra></extra>"
                if customdata is not None
                else "%{y:.2f} ha<extra></extra>",
            ),
            secondary_y=False,
        )

    # Update layout
    title = f"Lake {lake_id}" if lake_id else "Water Time Series"
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Date",
        yaxis_title="Area [ha]",
        height=height,
        width=width,
        hovermode="x unified",
        hoverlabel=dict(namelength=-1),  # Show all trace names in hover
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
        ),
        template="plotly_white",
    )

    # Add grid to y-axis (only for primary/left axis with absolute values)
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#aaaaaa",
        secondary_y=False,
    )

    # Add secondary y-axis for normalized values
    if normalization_factor is not None:
        fig.update_yaxes(title="Relative Area [%]", secondary_y=True, range=[0, 100])
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name="Normalized",
                mode="lines",
                line=dict(color="gray", width=1, dash="dot"),
                showlegend=False,
            ),
            secondary_y=True,
        )

    # Update x-axis to show years nicely
    fig.update_xaxes(
        dtick="M12",  # Tick every 12 months
        ticklabelmode="period",  # Labels in the middle of the period
        tickformat="%Y",
        hoverformat="%Y-%m",  # Show YYYY-MM on hover
        showgrid=True,
        gridwidth=1,
        gridcolor="#cccccc",
    )

    return fig


def plot_water_time_series_jrc_interactive(
    df: pd.DataFrame,
    first_break: pd.Timestamp | None = None,
    plot_variables: List[str] = None,
    normalization_factor: Optional[float] = None,
    lake_id: Optional[str] = None,
    height: int = 500,
    width: Optional[int] = None,
) -> go.Figure:
    """
    Create an interactive time series plot for JRC data using Plotly.

    This function creates an interactive visualization similar to the static
    plot_water_time_series_jrc but with Plotly for interactivity.

    Args:
        df: DataFrame with columns for date and water/land variables.
        first_break: Optional timestamp for breakpoint vertical line.
        plot_variables: List of variables to plot. Default: JRC default variables.
        normalization_factor: Optional factor for secondary y-axis (normalized %).
        lake_id: Optional lake identifier for the title.
        height: Plot height in pixels.
        width: Plot width in pixels (optional).

    Returns:
        plotly.graph_objects.Figure: Interactive Plotly figure.

    Example:
        >>> fig = plot_water_time_series_jrc_interactive(df, lake_id="abc123")
        >>> st.plotly_chart(fig)
    """
    if plot_variables is None:
        plot_variables = ["area_water_permanent", "area_water_seasonal", "area_land"]

    # Calculate total area for no-data shading
    df = df.copy()
    if "area_data" in df.columns and "area_nodata" in df.columns:
        df["area_total"] = df["area_data"] + df["area_nodata"]

    # Create figure with secondary y-axis support
    fig = make_subplots(specs=[[{"secondary_y": normalization_factor is not None}]])

    # Plot each variable
    for variable in plot_variables:
        if variable not in df.columns:
            continue

        color = JRC_COLOR_MAP.get(variable, "#888888")
        line_width = 2.5 if variable == "area_water_permanent" else 1.5
        marker_size = 8 if variable == "area_water_permanent" else 6

        # Create nice display name
        display_name = variable.replace("area_", "").replace("_", " ").title()

        # Calculate relative values if normalization factor is provided
        customdata = None
        if normalization_factor is not None and normalization_factor > 0:
            customdata = (df[variable] / normalization_factor * 100).values

        fig.add_trace(
            go.Scatter(
                x=df["date"],
                y=df[variable],
                name=display_name,
                mode="lines+markers",
                line=dict(color=color, width=line_width),
                marker=dict(size=marker_size, symbol="circle"),
                fill=None,  # No fill for any variable
                connectgaps=True,
                customdata=customdata,
                hovertemplate="%{y:.2f} ha | %{customdata:.1f}%<extra></extra>"
                if customdata is not None
                else "%{y:.2f} ha<extra></extra>",
            ),
            secondary_y=False,
        )

    # Update layout
    title = f"Lake {lake_id}" if lake_id else "JRC Water Time Series"
    fig.update_layout(
        title=dict(text=title, x=0.5),
        xaxis_title="Date",
        yaxis_title="Area [ha]",
        height=height,
        width=width,
        hovermode="x unified",
        hoverlabel=dict(namelength=-1),  # Show all trace names in hover
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
        ),
        template="plotly_white",
    )

    # Add grid to y-axis
    fig.update_yaxes(
        showgrid=True,
        gridwidth=1,
        gridcolor="#aaaaaa",
        secondary_y=False,
    )

    # Add secondary y-axis for normalized values
    if normalization_factor is not None:
        fig.update_yaxes(title="Relative Area [%]", secondary_y=True, range=[0, 100])
        fig.add_trace(
            go.Scatter(
                x=[],
                y=[],
                name="Normalized",
                mode="lines",
                line=dict(color="gray", width=1, dash="dot"),
                showlegend=False,
            ),
            secondary_y=True,
        )

    # Update x-axis to show years nicely
    fig.update_xaxes(
        dtick="M12",  # Tick every 12 months
        tick0=df["date"].min(),  # Start ticks at first data point
        tickmode="linear",
        tickformat="%Y",
        hoverformat="%Y",  # Show YYYY on hover
    )

    return fig


def create_comparison_plot(
    df_dw: pd.DataFrame,
    df_jrc: pd.DataFrame,
    lake_id: str,
    height: int = 600,
) -> go.Figure:
    """
    Create a side-by-side comparison of DW and JRC time series.

    Args:
        df_dw: DataFrame prepared for DW plotting.
        df_jrc: DataFrame for JRC plotting.
        lake_id: Lake identifier.
        height: Plot height in pixels.

    Returns:
        plotly.graph_objects.Figure: Combined figure with subplots.
    """
    fig = make_subplots(
        rows=2,
        cols=1,
        subplot_titles=("Dynamic World", "JRC Water"),
        vertical_spacing=0.15,
    )

    # Plot DW on top
    dw_vars = ["water", "bare", "vegetation", "snow_and_ice"]
    for var in dw_vars:
        if var not in df_dw["variable"].values:
            continue
        data = df_dw[df_dw["variable"] == var]
        color = DW_COLOR_MAP.get(var, "#888888")

        fig.add_trace(
            go.Scatter(
                x=data["date"],
                y=data["value"],
                name=var,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                legendgroup="dw",
                showlegend=True,
            ),
            row=1,
            col=1,
        )

    # Plot JRC on bottom
    jrc_vars = ["area_water_permanent", "area_water_seasonal", "area_land"]
    for var in jrc_vars:
        if var not in df_jrc.columns:
            continue
        color = JRC_COLOR_MAP.get(var, "#888888")
        display_name = var.replace("area_", "").replace("_", " ").title()

        fig.add_trace(
            go.Scatter(
                x=df_jrc["date"],
                y=df_jrc[var],
                name=display_name,
                mode="lines+markers",
                line=dict(color=color, width=2),
                marker=dict(size=6),
                legendgroup="jrc",
                showlegend=True,
            ),
            row=2,
            col=1,
        )

    fig.update_layout(
        title=dict(text=f"Lake {lake_id} - Comparison", x=0.5),
        height=height,
        hovermode="x unified",
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=1.02,
        ),
        template="plotly_white",
    )

    fig.update_xaxes(title_text="Date", row=2, col=1)
    fig.update_yaxes(title_text="Area [ha]", row=1, col=1)
    fig.update_yaxes(title_text="Area [ha]", row=2, col=1)

    return fig
