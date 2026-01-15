
from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import numpy as np


class plotsView: 
    def __init__(
        self,
        df: pd.DataFrame,
        time_col: str, 
        lab_col: str, 
        inf_col: str,
        config: dict = None
    ):
        self.df = df.copy()
        self.time_col = time_col
        self.lab_col = lab_col
        self.inf_col = inf_col
        self.config = config or {}
    
        # Ensure data types are correct for time series plotting
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col], errors="coerce")
        self.df = self.df.dropna(subset=[self.time_col])
        self.df = self.df.sort_values(self.time_col)



    def lab_vs_deviation_plot(self):
        units = self.config.get('unit_label')
        ref = self.config.get('reference_column')
        meas = self.config.get('measurement_column')
        delta_max = self.config.get('delta_max', 1.0)  # Get threshold from config

        # Calculate the Delta & pct Deviation
        # Use pd.to_numeric to ensure these are floats, not strings
        try:
            _inf = pd.to_numeric(self.df[self.inf_col], errors='coerce')
            _lab = pd.to_numeric(self.df[self.lab_col], errors='coerce')
            
            # Perform the math using the forced numeric versions
            self.df['delta'] = _inf - _lab
            
        except Exception as e:
            st.error(f"Math Error: Could not subtract columns. {e}")
            return
        
        # Identify alert points (beyond threshold)
        self.df['abs_delta'] = self.df['delta'].abs()
        alert_mask = self.df['abs_delta'] >= delta_max
        alert_points = self.df[alert_mask]
        normal_points = self.df[~alert_mask]
        
        # Create the Figure
        fig = go.Figure()

        # Add normal points
        fig.add_trace(
            go.Scatter(
                x=normal_points[self.lab_col],
                y=normal_points['delta'],
                mode='markers',
                name=f"Normal [{units}]",
                marker=dict(size=6, color='blue'),
                customdata=normal_points[self.time_col],
                hovertemplate=(
                    "<b>Timestamp:</b> %{customdata|%Y-%m-%d %H:%M}<br>" +
                    "<b>Lab Value:</b> %{x} " + f"{units}<br>" +
                    "<b>Deviation:</b> %{y} " + f"{units}<br>" +
                    "<extra></extra>"
                ),
            )
        )
        
        # Add alert points (if any)
        if len(alert_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=alert_points[self.lab_col],
                    y=alert_points['delta'],
                    mode='markers',
                    name=f"⚠️ Alert (|Δ| ≥ {delta_max})",
                    marker=dict(
                        size=10, 
                        color='red',
                        symbol='circle',
                        line=dict(width=2, color='darkred')
                    ),
                    customdata=alert_points[self.time_col],
                    hovertemplate=(
                        "<b>⚠️ ALERT</b><br>" +
                        "<b>Timestamp:</b> %{customdata|%Y-%m-%d %H:%M}<br>" +
                        "<b>Lab Value:</b> %{x} " + f"{units}<br>" +
                        "<b>Deviation:</b> %{y} " + f"{units}<br>" +
                        "<extra></extra>"
                    ),
                )
            )

        # Add Zero-Reference Line
        fig.add_hline(y=0, line_dash="dash", line_color="black", annotation_text="Zero Deviation")
        
        # Add threshold lines
        fig.add_hline(y=delta_max, line_dash="dot", line_color="red", 
                     annotation_text=f"+{delta_max}", annotation_position="right")
        fig.add_hline(y=-delta_max, line_dash="dot", line_color="red", 
                     annotation_text=f"-{delta_max}", annotation_position="right")

        # Update Layout
        alert_count = len(alert_points)
        fig.update_layout(
            title=dict(
                text=f"<b>Deviation Plot | Alerts: {alert_count}<b>",
                font=dict(size=20),
                y=.90),
            xaxis_title=f"Lab Value ({units})",
            yaxis_title=f"Deviation ({units})",
            template="plotly_white",
            hovermode="closest",
            hoverlabel=dict(font_size=16, namelength=-1),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-.6,
                xanchor="center",
                x=0.1)
        )

        st.plotly_chart(fig, use_container_width=True, key="deviation_plot")

    def XY_plot(self):
        units = self.config.get('unit_label')
        ref = self.config.get('reference_column')
        meas = self.config.get('measurement_column')
        delta_max = self.config.get('delta_max', 1.0)  # Get threshold from config
    
        # Convert to numeric
        _inf = pd.to_numeric(self.df[self.inf_col], errors='coerce')
        _lab = pd.to_numeric(self.df[self.lab_col], errors='coerce')
        
        # Calculate deviation
        deviation = (_inf - _lab).abs()
        alert_mask = deviation >= delta_max
        
        # Split into normal and alert points
        df_plot = pd.DataFrame({
            'inf': _inf,
            'lab': _lab,
            'deviation': deviation,
            'alert': alert_mask
        })
        df_plot['timestamp'] = self.df[self.time_col].values
        
        normal_points = df_plot[~df_plot['alert']]
        alert_points = df_plot[df_plot['alert']]

        # Create figure
        fig = go.Figure()

        # Normal scatter points
        fig.add_trace(
            go.Scatter(
                x=normal_points['inf'],
                y=normal_points['lab'],
                mode='markers',
                name="Normal",
                customdata=normal_points['timestamp'],
                marker=dict(
                    size=6, 
                    color='green', 
                    symbol="x-open",
                    line=dict(width=1, color='green'),
                ),
                hoverlabel=dict(font_size=16, namelength=-1),
                hovertemplate=(
                    "<b>Timestamp:</b> %{customdata|%Y-%m-%d %H:%M}<br>" +
                    "<b>Analyzer/Inf:</b> %{x} " + f"{units}<br>" +
                    "<b>Lab Result:</b> %{y} " + f"{units}<br>" +
                    "<extra></extra>"
                ),
            )
        )
        
        # Alert scatter points (if any)
        if len(alert_points) > 0:
            fig.add_trace(
                go.Scatter(
                    x=alert_points['inf'],
                    y=alert_points['lab'],
                    mode='markers',
                    name=f"⚠️ Alert (|Δ| ≥ {delta_max})",
                    customdata=alert_points['timestamp'],
                    marker=dict(
                        size=10, 
                        color='red', 
                        symbol="circle",
                        line=dict(width=2, color='darkred'),
                    ),
                    hoverlabel=dict(font_size=16, namelength=-1),
                    hovertemplate=(
                        "<b>⚠️ ALERT</b><br>" +
                        "<b>Timestamp:</b> %{customdata|%Y-%m-%d %H:%M}<br>" +
                        "<b>Analyzer/Inf:</b> %{x} " + f"{units}<br>" +
                        "<b>Lab Result:</b> %{y} " + f"{units}<br>" +
                        "<extra></extra>"
                    ),
                )
            )

        # 1:1 Line
        min_val = float(_inf.min())
        max_val = float(_inf.max())

        fig.add_trace(
            go.Scatter( 
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name="1:1 Line",
            line=dict(color='firebrick', width=2, dash='dash')
            )
        )

        alert_count = len(alert_points)
        fig.update_layout(
            title=dict(
                text=f"<b>XY Plot | Alerts: {alert_count}<b>",
                font=dict(size=20),
                y=.90),
            xaxis_title=f"Analyzer/Inf ({units})",
            yaxis_title=f"Lab Result ({units})",
            template="plotly_white",
            hovermode="closest",
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-.6,
                xanchor="center",
                x=0.1)
        )

        st.plotly_chart(fig, use_container_width=True, key="xy_plot")