
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import math
import pandas as pd
import numpy as np
import streamlit as st
import plotly.graph_objects as go


class notebook_compare:
    """
    Comparison analysis class for Lab vs Analyzer/Inferential data.
    Handles windowed statistics, t-tests, and deviation alerts.
    """
    
    def __init__(
        self,
        df: pd.DataFrame,
        time_col: str,
        lab_col: str,
        inf_col: str,
        config: dict = None,
        window_freq: str = "1H",
        t_threshold: float = 2.0,
        delta_max: float = 0.5,
        max_chart_points: int = 5000
    ):
        """
        Initialize comparison analysis.
        
        Parameters
        ----------
        df : pd.DataFrame
            Raw data with timestamp, reference, and measurement columns
        time_col : str
            Name of timestamp column
        lab_col : str
            Name of lab/reference column
        inf_col : str
            Name of analyzer/inferential measurement column
        config : dict, optional
            Configuration dictionary with metadata
        window_freq : str
            Window frequency (e.g., "8H", "1D")
        t_threshold : float
            Threshold for |t-statistic| alerts
        delta_max : float
            Threshold for absolute deviation alerts
        max_chart_points : int
            Maximum points to display in charts (default: 5000)
        """
        self.time_col = time_col
        self.lab_col = lab_col
        self.inf_col = inf_col
        self.config = config or {}
        self.window_freq = window_freq
        self.t_threshold = t_threshold
        self.delta_max = delta_max
        self.max_chart_points = max_chart_points
        
        # Prepare data
        self.df_clean = self._prepare_data(df)
        
        # Calculate alerts and statistics
        self._calculate_analysis()
    
    def _prepare_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and prepare data for analysis."""
        df_clean = df[[self.time_col, self.lab_col, self.inf_col]].copy()
        df_clean[self.time_col] = pd.to_datetime(df_clean[self.time_col], errors="coerce")
        df_clean[self.lab_col] = pd.to_numeric(df_clean[self.lab_col], errors="coerce")
        df_clean[self.inf_col] = pd.to_numeric(df_clean[self.inf_col], errors="coerce")
        df_clean = df_clean.dropna()
        df_clean = df_clean.set_index(self.time_col)
        return df_clean
    
    def _downsample_for_plotting(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Downsample data for chart rendering if needed.
        Uses simple every-Nth-point sampling for windowed data.
        """
        n_points = len(df)
        
        if n_points <= self.max_chart_points:
            return df
        
        # Calculate step size
        step = max(1, math.ceil(n_points / self.max_chart_points))
        
        print(f"Downsampling chart data: {n_points} → ~{n_points // step} points")
        
        # Keep every Nth point
        return df.iloc[::step].copy()
    
    def _agg_stats(self, series: pd.Series) -> pd.DataFrame:
        """Aggregate time series by window frequency."""
        s = pd.to_numeric(series, errors="coerce")
        out = s.resample(self.window_freq).agg(["mean", "std", "count"])
        out = out.rename(columns={"count": "n"})
        return out
    
    def _t_adaptive_pair(self, mean_a, std_a, n_a, mean_b, std_b, n_b) -> Tuple[pd.Series, pd.Series]:
        """Adaptive t-test: Welch if both vary, one-sample otherwise."""
        se = pd.Series(np.nan, index=mean_a.index, dtype=float)
        a_ok = (n_a >= 2) & np.isfinite(std_a) & np.isfinite(n_a)
        b_ok = (n_b >= 2) & np.isfinite(std_b) & np.isfinite(n_b)
        
        # Welch's t-test (both vary)
        both = a_ok & b_ok
        se.loc[both] = np.sqrt((std_b[both] ** 2 / n_b[both]) + (std_a[both] ** 2 / n_a[both]))
        
        # One-sample (only B varies)
        b_only = b_ok & ~a_ok
        se.loc[b_only] = std_b[b_only] / np.sqrt(n_b[b_only])
        
        # One-sample (only A varies)
        a_only = a_ok & ~b_ok
        se.loc[a_only] = std_a[a_only] / np.sqrt(n_a[a_only])

        t = (mean_b - mean_a) / se
        return t, se
    
    def _pair_bins(self) -> pd.DataFrame:
        """Calculate windowed statistics and t-tests."""
        A = self._agg_stats(self.df_clean[self.lab_col]).rename(
            columns={"mean": "A_avg", "std": "std_a", "n": "n_a"}
        )
        B = self._agg_stats(self.df_clean[self.inf_col]).rename(
            columns={"mean": "B_avg", "std": "std_b", "n": "n_b"}
        )
        base = pd.concat([A, B], axis=1)

        t_AB, se_AB = self._t_adaptive_pair(
            mean_a=base["A_avg"], std_a=base["std_a"], n_a=base["n_a"],
            mean_b=base["B_avg"], std_b=base["std_b"], n_b=base["n_b"],
        )
        base["t_AB"] = t_AB
        base["SE_AB"] = se_AB
        base["Diff"] = base["B_avg"] - base["A_avg"]

        base.index.name = "WindowStart"
        base = base.reset_index()
        
        # Calculate window endpoints and midpoint
        try:
            delta = pd.to_timedelta(self.window_freq)
        except Exception:
            delta = pd.Timedelta(hours=1)
        
        base["WindowEnd"] = base["WindowStart"] + delta
        base["Time"] = base["WindowStart"] + (base["WindowEnd"] - base["WindowStart"]) / 2
        
        return base
    
    def _calculate_analysis(self):
        """Calculate all statistics and prepare data for visualization."""
        # Calculate windowed stats and t-tests
        self.base = self._pair_bins()
        
        # Calculate alert flags
        self.base["AlertFlag"] = self.base["t_AB"].abs() >= float(self.t_threshold)
        self.base["DevAlert"] = self.base["Diff"].abs() >= float(self.delta_max)
        
        # Prepare summary statistics
        self.summary = {
            "total_windows": len(self.base),
            "t_alerts": int(self.base["AlertFlag"].sum()),
            "deviation_alerts": int(self.base["DevAlert"].sum()),
            "mean_deviation": float(np.nanmean(self.base["Diff"].values)),
            "std_deviation": float(np.nanstd(self.base["Diff"].values))
        }
        
        # Create downsampled version for plotting
        self.base_plot = self._downsample_for_plotting(self.base)
    
    def get_summary_metrics(self) -> Dict[str, Any]:
        """Return summary statistics for display."""
        return self.summary
    
    def render_t_threshold_chart(self):
        """Render t-threshold time series chart with Streamlit."""
        t_data = self._prepare_t_threshold_data()
        
        # Show downsampling info if applied
        if len(self.base_plot) < len(self.base):
            st.caption(f"📊 Chart: {len(self.base_plot):,} of {len(self.base):,} windows")
        
        fig = go.Figure()
        
        # Lab line
        fig.add_trace(go.Scatter(
            x=t_data["time"],
            y=t_data["lab_values"],
            mode="lines",
            name="Lab (Reference)",
            line=dict(color="#1f77b4", width=2),
            hovertemplate=(
                "<b>Lab</b><br>"
                "Lab: %{y:.4f}<br>"
                "<extra></extra>"
            )
        ))

        # Measurement line
        fig.add_trace(go.Scatter(
            x=t_data["time"],
            y=t_data["measurement_values"],
            mode="lines",
            name="Analyzer/Inferential",
            line=dict(color="#d62728", width=2),
            hovertemplate=(
                "<b>Analyzer/Inferential</b><br>"
                "Measurement: %{y:.4f}<br>"
                "<extra></extra>"
            )
        ))

        # Alert markers (use full dataset, not downsampled)
        if t_data["n_alerts"] > 0:
            alert_t_scores = t_data["t_scores"][t_data["alert_mask"]]
            
            fig.add_trace(go.Scatter(
                x=t_data["alert_times"],
                y=t_data["alert_measurement_values"],
                mode="markers",
                name=f"t-test Alert (|t| ≥ {self.t_threshold})",
                marker=dict(
                    size=12,
                    color="red",
                    symbol="x",
                    line=dict(width=2, color="darkred")
                ),
                customdata=np.column_stack([
                    t_data["alert_lab_values"],
                    alert_t_scores
                ]),
                hovertemplate=(
                    "<b>⚠️ ALERT</b><br>"
                    "Time: %{x}<br>"
                    "<b>t-score: %{customdata[1]:.2f}</b><br>"
                    f"Threshold: {self.t_threshold:.2f}<br>"
                    "<extra></extra>"
                )
            ))
        
        fig.update_layout(
            title=dict(
                text="<b>Time Series with t-test<b><br>"
                f"Alerts: {t_data['n_alerts']} -> |t| Threshold: {self.t_threshold:.2f}",
                font=dict(size=20),
                y=.91
            ),
            xaxis_title="Time (window midpoint)",
            yaxis_title="Value",
            template="plotly_white",
            width=None,
            height=500,
            hovermode="x unified",
            xaxis=dict(
                range=[t_data["time"].min(), t_data["time"].max()]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-.6,
                xanchor="center",
                x=0.1
            ))

        st.plotly_chart(fig, use_container_width=True, key="t_threshold_chart")
    
    def render_deviation_chart(self):
        """Render deviation time series chart with Streamlit."""
        dev_data = self._prepare_deviation_data()
        
        # Show downsampling info if applied
        if len(self.base_plot) < len(self.base):
            st.caption(f"📊 Chart: {len(self.base_plot):,} of {len(self.base):,} windows")

        fig = go.Figure()

        # Deviation line
        fig.add_trace(go.Scatter(
            x=dev_data["time"],
            y=dev_data["deviation"],
            mode="lines",
            name="Deviation (Meas - Lab)",
            line=dict(color="#1f77b4", width=2),
            hovertemplate=(
                'Deviation: <b>%{y:.3f}</b><br>'
                '<extra></extra>'
            )
        ))

        # Cumulative mean
        fig.add_trace(go.Scatter(
            x=dev_data["time"],
            y=dev_data["cumulative_mean"],
            mode="lines",
            name="Cumulative Average",
            line=dict(color="#ff7f0e", width=2, dash="dash"),
            hovertemplate=(
                'Cumulative Avg: <b>%{y:.3f}</b><br>'
                '<extra></extra>'
            )
        ))

        # Deviation alert markers (use full dataset for alerts)
        if dev_data["n_alerts"] > 0:
            fig.add_trace(go.Scatter(
                x=dev_data["alert_times"],
                y=dev_data["alert_deviations"],
                mode="markers",
                name=f"Deviation Alert (|Δ| ≥ {self.delta_max})",
                marker=dict(
                    size=12,
                    color="red",
                    symbol="circle",
                    line=dict(width=2, color="darkred")
                ),
                hovertemplate=(
                    '<b>⚠️ ALERT</b><br>'
                    '%{x|%Y-%m-%d %H:%M:%S}<br>'
                    'Deviation: <b>%{y:.3f}</b><br>'
                    '<extra></extra>'
                )
            ))

        fig.update_layout(
            title=dict(
                text="<b>Deviation from Lab<b><br>"
                f"Alerts: {dev_data['n_alerts']} | Threshold: ±{self.delta_max:.2f}",
                font=dict(size=20),
                y=.91
            ),
            xaxis_title="Time (window midpoint)",
            yaxis_title="Deviation",
            template="plotly_white",
            width=None,
            height=500,
            hovermode="x unified",
            xaxis=dict(
                range=[dev_data["time"].min(), dev_data["time"].max()]
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-.6,
                xanchor="center",
                x=0.1
            ))
        
        # Add threshold lines as shapes
        if len(dev_data["time"]) > 0:
            fig.add_hline(
                y=dev_data["upper_threshold"],
                line_dash="dot",
                line_color="gray",
                line_width=2,
                annotation_text=f"Δ max (+{self.delta_max})",
                annotation_position="right"
            )
            
            fig.add_hline(
                y=dev_data["lower_threshold"],
                line_dash="dot",
                line_color="gray",
                line_width=2,
                annotation_text=f"Δ max (-{self.delta_max})",
                annotation_position="right"
            )

        st.plotly_chart(fig, use_container_width=True, key="deviation_chart")
    
    def _prepare_t_threshold_data(self) -> Dict[str, Any]:
        """Prepare data for t-threshold chart (uses downsampled data for lines)."""
        # Use downsampled data for lines
        base_plot = self.base_plot
        
        # Use full data for alerts
        alert_mask = self.base["t_AB"].abs() >= float(self.t_threshold)
        alerts = self.base.loc[alert_mask]
        
        return {
            "time": base_plot["Time"].values,
            "lab_values": base_plot["A_avg"].values,
            "measurement_values": base_plot["B_avg"].values,
            "t_scores": self.base["t_AB"].values,  # Full dataset
            "alert_mask": alert_mask.values,
            "alert_times": alerts["Time"].values,
            "alert_lab_values": alerts["A_avg"].values,
            "alert_measurement_values": alerts["B_avg"].values,
            "threshold": self.t_threshold,
            "n_alerts": int(alert_mask.sum()),
            "alert_rate": float(alert_mask.mean() * 100) if len(self.base) > 0 else 0.0
        }
    
    def _prepare_deviation_data(self) -> Dict[str, Any]:
        """Prepare data for deviation chart (uses downsampled data for lines)."""
        # Use downsampled data for lines
        base_plot = self.base_plot
        deviation_plot = base_plot["Diff"].values
        cumulative_mean_plot = base_plot["Diff"].expanding(min_periods=1).mean().values
        
        # Use full data for alerts
        alert_mask = np.abs(self.base["Diff"].values) >= float(self.delta_max)
        alerts = self.base.loc[alert_mask]
        
        return {
            "time": base_plot["Time"].values,
            "deviation": deviation_plot,
            "cumulative_mean": cumulative_mean_plot,
            "alert_mask": alert_mask,
            "alert_times": alerts["Time"].values,
            "alert_deviations": alerts["Diff"].values,
            "delta_threshold": self.delta_max,
            "upper_threshold": self.delta_max,
            "lower_threshold": -self.delta_max,
            "n_alerts": int(alert_mask.sum()),
            "mean_deviation": float(np.nanmean(self.base["Diff"].values)),
            "std_deviation": float(np.nanstd(self.base["Diff"].values))
        }


# ============================================================================
# CONFIGURATION CONSTANTS
# ============================================================================

WINDOW_OPTIONS = {
    "30 seconds": "30S",
    "1 minute": "1min",
    "5 minutes": "5min",
    "10 minutes": "10min",
    "15 minutes": "15min",
    "30 minutes": "30min",
    "1 hour": "1H",
    "4 hours": "4H",
    "8 hours": "8H",
    "12 hours": "12H",
    "1 day": "1D",
    "1 week": "1W",
    "1 month": "30D",
}

DEFAULT_WINDOW = "1H"
DEFAULT_T_THRESHOLD = 2.0
DEFAULT_DELTA_MAX = 0.5
