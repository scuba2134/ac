

from __future__ import annotations
from typing import Optional, Tuple, Dict, Any, List
import pandas as pd
import streamlit as st
import plotly.graph_objects as go


class summary: 
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

    def summary_data(self) -> dict:
        units = self.config.get('unit_label')
        samplePt = self.config.get('reference_column')
        analyzer = self.config.get('measurement_column')
        
        # Use pd.to_numeric to ensure these are floats, not strings
        try:
            _inf = pd.to_numeric(self.df[self.inf_col], errors='coerce')
            _lab = pd.to_numeric(self.df[self.lab_col], errors='coerce')
            
            # Perform the math using the forced numeric versions
            self.df['delta'] = _inf - _lab
            
        except Exception as e:
            st.error(f"Math Error: Could not subtract columns. {e}")
            return
        
        max_error = self.df['delta'].abs().max()
        avg_error = self.df['delta'].mean()

        # Most recent readings
        recent_timestamp = self.df[self.time_col].iloc[-1]
        recent_lab = _lab.iloc[-1]      # Use converted numeric value
        recent_inf = _inf.iloc[-1]      # Use converted numeric value

        # Identify Max Error Row
        idx_max = self.df['delta'].abs().idxmax()
        worst_row = self.df.loc[idx_max]  

        # Table of key points  
        metrics = {
            "Sample Point": samplePt,
            "Sample": None,
            "Analysis": None,
            "Functionality": units,
            "Descriptor": None,
            "Analyzer": analyzer,
            "Latest Timestamp": f"{recent_timestamp}",
            "Latest Lab": f"{recent_lab:.4f} {units}",
            "Latest Inference": f"{recent_inf:.4f} {units}",
        }

       # Convert to DataFrame
        summary_df = pd.DataFrame(list(metrics.items()),columns=['Data Point', 'Value'])

        # Render in Streamlit
        st.markdown("#### Summary")
        st.markdown("<style>h3 {margin-bottom: 0px;}</style>", unsafe_allow_html=True)

        st.dataframe(
            summary_df,
            use_container_width=True,
            hide_index=True,
            height= 350,
            width=350,
        )

    def render(self):
        self.summary_data()
