
# from __future__ import annotations

import pandas as pd
import streamlit as st
import plotly.graph_objects as go
import matplotlib.pyplot as plt


# Rename the class here to match your 'ts = timeSeries(...)' call
class timeSeries: 
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
    
        # Logic for data cleaning
        self.df[self.time_col] = pd.to_datetime(self.df[self.time_col], errors="coerce")
        self.df = self.df.dropna(subset=[self.time_col]).sort_values(self.time_col)


    
    def timeSeries(self):
        tag_name = self.config.get('tag', 'Lab & Analyzer Values')
        unit_label = self.config.get('unit_label')
        
        st.subheader(f"{tag_name}: Time Series Comparison")

        if self.df.empty:
            st.warning("No valid data to plot.")
            return

        fig = go.Figure()

        # Add Reference Line
        fig.add_trace(go.Scatter(
            x=self.df[self.time_col],
            y=self.df[self.lab_col],
            name=f"Lab Value ({unit_label})",
            connectgaps=False,
            mode='lines'
        ))

        # Add Measurement Line
        fig.add_trace(go.Scatter(
            x=self.df[self.time_col],
            y=self.df[self.inf_col],
            name=f"Analyzer/Inf ({unit_label})",
            mode='lines'
        ))

        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title=f"Measurement ({unit_label})",
            template="plotly_white",
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,          # Makes the text larger
                font_family="Arial",
                namelength=-1          # Forces Plotly to show the full text (no cutoff)
            )
        )

        st.plotly_chart(fig, use_container_width=True)


    def dev_sample(self):
        tag_name = self.config.get('tag', 'Lab & Analyzer Values')
        unit_label = self.config.get('unit_label')

        st.subheader(f"Deviation Sample: {tag_name}")

        self.df[self.lab_col] = pd.to_numeric(self.df[self.lab_col], errors="coerce")
        self.df[self.inf_col] = pd.to_numeric(self.df[self.inf_col], errors="coerce")

        self.df["delta"] = self.df[self.inf_col] - self.df[self.lab_col]

        ninety_day = (
            self.df.sort_values(self.time_col)
           .rolling("90D", on=self.time_col, min_periods=1)["delta"]
           .mean()
        )
            
        fig = go.Figure()

        # Add Reference Line
        fig.add_trace(go.Scatter(
            x=self.df[self.time_col],
            y=self.df["delta"],
            name=f"Deviation ({unit_label})",
            connectgaps=False,
            mode='lines'
        ))

        # Add 90 day rolling Line
        fig.add_trace(go.Scatter(
            x=self.df[self.time_col],
            y=ninety_day,
            name=f"90D RAE ({unit_label})",
            connectgaps=False,
            mode='lines'
        ))

        fig.update_layout(
            xaxis_title="Timestamp",
            yaxis_title=f"Deviation ({unit_label})",
            template="plotly_white" ,
            hovermode="x unified",
            hoverlabel=dict(
                bgcolor="white",
                font_size=16,          # Makes the text larger
                font_family="Arial",
                namelength=-1          # Forces Plotly to show the full text (no cutoff)
            )
        )

        st.plotly_chart(fig, use_container_width=True)

    def render(self):
        self.timeSeries()
        self.dev_sample()
