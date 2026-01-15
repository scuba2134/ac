
import pandas as pd
import streamlit as st
import numpy as np
from notebooks.mainPage import mainPage
from notebooks.plotsView import plotsView
from notebooks.summary import summary
import plotly.graph_objects as go
from utils.data_io import load_data_profile, load_dataframe_from_profile
from notebooks.notebook_compare import notebook_compare


def _inject_styles() -> None:
    st.markdown(
        """
        <style>
            /* Make the navigation button big */
            div.stButton > button {
                font-size: 2rem;
                padding: 1.25rem 1.5rem;
                width: min(42rem, 100%);
                margin: 0 auto;
                display: block;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


def _page_one() -> None:
    st.header("Main View")
    
    # No parameters needed - all config comes from profiles
    view = mainPage()
    view.render()

def _page_two() -> None:
    col1, spacer = st.columns([1, 8])
    
    with col1:
        if st.button("← Go back"):
            st.session_state["page"] = 1
            st.rerun()

    clicked_meas = st.session_state.get("selected_measurement")
    
    if clicked_meas:
        profiles = load_data_profile()
        
        profile = next(
            (p for p in profiles if str(p.get("measurement_column")).strip() == clicked_meas), 
            None
        )
        
        if profile:
            # LOAD DATA
            df, descriptions = load_dataframe_from_profile(profile)
            
            # CREATE CONFIG
            config = {
                "tag": profile.get("tag"),
                "unit_label": profile.get("unit_label"),
                "descriptions": descriptions,
                "reference_column": profile.get("reference_column"),
                "measurement_column": profile.get("measurement_column"),
                "delta_max": profile.get("delta_max")
            }

            tag = profile.get("tag")
            st.title(f"{tag}: Comparison Analysis")

            # ========== CONTROLS ==========
            col1, col2, col3, spacer = st.columns([1.5, 1.5, 1.5, 2.5])

            with col1:
                window_freq = st.selectbox(
                    "Window",
                    options=["30S", "1min", "5min", "10min", "15min", "30min", "1H", "4H", "8H", "12H", "1D", "1W", "30D"],
                    format_func=lambda x: {
                        "30S": "30 seconds", "1min": "1 minute", "5min": "5 minutes",
                        "10min": "10 minutes", "15min": "15 minutes", "30min": "30 minutes",
                        "1H": "1 hour", "4H": "4 hours", "8H": "8 hours",
                        "12H": "12 hours", "1D": "1 day", "1W": "1 week", "30D": "1 month"
                    }.get(x, x),
                    index=10
                )

            with col2:
                t_threshold = st.slider("t-threshold", 0.0, 10.0, 2.0, 0.5)

            with col3:
                delta_max = st.slider("Delta max", 0.0, 11.0, 2.0, 0.5)

            # ========== CREATE COMPARISON CLASS INSTANCE ==========
            compare = notebook_compare(
                df=df,
                time_col="Timestamp",
                lab_col="Reference",
                inf_col="Measurement",
                config=config,
                window_freq=window_freq,
                t_threshold=t_threshold,
                delta_max=delta_max
            )

            # ========== SUMMARY METRICS ==========
            metric = compare.get_summary_metrics()
            col_a, col_b, col_c, col_d, spacer = st.columns([1.5, 1.5, 1.5, 1.5, 3])

            with col_a:
                st.metric("Total Windows", metric["total_windows"])

            with col_b:
                st.metric("t-test Alerts", metric["t_alerts"])

            with col_c:
                st.metric("Deviation Alerts", metric["deviation_alerts"])

            with col_d:
                mean_dev = metric.get('mean_deviation', 0.0)
                if pd.isna(mean_dev):
                    st.metric("Mean Deviation", "N/A")
                else:
                    st.metric("Mean Deviation", f"{mean_dev:.3f}")

            # ========== First Row PLOTS ==========
            col1, spacer, col2, spacer, col3= st.columns([5, .5, 5, .5, 5])

            # New comparison charts using the class
            with col1:
                compare.render_t_threshold_chart()

            with col2:
                compare.render_deviation_chart()

            # Summary view
            with col3:
                st.markdown("<br>", unsafe_allow_html=True)
                su = summary(
                    df=df,
                    time_col="Timestamp", 
                    lab_col="Reference", 
                    inf_col="Measurement",
                    config=config
                )
                su.render()
            

            # ========== Second Row PLOTS ==========
            col1, spacer, col2, spacer, col3= st.columns([5, .5, 5, .5, 5])

            with col1:
                pl = plotsView(
                    df=df,
                    time_col="Timestamp", 
                    lab_col="Reference", 
                    inf_col="Measurement",
                    config=config
                )
                pl.XY_plot()
            
            with col2:
                pl.lab_vs_deviation_plot()

    else:
        st.error(f"No profile found for measurement: {clicked_meas}")


def main() -> None:
    st.set_page_config(
        page_title="ALI-C",
        layout="wide",
        initial_sidebar_state="collapsed",
     )
    _inject_styles()
    st.session_state.setdefault("page", 1)
    
    if st.session_state["page"] == 1:
        _page_one()
    else:
        _page_two()

if __name__ == "__main__":
    main()