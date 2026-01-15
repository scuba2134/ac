import streamlit as st
import pandas as pd
import numpy as np
from utils.data_io import load_data_profile, load_dataframe_from_profile


class mainPage:
    def __init__(self):
        """
        Builds summary table from profiles.
        All thresholds come from data_profile.json
        """
        self.df = None
        
    # COMMENTED OUT FOR FUTURE USE: T-statistic calculation
    # def calculate_t_statistic(self, ref_series: pd.Series, meas_series: pd.Series) -> float:
    #     """
    #     Calculate t-statistic for comparing two series.
    #     Uses Welch's t-test (unequal variances).
    #     """
    #     ref_clean = pd.to_numeric(ref_series, errors='coerce').dropna()
    #     meas_clean = pd.to_numeric(meas_series, errors='coerce').dropna()
    #     
    #     if len(ref_clean) < 2 or len(meas_clean) < 2:
    #         return np.nan
    #     
    #     mean_ref = ref_clean.mean()
    #     mean_meas = meas_clean.mean()
    #     std_ref = ref_clean.std()
    #     std_meas = meas_clean.std()
    #     n_ref = len(ref_clean)
    #     n_meas = len(meas_clean)
    #     
    #     # Standard error for Welch's t-test
    #     se = np.sqrt((std_ref**2 / n_ref) + (std_meas**2 / n_meas))
    #     
    #     if se == 0:
    #         return np.nan
    #     
    #     t_stat = (mean_meas - mean_ref) / se
    #     return abs(t_stat)
        
    def build_summary_from_profiles(self) -> pd.DataFrame:
        """
        Build complete summary table by loading all profiles and calculating statistics.
        Alert logic based only on delta_max threshold.
        """
        profiles = load_data_profile()
        summary_rows = []
        
        for profile in profiles:
            try:
                # Get configuration from profile
                tag = profile.get("tag", "Unknown")
                measurement_col = profile.get("measurement_column", "")
                unit_label = profile.get("unit_label", "")
                
                # Get delta_max threshold from profile (required)
                delta_max = profile.get("delta_max")
                # alert_threshold = profile.get("alert_threshold")  # Kept for future use
                
                # Check if threshold is defined
                if delta_max is None:
                    summary_rows.append({
                        "Tag": tag,
                        "Analyser/Tagname": measurement_col,
                        "Avg Lab": "N/A",
                        "Avg Measurement": "N/A",
                        "Avg Deviation": "N/A",
                        "Delta Threshold": "Not configured",
                        "Alert": "⚠ No Threshold"
                    })
                    print(f"Warning: {tag} missing delta_max in profile")
                    continue
                
                # Load the full dataset
                df_data, _ = load_dataframe_from_profile(profile)
                
                # Convert to numeric
                df_data['Reference'] = pd.to_numeric(df_data['Reference'], errors='coerce')
                df_data['Measurement'] = pd.to_numeric(df_data['Measurement'], errors='coerce')
                
                # Drop NaN values
                df_data = df_data.dropna(subset=['Reference', 'Measurement'])
                
                # Calculate statistics
                avg_lab = df_data['Reference'].mean()
                avg_measurement = df_data['Measurement'].mean()
                avg_deviation = (df_data['Measurement'] - df_data['Reference']).mean()
                
                # Calculate deviation alerts (delta_max threshold)
                df_data['Deviation'] = (df_data['Measurement'] - df_data['Reference']).abs()
                deviation_alerts = (df_data['Deviation'] >= delta_max).sum()
                max_deviation = df_data['Deviation'].max()
                
                # COMMENTED OUT FOR FUTURE USE: T-statistic alerts
                # t_stat = self.calculate_t_statistic(
                #     df_data['Reference'], 
                #     df_data['Measurement']
                # )
                # t_alert = t_stat >= alert_threshold if not np.isnan(t_stat) else False
                
                # Determine alert status (based only on delta_max)
                if deviation_alerts > 0:
                    alert_status = f"⚠️ Alert ({deviation_alerts})"
                else:
                    alert_status = "✓ Normal"
                
                # Build row
                summary_rows.append({
                    "Tag": tag,
                    "Analyser/Tagname": measurement_col,
                    "Avg Lab": f"{avg_lab:.4f} {unit_label}" if not np.isnan(avg_lab) else "N/A",
                    "Avg Measurement": f"{avg_measurement:.4f} {unit_label}" if not np.isnan(avg_measurement) else "N/A",
                    "Avg Deviation": f"{avg_deviation:.4f} {unit_label}" if not np.isnan(avg_deviation) else "N/A",
                    "Delta Threshold": f"{delta_max} {unit_label}",
                    "Alert": alert_status
                    # FUTURE: Can add back t-statistic column here
                    # "t-statistic": f"{t_stat:.2f}" if not np.isnan(t_stat) else "N/A",
                })
                
            except Exception as e:
                # If loading fails, add error row
                summary_rows.append({
                    "Tag": profile.get("tag", "Unknown"),
                    "Analyser/Tagname": profile.get("measurement_column", ""),
                    "Avg Lab": "Error",
                    "Avg Measurement": "Error",
                    "Avg Deviation": "Error",
                    "Delta Threshold": "",
                    "Alert": "⚠ Error"
                })
                print(f"Error processing {profile.get('tag', 'Unknown')}: {e}")
                import traceback
                traceback.print_exc()
        
        return pd.DataFrame(summary_rows)

    def render(self):
        # Build summary table from underlying data
        with st.spinner("Loading data and calculating statistics..."):
            self.df = self.build_summary_from_profiles()
        
        # Count alerts (only deviation-based alerts)
        alert_count = len(self.df[self.df['Alert'].str.contains("⚠️ Alert", na=False, regex=False)])
        color = "#FF4B4B" if alert_count > 0 else "#28a745"
        
        # Display metrics
        cols = st.columns(6)

        with cols[0]:
            st.markdown(f"""
                <div style="
                    background-color: {color};
                    padding: 10px;
                    border-radius: 8px;
                    color: white !important;
                    text-align: center;
                    height: 80px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center !important;
                ">
                    <p style="margin: 0; font-size: 14px; font-weight: bold; opacity: 0.9;">TOTAL ALERTS</p>
                    <p style="margin: 0; font-size: 20px; font-weight: bold; line-height: 1;">{alert_count}</p>
                </div>
                """, unsafe_allow_html=True)
        
        with cols[1]:
            total_analyzers = len(self.df)
            st.markdown(f"""
                <div style="
                    background-color: #007bff;
                    padding: 10px;
                    border-radius: 8px;
                    color: white !important;
                    text-align: center;
                    height: 80px;
                    display: flex;
                    flex-direction: column;
                    justify-content: center !important;
                ">
                    <p style="margin: 0; font-size: 14px; font-weight: bold; opacity: 0.9;">ANALYZERS</p>
                    <p style="margin: 0; font-size: 20px; font-weight: bold; line-height: 1;">{total_analyzers}</p>
                </div>
                """, unsafe_allow_html=True)
            
        st.markdown("\n\n")
            
        # Render the table with styled alerts
        def highlight_alerts(row):
            alert_val = str(row.get('Alert', ''))
            if "⚠️ Alert" in alert_val:
                return ['background-color: #ffcccc'] * len(row)
            elif "✓ Normal" in alert_val:
                return ['background-color: #d4edda'] * len(row)
            elif "⚠ Error" in alert_val or "⚠ No Threshold" in alert_val:
                return ['background-color: #fff3cd'] * len(row)
            return [''] * len(row)
        
        styled_df = self.df.style.apply(highlight_alerts, axis=1)
        
        st.dataframe(
            styled_df,
            use_container_width=True,
            hide_index=True,
            on_select="rerun",
            selection_mode="single-row",
            key="main_table",
        )

        # Handle row selection for navigation
        selection = st.session_state.get("main_table", {}).get("selection", {})
        selected_rows = selection.get("rows", [])

        if selected_rows:
            idx = selected_rows[0]
            target_id = str(self.df.iloc[idx]["Analyser/Tagname"]).strip()
            
            st.session_state["selected_measurement"] = target_id
            st.session_state["page"] = 2
            st.session_state["main_table"]["selection"]["rows"] = []
            
            st.rerun()