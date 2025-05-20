import io

import pandas as pd
import plotly.graph_objects as go
import numpy as np

import streamlit as st
from datetime import datetime

st.set_page_config(layout="wide", page_title="Voyage Analysis Tool")


# Load data caching
@st.cache_data
def load_data(file_path):
    df = pd.read_excel(file_path, header=None)
    df[1] = pd.to_datetime(df[1], errors="coerce")
    df = df.sort_values(by=1).reset_index(drop=True)

    cols = [1, 2, 7, 8, 9, 15, 22, 44, 45, 48, 49]
    df_calc = df.iloc[:, cols].copy()
    df_calc.columns = [
        "date",
        "type",
        "miles_slc",
        "hours_slc",
        "minutes_slc",
        "engine_rpm",
        "propeller_pitch",
        "me_hsfo_cons",
        "me_lsfo_cons",
        "ae_hsfo_cons",
        "ae_lsfo_cons",
    ]

    # Calculate metrics
    df_calc["min_to_hrs"] = df_calc["minutes_slc"] / 60
    df_calc["total_hrs"] = df_calc["hours_slc"] + df_calc["min_to_hrs"]
    df_calc["vessel_speed"] = df_calc.apply(
        lambda row: row["miles_slc"] / row["total_hrs"] if row["total_hrs"] > 0 else 0,
        axis=1,
    )
    df_calc["engine_distance"] = df_calc.apply(
        lambda row: (row["engine_rpm"] * row["propeller_pitch"] * row["total_hrs"] * 60)
        / 1852
        if row["total_hrs"] > 0
        else 0,
        axis=1,
    )
    df_calc["slip_percentage"] = df_calc.apply(
        lambda row: (1 - row["miles_slc"] / row["engine_distance"]) * 100
        if row["engine_distance"] > 0
        else 0,
        axis=1,
    )

    return df, df_calc


def plot_metrics(df, date_range, metrics):
    all_dates = pd.date_range(start=date_range[0], end=date_range[1], freq="D")
    date_strs = [d.strftime("%Y-%m-%d") for d in all_dates]

    df_filtered = df[df["total_hrs"] > 10].copy()
    df_filtered["date_str"] = df_filtered["date"].dt.strftime("%Y-%m-%d")

    n_metrics = len(metrics)
    bar_width = 0.2
    spacing = bar_width * (n_metrics + 1.2)
    x = np.arange(len(date_strs)) * spacing

    fig = go.Figure()

    for i, metric in enumerate(metrics):
        values = df_filtered.groupby("date_str")[metric].mean().reindex(date_strs)
        x_pos = x + i * bar_width

        fig.add_trace(
            go.Bar(
                x=x_pos,
                y=values,
                name=metric,
                text=values.round(1),
                textposition="outside",
                hovertemplate=f"%{{y:.2f}}<br>{metric}<extra></extra>",
            )
        )

    # Update layout
    fig.update_layout(
        title=f"Performance Metrics ({date_range[0]} to {date_range[1]})",
        xaxis=dict(
            tickmode="array",
            tickvals=x + bar_width * (n_metrics - 1) / 2,
            ticktext=date_strs,
            tickangle=45,
        ),
        yaxis_title="Value",
        barmode="group",
        bargap=0.15,
        bargroupgap=0.05,
        legend=dict(x=1.05, y=1),
        height=600,
        width=1400,
        margin=dict(t=80, b=120),
    )

    return fig


# Main function
def main():
    st.title("Voyage Analysis Tool")

    uploaded_file = st.file_uploader("Upload Excel File", type=["xlsx"])

    if uploaded_file is not None:
        df, df_calc = load_data(uploaded_file)

        # tabs
        tab1, tab2 = st.tabs(["Overall Metrics", "Voyage Analysis"])

        with tab1:
            st.header("Overall Metrics")

            # Display fuel consumption statistics
            st.subheader("Fuel Consumption Summary")

            cols = st.columns(4)
            fuel_metrics = [
                ("ME HSFO", "me_hsfo_cons"),
                ("ME LSFO", "me_lsfo_cons"),
                ("AE HSFO", "ae_hsfo_cons"),
                ("AE LSFO", "ae_lsfo_cons"),
            ]

            for col, (label, metric) in zip(cols, fuel_metrics):
                with col:
                    st.metric(
                        label=f"{label} Consumption (tons)",
                        value=f"{round(df_calc[metric].sum(), 2):,}",
                        help=f"Total {label} consumption",
                    )

            # Display basic statistics
            st.subheader("General Information")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Total Records", len(df))
                st.metric(
                    "Average Vessel Speed",
                    f"{df_calc['vessel_speed'].mean():.2f} knots",
                )
            with col2:
                st.metric(
                    "Date Range",
                    f"{df_calc['date'].min().date()} to {df_calc['date'].max().date()}",
                )
                st.metric("Average Engine RPM", f"{df_calc['engine_rpm'].mean():.1f}")

            # Show raw data
            st.subheader("Processed Data")
            st.dataframe(
                df_calc.style.format(
                    {
                        "date": lambda x: x.strftime("%Y-%m-%d"),
                        "vessel_speed": "{:.2f}",
                        "slip_percentage": "{:.2f}%",
                    }
                )
            )

        with tab2:
            st.header("Voyage Analysis by Date Range")

            # Date range selection
            min_date = df[1].min().date()
            max_date = df[1].max().date()

            col1, col2 = st.columns(2)
            with col1:
                start_date = st.date_input(
                    "Start Date", min_date, min_value=min_date, max_value=max_date
                )
            with col2:
                end_date = st.date_input(
                    "End Date", max_date, min_value=min_date, max_value=max_date
                )

            if start_date > end_date:
                st.error("End date must be after start date")
            else:
                # Filter data
                mask = (df_calc["date"].dt.date >= start_date) & (
                    df_calc["date"].dt.date <= end_date
                )
                df_voyage = df_calc[mask].copy()

                if not df_voyage.empty:
                    # Calculate FO metrics
                    df_raw_voyage = df[
                        (df[1].dt.date >= start_date) & (df[1].dt.date <= end_date)
                    ]

                    if not df_raw_voyage.empty:
                        fo_rob_initial = df_raw_voyage.iloc[0, 4]
                        fo_rob_final = df_raw_voyage.iloc[-1, 4]
                        supplied_fo = df_raw_voyage[34].sum()
                        fo_consumed = fo_rob_initial - fo_rob_final
                        if fo_consumed < 0:
                            fo_consumed += supplied_fo

                    # Display FO summary
                    st.subheader("Fuel Oil Summary")

                    cols = st.columns(4)
                    summary_metrics = [
                        (
                            "FO ROB Initial",
                            fo_rob_initial,
                            "Initial fuel remaining on board",
                        ),
                        ("FO ROB Final", fo_rob_final, "Final fuel remaining on board"),
                        (
                            "Supplied FO",
                            supplied_fo,
                            "Total fuel supplied during voyage",
                        ),
                        (
                            "FO Consumed",
                            fo_consumed,
                            "Total fuel consumed during voyage",
                        ),
                    ]

                    for col, (label, value, help_text) in zip(cols, summary_metrics):
                        with col:
                            st.metric(
                                label=label, value=f"{value:,.2f}", help=help_text
                            )

                    # Display performance metrics as bar chart
                    st.subheader("Performance Metrics")

                    # Select metrics to display
                    default_metrics = ["total_hrs", "engine_rpm", "vessel_speed"]
                    selected_metrics = st.multiselect(
                        "Select metrics to display",
                        options=[
                            "total_hrs",
                            "engine_rpm",
                            "me_lsfo_cons",
                            "vessel_speed",
                            "slip_percentage",
                        ],
                        default=default_metrics,
                    )

                    if selected_metrics:
                        date_range = (
                            start_date.strftime("%Y-%m-%d"),
                            end_date.strftime("%Y-%m-%d"),
                        )
                        fig = plot_metrics(df_voyage, date_range, selected_metrics)
                        st.plotly_chart(fig, use_container_width=True)

                    else:
                        st.warning("Please select at least one metric to display")

                    # Show voyage data
                    st.subheader("Voyage Data")
                    st.dataframe(
                        df_voyage.style.format(
                            {
                                "date": lambda x: x.strftime("%Y-%m-%d"),
                                "vessel_speed": "{:.2f}",
                                "slip_percentage": "{:.2f}%",
                            }
                        )
                    )
                else:
                    st.warning("No data available for selected date range")


if __name__ == "__main__":
    main()
