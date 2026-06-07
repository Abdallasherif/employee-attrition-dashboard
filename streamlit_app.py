from __future__ import annotations

import base64
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import streamlit.components.v1 as components


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Dataset" / "cleaned.csv"
LOGO_PATH = BASE_DIR / "Assets" / "logo_full_black.svg"
HEADER_LOGO_PATH = BASE_DIR / "Assets" / "download.png"
ICON_PATH = BASE_DIR / "Assets" / "images.png"


@dataclass(frozen=True)
class KayfaBrand:
    PRIMARY: str = "#4751D5"
    PRIMARY_DARK: str = "#2E37A8"
    PRIMARY_LIGHT: str = "#8A91F2"
    PRIMARY_SOFT: str = "#ECEEFB"
    SURFACE: str = "#F6F6FB"
    CARD: str = "#FFFFFF"
    TEXT: str = "#14132B"
    MUTED: str = "#67667E"
    BORDER: str = "#E6E5F2"
    ACCENT: str = "#FF6B4A"
    TEAL: str = "#119C8B"
    AMBER: str = "#E8930C"

    FONT_STACK: str = "DM Sans, system-ui, -apple-system, BlinkMacSystemFont, sans-serif"
    DISPLAY_STACK: str = "DM Serif Display, Georgia, serif"

    COLORWAY: tuple[str, ...] = (
        PRIMARY,
        ACCENT,
        TEAL,
        AMBER,
        PRIMARY_LIGHT,
    )

    WLB_MAP = {0: "Poor", 1: "Fair", 2: "Good", 3: "Excellent"}
    JOB_SAT_MAP = {0: "Low", 1: "Medium", 2: "High", 3: "Very High"}
    PERF_MAP = {0: "Low", 1: "Below Average", 2: "Average", 3: "High"}
    EDU_MAP = {
        0: "High School",
        1: "Associate Degree",
        2: "Bachelor's Degree",
        3: "Master's Degree",
        4: "PhD",
    }
    JOB_LEVEL_MAP = {0: "Entry", 1: "Mid", 2: "Senior"}
    COMPANY_SIZE_MAP = {0: "Small", 1: "Medium", 2: "Large"}
    BINARY_MAP = {0: "No", 1: "Yes"}
    ATTRITION_MAP = {0: "Stayed", 1: "Left"}


@st.cache_data(show_spinner=False)
def load_data(csv_path: str) -> pd.DataFrame:
    return pd.read_csv(csv_path)


class AttritionAnalyzer:
    def __init__(self, raw_df: pd.DataFrame):
        self.raw_df = raw_df.copy()
        self.display_df = self._build_display_frame(self.raw_df)

    @staticmethod
    def _build_display_frame(df: pd.DataFrame) -> pd.DataFrame:
        display_df = df.copy()
        display_df["attrition_label"] = display_df["attrition"].map(KayfaBrand.ATTRITION_MAP)
        display_df["work_life_balance_label"] = display_df["work_life_balance"].map(KayfaBrand.WLB_MAP)
        display_df["job_satisfaction_label"] = display_df["job_satisfaction"].map(KayfaBrand.JOB_SAT_MAP)
        display_df["performance_rating_label"] = display_df["performance_rating"].map(KayfaBrand.PERF_MAP)
        display_df["education_level_label"] = display_df["education_level"].map(KayfaBrand.EDU_MAP)
        display_df["job_level_label"] = display_df["job_level"].map(KayfaBrand.JOB_LEVEL_MAP)
        display_df["company_size_label"] = display_df["company_size"].map(KayfaBrand.COMPANY_SIZE_MAP)
        display_df["overtime_label"] = display_df["overtime"].map(KayfaBrand.BINARY_MAP)
        display_df["remote_work_label"] = display_df["remote_work"].map(KayfaBrand.BINARY_MAP)
        display_df["leadership_opportunities_label"] = display_df["leadership_opportunities"].map(KayfaBrand.BINARY_MAP)
        display_df["innovation_opportunities_label"] = display_df["innovation_opportunities"].map(KayfaBrand.BINARY_MAP)
        display_df["company_reputation_label"] = display_df["company_reputation"].map(KayfaBrand.WLB_MAP)
        display_df["employee_recognition_label"] = display_df["employee_recognition"].map(KayfaBrand.JOB_SAT_MAP)
        return display_df

    def filtered(self, filters: dict[str, list[str]]) -> pd.DataFrame:
        filtered = self.display_df.copy()
        for column, selections in filters.items():
            if selections:
                filtered = filtered[filtered[column].isin(selections)]
        return filtered

    @staticmethod
    def format_percent(value: float) -> str:
        return f"{value:.1f}%"

    @staticmethod
    def style_figure(fig: go.Figure, height: int = 420, title: str | None = None) -> go.Figure:
        fig.update_layout(
            template="plotly_white",
            height=height,
            title=dict(text=title, x=0.02, xanchor="left", font=dict(size=22, family=KayfaBrand.DISPLAY_STACK)),
            font=dict(family=KayfaBrand.FONT_STACK, color=KayfaBrand.TEXT),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=18, r=18, t=70, b=20),
            colorway=list(KayfaBrand.COLORWAY),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        fig.update_xaxes(showgrid=False, linecolor=KayfaBrand.BORDER, zeroline=False)
        fig.update_yaxes(gridcolor=KayfaBrand.BORDER, linecolor=KayfaBrand.BORDER)
        return fig

    def kpis(self, df: pd.DataFrame) -> dict[str, str]:
        total = len(df)
        attrition_rate = df["attrition"].mean() * 100 if total else 0
        avg_income = df["monthly_income"].mean() if total else 0
        avg_tenure = df["years_at_company"].mean() if total else 0
        avg_age = df["age"].mean() if total else 0
        left = int(df["attrition"].sum()) if total else 0
        return {
            "employees": f"{total:,}",
            "attrition_rate": self.format_percent(attrition_rate),
            "left": f"{left:,}",
            "avg_income": f"{avg_income:,.0f}",
            "avg_tenure": f"{avg_tenure:.1f}",
            "avg_age": f"{avg_age:.1f}",
        }

    def chart_attrition_pie(self, df: pd.DataFrame) -> go.Figure:
        counts = df["attrition_label"].value_counts().reindex(["Stayed", "Left"]).fillna(0)
        fig = px.pie(
            values=counts.values,
            names=counts.index,
            hole=0.62,
            color=counts.index,
            color_discrete_map={"Stayed": KayfaBrand.PRIMARY, "Left": KayfaBrand.ACCENT},
        )
        fig.update_traces(
            textposition="inside",
            textinfo="percent+label",
            marker=dict(line=dict(color="white", width=2)),
        )
        fig.update_layout(showlegend=False)
        fig.add_annotation(
            text=f"<b>{self.format_percent(df['attrition'].mean() * 100)}</b><br><span style='font-size:12px;color:{KayfaBrand.MUTED}'>Attrition</span>",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=16, color=KayfaBrand.TEXT),
        )
        return self.style_figure(fig, title="Overall Attrition Mix")

    def chart_job_role_attrition(self, df: pd.DataFrame) -> go.Figure:
        series = (
            df.groupby("job_role", observed=False)["attrition"]
            .mean()
            .sort_values(ascending=False)
            .mul(100)
            .reset_index(name="attrition_rate")
        )
        fig = px.bar(
            series,
            x="job_role",
            y="attrition_rate",
            text=series["attrition_rate"].map(lambda x: f"{x:.1f}%"),
            color="attrition_rate",
            color_continuous_scale=[KayfaBrand.PRIMARY_SOFT, KayfaBrand.PRIMARY, KayfaBrand.PRIMARY_DARK],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False, showlegend=False)
        fig.update_yaxes(title_text="Attrition Rate (%)")
        fig.update_xaxes(title_text="")
        return self.style_figure(fig, title="Attrition by Job Role")

    def chart_pay_fairness(self, df: pd.DataFrame) -> go.Figure:
        level_order = ["Entry", "Mid", "Senior"]
        band_order = ["Lowest", "Lower", "Middle", "Upper", "Highest"]
        color_map = {
            "Entry": KayfaBrand.PRIMARY,
            "Mid": KayfaBrand.ACCENT,
            "Senior": KayfaBrand.TEAL,
        }

        fig = make_subplots(
            rows=1,
            cols=3,
            subplot_titles=tuple(level_order),
            horizontal_spacing=0.10,
        )

        for col_index, level in enumerate(level_order, start=1):
            sub = df[df["job_level_label"] == level].copy()
            if sub.empty:
                series = pd.DataFrame({"pay_band": band_order, "attrition_rate": [0.0] * len(band_order)})
            elif sub["monthly_income"].nunique() < 2:
                series = pd.DataFrame(
                    {
                        "pay_band": ["All"],
                        "attrition_rate": [sub["attrition"].mean() * 100],
                    }
                )
            else:
                _, bins = pd.qcut(
                    sub["monthly_income"],
                    q=min(5, sub["monthly_income"].nunique()),
                    retbins=True,
                    duplicates="drop",
                )
                band_count = max(1, len(bins) - 1)
                labels = band_order[:band_count]
                sub["pay_band"] = pd.qcut(
                    sub["monthly_income"],
                    q=band_count,
                    labels=labels,
                    duplicates="drop",
                )
                series = (
                    sub.groupby("pay_band", observed=False)["attrition"]
                    .mean()
                    .reindex(labels)
                    .mul(100)
                    .reset_index(name="attrition_rate")
                )
            fig.add_trace(
                go.Scatter(
                    x=series["pay_band"],
                    y=series["attrition_rate"],
                    mode="lines+markers+text",
                    line=dict(color=color_map[level], width=3),
                    marker=dict(size=10),
                    text=series["attrition_rate"].map(lambda x: f"{x:.1f}%"),
                    textposition="top center",
                    name=level,
                ),
                row=1,
                col=col_index,
            )
            fig.update_xaxes(title_text="Pay band", row=1, col=col_index)
            fig.update_yaxes(title_text="Attrition Rate (%)" if col_index == 1 else "", row=1, col=col_index)

        fig.update_layout(showlegend=False)
        return self.style_figure(fig, height=470, title="Attrition by Pay Band")

    def chart_retention_timeline(self, df: pd.DataFrame) -> go.Figure:
        yearly = (
            df.groupby("years_at_company", observed=False)["attrition"]
            .mean()
            .mul(100)
            .reset_index(name="attrition_rate")
            .sort_values("years_at_company")
        )
        yearly["rolling"] = yearly["attrition_rate"].rolling(window=3, center=True, min_periods=1).mean()
        yearly_focus = yearly[yearly["years_at_company"] <= 20].copy()

        tenure_order = ["0-2 years (New)", "2-5 years (Established)", "5-10 years (Loyal)", "10+ years (Veteran)"]
        tenure_group = (
            df.groupby("tenure_group", observed=False)["attrition"]
            .mean()
            .reindex(tenure_order)
            .mul(100)
            .reset_index(name="attrition_rate")
        )

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Attrition by Year at Company", "Attrition by Tenure Group"),
            horizontal_spacing=0.12,
        )
        fig.add_trace(
            go.Scatter(
                x=yearly_focus["years_at_company"],
                y=yearly_focus["attrition_rate"],
                mode="lines+markers",
                line=dict(color=KayfaBrand.PRIMARY, width=3),
                marker=dict(size=7),
                name="Annual attrition",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=yearly_focus["years_at_company"],
                y=yearly_focus["rolling"],
                mode="lines",
                line=dict(color=KayfaBrand.ACCENT, width=3, dash="dash"),
                name="3-year rolling average",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=tenure_group["tenure_group"],
                y=tenure_group["attrition_rate"],
                marker_color=[KayfaBrand.PRIMARY, KayfaBrand.ACCENT, KayfaBrand.PRIMARY_LIGHT, KayfaBrand.TEAL],
                text=tenure_group["attrition_rate"].map(lambda x: f"{x:.1f}%"),
                textposition="outside",
                name="Tenure group",
            ),
            row=1,
            col=2,
        )
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text="Years at company", row=1, col=1)
        fig.update_yaxes(title_text="Attrition Rate (%)", row=1, col=1)
        fig.update_xaxes(title_text="", row=1, col=2)
        fig.update_yaxes(title_text="Attrition Rate (%)", row=1, col=2)
        return self.style_figure(fig, height=470, title="Attrition Across Tenure")

    def chart_engagement_heatmap(self, df: pd.DataFrame) -> go.Figure:
        wlb_order = ["Poor", "Fair", "Good", "Excellent"]
        sat_order = ["Low", "Medium", "High", "Very High"]
        pivot = (
            df.pivot_table(
                index="work_life_balance_label",
                columns="job_satisfaction_label",
                values="attrition",
                aggfunc="mean",
            )
            .reindex(index=wlb_order, columns=sat_order)
            .mul(100)
        )
        fig = go.Figure(
            go.Heatmap(
                z=pivot.values,
                x=pivot.columns,
                y=pivot.index,
                colorscale=[
                    [0.0, "#ECEEFB"],
                    [0.5, "#8A91F2"],
                    [1.0, "#FF6B4A"],
                ],
                text=pivot.round(1).astype(str).values,
                texttemplate="%{text}%",
                textfont={"size": 12, "color": "#14132B"},
                hovertemplate="WLB=%{y}<br>Satisfaction=%{x}<br>Attrition=%{z:.1f}%<extra></extra>",
                colorbar=dict(title="Attrition %"),
            )
        )
        fig.update_layout(
            height=470,
            title=dict(text="Work-Life Balance vs Satisfaction", x=0.02, xanchor="left", font=dict(size=22, family=KayfaBrand.DISPLAY_STACK)),
            font=dict(family=KayfaBrand.FONT_STACK, color=KayfaBrand.TEXT),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=18, r=18, t=70, b=20),
        )
        return fig

    def chart_life_stage(self, df: pd.DataFrame) -> go.Figure:
        age_order = ["<30 (Young)", "30-44 (Mid-Career)", "45+ (Senior)"]
        marital_order = ["Single", "Married", "Divorced"]
        age_marital = (
            df.pivot_table(
                index="age_group",
                columns="marital_status",
                values="attrition",
                aggfunc="mean",
            )
            .reindex(index=age_order, columns=marital_order)
            .mul(100)
        )
        dependents = (
            df.groupby("number_of_dependents", observed=False)["attrition"]
            .mean()
            .mul(100)
            .reset_index(name="attrition_rate")
        )
        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Age Group x Marital Status", "Attrition by Number of Dependents"),
            horizontal_spacing=0.16,
        )
        fig.add_trace(
            go.Heatmap(
                z=age_marital.values,
                x=age_marital.columns,
                y=age_marital.index,
                colorscale=[
                    [0.0, "#ECEEFB"],
                    [0.5, "#8A91F2"],
                    [1.0, "#FF6B4A"],
                ],
                text=age_marital.round(1).astype(str).values,
                texttemplate="%{text}%",
                hovertemplate="Age=%{y}<br>Marital=%{x}<br>Attrition=%{z:.1f}%<extra></extra>",
                colorbar=dict(title="Attrition %"),
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=dependents["number_of_dependents"],
                y=dependents["attrition_rate"],
                marker_color=KayfaBrand.PRIMARY,
                text=dependents["attrition_rate"].map(lambda x: f"{x:.1f}%"),
                textposition="outside",
            ),
            row=1,
            col=2,
        )
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text="", row=1, col=1)
        fig.update_yaxes(title_text="", row=1, col=1)
        fig.update_xaxes(title_text="Number of dependents", row=1, col=2)
        fig.update_yaxes(title_text="Attrition Rate (%)", row=1, col=2)
        return self.style_figure(fig, height=560, title="Life Stage Patterns")

    def chart_career_stagnation(self, df: pd.DataFrame) -> go.Figure:
        promo_order = [0, 1, 2, 3, 4]
        promo_rates = (
            df.groupby("number_of_promotions", observed=False)["attrition"]
            .mean()
            .reindex(promo_order)
            .mul(100)
            .reset_index(name="attrition_rate")
        )
        level_rates = (
            df.groupby("job_level", observed=False)["attrition"]
            .mean()
            .sort_index()
            .mul(100)
            .reset_index(name="attrition_rate")
        )
        level_rates["job_level"] = level_rates["job_level"].map(KayfaBrand.JOB_LEVEL_MAP)
        leadership = (
            df.groupby("leadership_opportunities", observed=False)["attrition"]
            .mean()
            .sort_index()
            .mul(100)
            .reset_index(name="attrition_rate")
        )
        leadership["leadership_opportunities"] = leadership["leadership_opportunities"].map(KayfaBrand.BINARY_MAP)
        innovation = (
            df.groupby("innovation_opportunities", observed=False)["attrition"]
            .mean()
            .sort_index()
            .mul(100)
            .reset_index(name="attrition_rate")
        )
        innovation["innovation_opportunities"] = innovation["innovation_opportunities"].map(KayfaBrand.BINARY_MAP)

        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=("Promotions", "Job Level", "Leadership Opportunities", "Innovation Opportunities"),
            horizontal_spacing=0.12,
            vertical_spacing=0.18,
        )
        fig.add_trace(
            go.Bar(
                x=promo_rates["number_of_promotions"],
                y=promo_rates["attrition_rate"],
                marker_color=KayfaBrand.PRIMARY,
                text=promo_rates["attrition_rate"].map(lambda x: f"{x:.1f}%"),
                textposition="outside",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=level_rates["job_level"],
                y=level_rates["attrition_rate"],
                marker_color=[KayfaBrand.PRIMARY, KayfaBrand.ACCENT, KayfaBrand.TEAL],
                text=level_rates["attrition_rate"].map(lambda x: f"{x:.1f}%"),
                textposition="outside",
            ),
            row=1,
            col=2,
        )
        fig.add_trace(
            go.Bar(
                x=leadership["leadership_opportunities"],
                y=leadership["attrition_rate"],
                marker_color=[KayfaBrand.ACCENT, KayfaBrand.PRIMARY],
                text=leadership["attrition_rate"].map(lambda x: f"{x:.1f}%"),
                textposition="outside",
            ),
            row=2,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=innovation["innovation_opportunities"],
                y=innovation["attrition_rate"],
                marker_color=[KayfaBrand.ACCENT, KayfaBrand.PRIMARY],
                text=innovation["attrition_rate"].map(lambda x: f"{x:.1f}%"),
                textposition="outside",
            ),
            row=2,
            col=2,
        )
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text="", row=1, col=1)
        fig.update_xaxes(title_text="", row=1, col=2)
        fig.update_xaxes(title_text="", row=2, col=1)
        fig.update_xaxes(title_text="", row=2, col=2)
        fig.update_yaxes(title_text="Attrition Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Attrition Rate (%)", row=1, col=2)
        fig.update_yaxes(title_text="Attrition Rate (%)", row=2, col=1)
        fig.update_yaxes(title_text="Attrition Rate (%)", row=2, col=2)
        return self.style_figure(fig, height=650, title="Growth and Mobility Signals")

    def chart_profile_vs_company(self, baseline_rate: float, profile_rate: float) -> go.Figure:
        fig = px.bar(
            x=["Company average", "Highest-risk profile"],
            y=[baseline_rate, profile_rate],
            text=[f"{baseline_rate:.1f}%", f"{profile_rate:.1f}%"],
            color=["Company average", "Highest-risk profile"],
            color_discrete_map={"Company average": KayfaBrand.PRIMARY_LIGHT, "Highest-risk profile": KayfaBrand.ACCENT},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="Attrition Rate (%)")
        return self.style_figure(fig, title="High-Risk Profile vs Company Average")

    def chart_action_levers(self, df: pd.DataFrame) -> go.Figure:
        baseline = df["attrition"].mean() * 100
        metrics = [
            ("Work-life balance: Poor", df[df["work_life_balance"] == 0]["attrition"].mean() * 100 - baseline),
            ("Remote work: No", df[df["remote_work"] == 0]["attrition"].mean() * 100 - baseline),
            ("Overtime: Yes", df[df["overtime"] == 1]["attrition"].mean() * 100 - baseline),
        ]
        metrics = sorted(metrics, key=lambda item: item[1], reverse=True)
        fig = px.bar(
            x=[m[0] for m in metrics],
            y=[m[1] for m in metrics],
            text=[f"+{m[1]:.1f} pp" for m in metrics],
            color=[m[1] for m in metrics],
            color_continuous_scale=[KayfaBrand.PRIMARY_SOFT, KayfaBrand.PRIMARY, KayfaBrand.ACCENT],
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(coloraxis_showscale=False, showlegend=False)
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="Lift above company average (pp)")
        return self.style_figure(fig, title="Potential Impact of Next-Step Levers")

    def chart_attrition_by_overtime(self, df: pd.DataFrame) -> go.Figure:
        series = (
            df.groupby("overtime_label", observed=False)["attrition"]
            .mean()
            .reindex(["No", "Yes"])
            .mul(100)
            .reset_index(name="attrition_rate")
        )
        fig = px.bar(
            series,
            x="overtime_label",
            y="attrition_rate",
            text=series["attrition_rate"].map(lambda x: f"{x:.1f}%"),
            color="overtime_label",
            color_discrete_map={"No": KayfaBrand.PRIMARY, "Yes": KayfaBrand.ACCENT},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text="Overtime")
        fig.update_yaxes(title_text="Attrition Rate (%)")
        return self.style_figure(fig, title="Attrition by Overtime")

    def chart_attrition_by_remote_work(self, df: pd.DataFrame) -> go.Figure:
        series = (
            df.groupby("remote_work_label", observed=False)["attrition"]
            .mean()
            .reindex(["No", "Yes"])
            .mul(100)
            .reset_index(name="attrition_rate")
        )
        fig = px.bar(
            series,
            x="remote_work_label",
            y="attrition_rate",
            text=series["attrition_rate"].map(lambda x: f"{x:.1f}%"),
            color="remote_work_label",
            color_discrete_map={"No": KayfaBrand.PRIMARY, "Yes": KayfaBrand.TEAL},
        )
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text="Remote work")
        fig.update_yaxes(title_text="Attrition Rate (%)")
        return self.style_figure(fig, title="Attrition by Remote Work")

    def chart_remote_overtime(self, df: pd.DataFrame) -> go.Figure:
        overtime = (
            df.groupby("overtime_label", observed=False)["attrition"]
            .mean()
            .reindex(["No", "Yes"])
            .mul(100)
            .reset_index(name="attrition_rate")
        )
        remote = (
            df.groupby("remote_work_label", observed=False)["attrition"]
            .mean()
            .reindex(["No", "Yes"])
            .mul(100)
            .reset_index(name="attrition_rate")
        )

        fig = make_subplots(rows=1, cols=2, subplot_titles=("Overtime", "Remote Work"))
        fig.add_trace(
            go.Bar(x=overtime["overtime_label"], y=overtime["attrition_rate"], marker_color=KayfaBrand.ACCENT, text=overtime["attrition_rate"].map(lambda x: f"{x:.1f}%"), textposition="outside"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(x=remote["remote_work_label"], y=remote["attrition_rate"], marker_color=KayfaBrand.TEAL, text=remote["attrition_rate"].map(lambda x: f"{x:.1f}%"), textposition="outside"),
            row=1,
            col=2,
        )
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title_text="Attrition Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Attrition Rate (%)", row=1, col=2)
        return self.style_figure(fig, height=460, title="Work Pattern Signals")

    def top_insights(self, df: pd.DataFrame) -> list[tuple[str, str, str]]:
        if df.empty:
            return []

        overall_rate = df["attrition"].mean() * 100

        role_rates = df.groupby("job_role", observed=False)["attrition"].mean().sort_values(ascending=False)
        role_departures = df.groupby("job_role", observed=False)["attrition"].sum().sort_values(ascending=False)
        top_rate_role = role_rates.index[0]
        top_dep_role = role_departures.index[0]
        top_rate_role_rate = role_rates.iloc[0] * 100

        overtime_no = df[df["overtime"] == 0]["attrition"].mean() * 100
        overtime_yes = df[df["overtime"] == 1]["attrition"].mean() * 100
        remote_no = df[df["remote_work"] == 0]["attrition"].mean() * 100
        remote_yes = df[df["remote_work"] == 1]["attrition"].mean() * 100
        remote_share = df["remote_work"].mean() * 100

        yearly = df.groupby("years_at_company", observed=False)["attrition"].mean().mul(100)
        peak_year = int(yearly.idxmax())
        peak_year_rate = float(yearly.max())
        tenured = df.groupby("tenure_group", observed=False)["attrition"].mean().sort_values(ascending=False).mul(100)
        tenured_worst = tenured.index[0]
        tenured_rate = float(tenured.iloc[0])

        promo_0_2 = df[df["number_of_promotions"].isin([0, 1, 2])]["attrition"].mean() * 100
        promo_3_plus = df[df["number_of_promotions"].isin([3, 4])]["attrition"].mean() * 100
        entry_rate = df[df["job_level"] == 0]["attrition"].mean() * 100
        senior_rate = df[df["job_level"] == 2]["attrition"].mean() * 100

        poor_wlb_rate = df[df["work_life_balance"] == 0]["attrition"].mean() * 100
        good_wlb_rate = df[df["work_life_balance"] == 2]["attrition"].mean() * 100
        poor_wlb_count = int((df["work_life_balance"] == 0).sum())
        prevented_if_improved = round((poor_wlb_rate - good_wlb_rate) / 100 * poor_wlb_count)

        profile_mask = (
            (df["age_group"] == "<30 (Young)")
            & (df["marital_status"] == "Single")
            & (df["job_level"] == 0)
            & (df["overtime"] == 1)
        )
        profile_df = df[profile_mask]
        profile_rate = profile_df["attrition"].mean() * 100
        profile_count = len(profile_df)

        return [
            (
                "Start with Education, keep an eye on Technology",
                (
                    f"Overall attrition sits at {overall_rate:.1f}%. Education has the highest attrition rate at {top_rate_role_rate:.1f}%, "
                    f"while {top_dep_role} contributes the most departures because it is the largest role. That makes Education the cleanest first target."
                ),
                "Focus role-specific workload, manager support, and career-path visibility in Education first, then watch Technology for volume.",
            ),
            (
                "Overtime and flexibility are the clearest operating signals",
                (
                    f"Employees who work overtime leave at {overtime_yes:.1f}% versus {overtime_no:.1f}% for those who do not. "
                    f"Remote employees leave at {remote_yes:.1f}% compared with {remote_no:.1f}% on-site, but only {remote_share:.1f}% of staff work remotely."
                ),
                "Treat overtime as a workload issue, and expand flexibility where roles allow it while validating the remote-work pattern as adoption grows.",
            ),
            (
                "The first decade is the danger window",
                (
                    f"Attrition peaks around year {peak_year} at {peak_year_rate:.1f}%, and the {tenured_worst} tenure group still records {tenured_rate:.1f}% attrition. "
                    "The pressure is strongest early and mid-career, not in long-tenure veterans."
                ),
                "Invest in onboarding, 90-day follow-ups, and growth conversations in the 2-10 year window.",
            ),
            (
                "People leave when they feel stuck",
                (
                    f"Employees with 0-2 promotions leave at {promo_0_2:.1f}%, while 3+ promotions drop to {promo_3_plus:.1f}%. "
                    f"Entry-level roles sit at {entry_rate:.1f}% attrition versus {senior_rate:.1f}% for senior roles, which reinforces the same story."
                ),
                "Make internal mobility visible with clearer promotion paths, role rotations, and faster movement out of entry level.",
            ),
            (
                "The highest-risk cohort needs direct intervention",
                (
                    f"A young, single, entry-level employee who works overtime leaves at {profile_rate:.1f}% versus {overall_rate:.1f}% company-wide, "
                    f"and the profile covers {profile_count:,} employees. This is large enough to justify a targeted response."
                ),
                "Use this cohort for workload reduction, manager coaching, and early-career retention support.",
            ),
            (
                "Work-life balance is the fastest next-quarter lever",
                (
                    f"Poor work-life balance is the biggest lift above baseline. It sits at {poor_wlb_rate:.1f}% attrition versus {good_wlb_rate:.1f}% for the good group, "
                    f"so moving that group up a notch could prevent roughly {prevented_if_improved:,} departures."
                ),
                "Focus next quarter on workload redesign, manager accountability, and protected time off.",
            ),
        ]


class AttritionDashboard:
    def __init__(self, analyzer: AttritionAnalyzer):
        self.analyzer = analyzer

    @staticmethod
    def _asset_to_data_uri(path: Path) -> str:
        mime = "image/png" if path.suffix.lower() == ".png" else "image/svg+xml"
        encoded = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime};base64,{encoded}"

    @staticmethod
    def inject_styles() -> None:
        st.markdown(
            f"""
            <style>
                @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

                :root {{
                    --brand: {KayfaBrand.PRIMARY};
                    --brand-dark: {KayfaBrand.PRIMARY_DARK};
                    --brand-light: {KayfaBrand.PRIMARY_LIGHT};
                    --surface: {KayfaBrand.SURFACE};
                    --card: {KayfaBrand.CARD};
                    --text: {KayfaBrand.TEXT};
                    --muted: {KayfaBrand.MUTED};
                    --border: {KayfaBrand.BORDER};
                }}

                html, body, [class*="css"] {{
                    font-family: {KayfaBrand.FONT_STACK};
                    color: var(--text);
                }}

                .stApp {{
                    background:
                        radial-gradient(1100px 520px at 78% -10%, rgba(71,81,213,0.12) 0%, rgba(71,81,213,0) 55%),
                        radial-gradient(900px 600px at 8% 120%, rgba(46,55,168,0.08) 0%, rgba(46,55,168,0) 60%),
                        linear-gradient(180deg, #FFFFFF 0%, var(--surface) 100%);
                }}

                .block-container {{
                    padding-top: 0.85rem;
                    padding-bottom: 2.5rem;
                    max-width: 100%;
                }}

                .kayfa-hero {{
                    background:
                        radial-gradient(900px 420px at 82% 14%, rgba(71,81,213,0.95) 0%, rgba(71,81,213,0) 56%),
                        radial-gradient(620px 320px at 18% 100%, rgba(92,102,230,0.34) 0%, rgba(92,102,230,0) 64%),
                        linear-gradient(150deg, #1A1844 0%, #12122F 48%, #0B0B1F 100%);
                    border: 1px solid rgba(255,255,255,0.08);
                    border-radius: 28px;
                    padding: 1.5rem 1.7rem 1.25rem;
                    position: relative;
                    overflow: hidden;
                    box-shadow: 0 22px 60px rgba(20,19,43,0.22);
                    margin-bottom: 1.2rem;
                    min-height: 420px;
                }}

                .kayfa-hero::before {{
                    content: "";
                    position: absolute;
                    inset: 0;
                    background:
                        linear-gradient(rgba(255,255,255,0.035) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(255,255,255,0.035) 1px, transparent 1px);
                    background-size: 44px 44px;
                    pointer-events: none;
                    opacity: 0.45;
                }}

                .hero-top {{
                    display: flex;
                    align-items: flex-start;
                    justify-content: space-between;
                    gap: 1rem;
                    position: relative;
                    z-index: 1;
                }}

                .hero-eyebrow {{
                    color: rgba(255,255,255,0.45);
                    font-size: 0.92rem;
                    font-weight: 700;
                    letter-spacing: 0.16em;
                    text-transform: uppercase;
                    line-height: 1.45;
                }}

                .hero-badge {{
                    display: inline-flex;
                    align-items: center;
                    justify-content: center;
                    background: {KayfaBrand.ACCENT};
                    color: #FFFFFF;
                    border-radius: 999px;
                    padding: 0.55rem 1rem;
                    font-size: 0.78rem;
                    font-weight: 700;
                    letter-spacing: 0.12em;
                    text-transform: uppercase;
                    margin-top: 1rem;
                    box-shadow: 0 10px 22px rgba(255,107,74,0.22);
                }}

                .hero-logo {{
                    width: clamp(150px, 14vw, 230px);
                    flex: 0 0 auto;
                    margin-top: 0.1rem;
                }}

                .hero-logo img {{
                    width: 100%;
                    height: auto;
                    display: block;
                    filter: brightness(1.2) contrast(1.05);
                }}

                .hero-kicker {{
                    display: flex;
                    align-items: center;
                    gap: 0.9rem;
                    margin-top: 3.1rem;
                    position: relative;
                    z-index: 1;
                }}

                .hero-kicker::before {{
                    content: "";
                    width: 34px;
                    height: 1px;
                    background: rgba(138,145,242,0.72);
                    flex: 0 0 auto;
                }}

                .hero-kicker-text {{
                    color: #7F89F2;
                    font-size: 0.9rem;
                    font-weight: 700;
                    letter-spacing: 0.22em;
                    text-transform: uppercase;
                }}

                .hero-title {{
                    color: #FFFFFF;
                    font-family: {KayfaBrand.DISPLAY_STACK};
                    font-size: clamp(2.45rem, 4.8vw, 4.55rem);
                    line-height: 1.02;
                    margin: 1rem 0 0;
                    position: relative;
                    z-index: 1;
                    max-width: 980px;
                }}

                .hero-subtitle {{
                    color: #8F96FF;
                    font-family: {KayfaBrand.DISPLAY_STACK};
                    font-size: clamp(1.45rem, 2.8vw, 2.65rem);
                    line-height: 1.08;
                    margin: 0.55rem 0 0;
                    position: relative;
                    z-index: 1;
                    max-width: 1040px;
                }}

                .hero-copy {{
                    color: rgba(255,255,255,0.72);
                    font-size: 1.02rem;
                    line-height: 1.7;
                    max-width: 780px;
                    margin: 1.35rem 0 0;
                    position: relative;
                    z-index: 1;
                }}

                .hero-stats {{
                    display: grid;
                    grid-template-columns: repeat(6, minmax(0, 1fr));
                    gap: 1rem;
                    margin-top: 3.05rem;
                    position: relative;
                    z-index: 1;
                }}

                .hero-stat-label {{
                    color: rgba(255,255,255,0.30);
                    font-size: 0.72rem;
                    font-weight: 700;
                    letter-spacing: 0.14em;
                    text-transform: uppercase;
                    margin-bottom: 0.45rem;
                }}

                .hero-stat-value {{
                    color: rgba(255,255,255,0.93);
                    font-size: 0.98rem;
                    line-height: 1.45;
                    font-weight: 600;
                }}

                @media (max-width: 1100px) {{
                    .hero-stats {{
                        grid-template-columns: repeat(3, minmax(0, 1fr));
                    }}
                }}

                @media (max-width: 700px) {{
                    .kayfa-hero {{
                        padding: 1.25rem 1rem 1rem;
                        min-height: auto;
                    }}

                    .hero-top {{
                        flex-direction: column;
                    }}

                    .hero-logo {{
                        width: 180px;
                        margin-top: 0.35rem;
                    }}

                    .hero-kicker {{
                        margin-top: 2rem;
                    }}

                    .hero-stats {{
                        grid-template-columns: repeat(2, minmax(0, 1fr));
                        gap: 0.85rem;
                        margin-top: 2rem;
                    }}
                }}

                .section-title {{
                    font-family: {KayfaBrand.DISPLAY_STACK};
                    font-size: 1.9rem;
                    font-weight: 400;
                    margin: 0.15rem 0 0.2rem;
                    color: var(--text);
                }}

                .section-subtitle {{
                    color: var(--muted);
                    margin-bottom: 0.9rem;
                    font-size: 0.96rem;
                }}

                .question-title {{
                    font-size: 1.2rem;
                    font-weight: 700;
                    color: var(--text);
                    margin: 1.1rem 0 0.1rem;
                }}

                .insight-card {{
                    background: var(--card);
                    border: 1px solid var(--border);
                    border-radius: 18px;
                    padding: 1rem 1rem 0.95rem;
                    box-shadow: 0 10px 28px rgba(20,19,43,0.05);
                    height: 100%;
                    margin-bottom: 1rem;
                }}

                .insight-kicker {{
                    display: inline-block;
                    background: var(--primary-soft, #ECEEFB);
                    color: var(--brand-dark);
                    border-radius: 999px;
                    padding: 0.28rem 0.6rem;
                    font-size: 0.76rem;
                    font-weight: 700;
                    letter-spacing: 0.04em;
                    text-transform: uppercase;
                    margin-bottom: 0.8rem;
                }}

                .insight-card h4 {{
                    margin: 0 0 0.5rem;
                    color: var(--text);
                    font-size: 1.05rem;
                }}

                .insight-card p {{
                    margin: 0;
                    color: var(--muted);
                    font-size: 0.95rem;
                    line-height: 1.65;
                }}

                .page-footer {{
                    margin-top: 1.25rem;
                    background: linear-gradient(135deg, #11112B 0%, #0B0B1F 100%);
                    color: rgba(255,255,255,0.44);
                    padding: 1rem 1.2rem;
                    border-radius: 18px;
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    gap: 1rem;
                    flex-wrap: wrap;
                    font-size: 0.78rem;
                    border: 1px solid rgba(255,255,255,0.08);
                }}

                .page-footer span {{
                    line-height: 1.55;
                }}

                .footer-logo {{
                    height: 24px;
                    width: auto;
                    opacity: 0.9;
                }}

                .soft-panel {{
                    background: rgba(255,255,255,0.72);
                    border: 1px solid var(--border);
                    border-radius: 18px;
                    padding: 0.9rem 1rem;
                    box-shadow: 0 10px 24px rgba(20,19,43,0.04);
                }}

                .stDataFrame, .stTable {{
                    border-radius: 14px;
                }}

                div[data-baseweb="tab-list"] {{
                    gap: 0.25rem;
                    border-bottom: 1px solid rgba(20, 19, 43, 0.12);
                }}

                div[data-baseweb="tab-list"] button[role="tab"] {{
                    font-family: {KayfaBrand.FONT_STACK};
                    font-size: 1rem;
                    font-weight: 700;
                    letter-spacing: 0.01em;
                    padding: 0.9rem 1.25rem 1rem;
                    min-height: 52px;
                    color: {KayfaBrand.TEXT};
                    border-radius: 14px 14px 0 0;
                }}

                div[data-baseweb="tab-list"] button[role="tab"][aria-selected="true"] {{
                    color: {KayfaBrand.PRIMARY};
                    background: rgba(71, 81, 213, 0.08);
                }}

                div[data-baseweb="tab-list"] button[role="tab"]:hover {{
                    background: rgba(71, 81, 213, 0.05);
                    color: {KayfaBrand.PRIMARY_DARK};
                }}
            </style>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_header() -> None:
        logo_uri = AttritionDashboard._asset_to_data_uri(HEADER_LOGO_PATH)
        components.html(
            dedent(
                """
                <html>
                <head>
                  <style>
                    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=DM+Serif+Display&display=swap');

                    html, body {
                      margin: 0;
                      padding: 0;
                      background: transparent;
                      overflow: hidden;
                      font-family: 'DM Sans', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
                    }

                    * { box-sizing: border-box; }

                    .page-header {
                      position: relative;
                      overflow: hidden;
                      background:
                        radial-gradient(1100px 520px at 78% -10%, #3A3FA8 0%, rgba(58,63,168,0) 55%),
                        radial-gradient(900px 600px at 8% 120%, #232a78 0%, rgba(35,42,120,0) 60%),
                        linear-gradient(150deg, #1A1B45 0%, #14132B 60%, #0D0C1F 100%);
                      color: #fff;
                      padding: 44px 64px 36px;
                      border-radius: 28px 28px 32px 32px;
                      min-height: 610px;
                      width: 100%;
                    }

                    .blob {
                      position: absolute;
                      border-radius: 50%;
                      filter: blur(70px);
                      opacity: 0.5;
                      pointer-events: none;
                    }

                    .blob-1 {
                      width: 420px;
                      height: 420px;
                      top: -180px;
                      right: 6%;
                      background: #4751D5;
                      animation: drift1 16s ease-in-out infinite;
                    }

                    .blob-2 {
                      width: 360px;
                      height: 360px;
                      bottom: -200px;
                      left: 18%;
                      background: #5C66E6;
                      opacity: 0.35;
                      animation: drift2 20s ease-in-out infinite;
                    }

                    @keyframes drift1 {
                      0%,100% { transform: translate(0,0); }
                      50% { transform: translate(-30px, 24px); }
                    }

                    @keyframes drift2 {
                      0%,100% { transform: translate(0,0); }
                      50% { transform: translate(26px, -20px); }
                    }

                    .iso-grid {
                      position: absolute;
                      inset: 0;
                      pointer-events: none;
                      opacity: 0.05;
                      background-image:
                        linear-gradient(rgba(255,255,255,.6) 1px, transparent 1px),
                        linear-gradient(90deg, rgba(255,255,255,.6) 1px, transparent 1px);
                      background-size: 44px 44px;
                      mask-image: radial-gradient(70% 90% at 60% 40%, #000 0%, transparent 75%);
                      -webkit-mask-image: radial-gradient(70% 90% at 60% 40%, #000 0%, transparent 75%);
                    }

                    .header-inner {
                      position: relative;
                      z-index: 2;
                      width: 100%;
                      max-width: none;
                    }

                    .brand-bar {
                      display: flex;
                      align-items: center;
                      justify-content: space-between;
                      gap: 1rem;
                      margin-bottom: 40px;
                    }

                    .brand-credit {
                      display: flex;
                      flex-direction: column;
                      align-items: flex-start;
                      gap: 0.55rem;
                    }

                    .task-pill {
                      display: inline-flex;
                      align-items: center;
                      justify-content: center;
                      padding: 0.5rem 0.85rem;
                      border-radius: 999px;
                      background: {KayfaBrand.ACCENT};
                      border: 1px solid {KayfaBrand.ACCENT};
                      color: #FFFFFF;
                      font-size: 0.76rem;
                      font-weight: 800;
                      letter-spacing: 0.14em;
                      text-transform: uppercase;
                      white-space: nowrap;
                    }

                    .brand-tagline {
                      font-size: 11px;
                      font-weight: 500;
                      letter-spacing: 0.16em;
                      text-transform: uppercase;
                      color: {KayfaBrand.ACCENT};
                      text-align: left;
                      line-height: 1.6;
                    }

                    .brand-logo {
                      width: clamp(220px, 18vw, 290px);
                      height: auto;
                      display: block;
                    }

                    .header-title {
                      font-family: 'DM Serif Display', serif;
                      font-weight: 400;
                      line-height: 1.05;
                      font-size: clamp(42px, 4.1vw, 52px);
                      color: #fff;
                      margin: 8px 0 8px;
                    }

                    .header-sub {
                      font-family: 'DM Serif Display', serif;
                      font-size: clamp(22px, 2.35vw, 30px);
                      line-height: 1.1;
                      color: #8A91F2;
                      margin-bottom: 22px;
                    }

                    .header-desc {
                      font-size: 16px;
                      color: rgba(255,255,255,0.66);
                      max-width: 620px;
                      margin-bottom: 38px;
                      line-height: 1.8;
                    }

                    .header-meta {
                      display: grid;
                      grid-template-columns: repeat(6, minmax(0, 1fr));
                      gap: 14px;
                      width: 100%;
                      margin-top: 24px;
                      margin-bottom: 34px;
                    }

                    .meta-item {
                      display: flex;
                      flex-direction: column;
                      gap: 8px;
                      background: rgba(255,255,255,0.04);
                      border: 1px solid rgba(255,255,255,0.08);
                      border-radius: 16px;
                      padding: 16px 16px 15px;
                      min-height: 92px;
                    }

                    .meta-label {
                      font-size: 10px;
                      font-weight: 700;
                      letter-spacing: 0.14em;
                      text-transform: uppercase;
                      color: rgba(255,255,255,0.38);
                    }

                .meta-value {
                      font-size: 18px;
                      font-weight: 700;
                      color: rgba(255,255,255,0.96);
                      line-height: 1.15;
                }

                .segment-mini-grid {
                      display: grid;
                      grid-template-columns: repeat(3, minmax(0, 1fr));
                      gap: 16px;
                      margin-top: 18px;
                }

                .segment-mini-card {
                      position: relative;
                      overflow: hidden;
                      background:
                        linear-gradient(180deg, rgba(255,255,255,0.95) 0%, rgba(247,247,255,0.94) 100%);
                      border: 1px solid rgba(20,19,43,0.10);
                      border-radius: 20px;
                      padding: 18px 18px 16px;
                      box-shadow: 0 14px 30px rgba(20,19,43,0.07);
                      min-height: 112px;
                }

                .segment-mini-card::before {
                      content: "";
                      position: absolute;
                      inset: 0 auto auto 0;
                      width: 100%;
                      height: 5px;
                      background: linear-gradient(90deg, {KayfaBrand.PRIMARY} 0%, {KayfaBrand.PRIMARY_LIGHT} 100%);
                }

                .segment-mini-label {
                      position: relative;
                      z-index: 1;
                      font-size: 11px;
                      font-weight: 700;
                      letter-spacing: 0.08em;
                      text-transform: uppercase;
                      color: #6B6A82;
                      margin-bottom: 10px;
                }

                .segment-mini-value {
                      position: relative;
                      z-index: 1;
                      font-size: 1.55rem;
                      line-height: 1.05;
                      font-weight: 700;
                      color: {KayfaBrand.TEXT};
                      overflow: hidden;
                      text-overflow: ellipsis;
                      white-space: nowrap;
                }

                @media (max-width: 1100px) {
                      .header-meta {
                        grid-template-columns: repeat(3, minmax(0, 1fr));
                      }

                      .segment-mini-grid {
                        grid-template-columns: repeat(2, minmax(0, 1fr));
                      }
                }

                @media (max-width: 700px) {
                      .page-header {
                        padding: 28px 22px 20px;
                        min-height: auto;
                      }

                    .brand-bar {
                        margin-bottom: 28px;
                      }

                      .brand-logo {
                        width: 190px;
                      }

                      .header-meta {
                        grid-template-columns: repeat(2, minmax(0, 1fr));
                        gap: 10px;
                      }

                      .segment-mini-grid {
                        grid-template-columns: 1fr;
                      }
                }

                .segment-mini-card:hover {
                      transform: translateY(-2px);
                      box-shadow: 0 18px 38px rgba(20,19,43,0.10);
                      transition: transform 160ms ease, box-shadow 160ms ease;
                }
                  </style>
                </head>
                <body>
                  <section class="page-header">
                    <div class="blob blob-1"></div>
                    <div class="blob blob-2"></div>
                    <div class="iso-grid"></div>

                      <div class="header-inner">
                      <div class="brand-bar">
                        <div class="brand-tagline">BY Abdalla Omar<br>Week 1 Task</div>
                        <img class="brand-logo" src="__LOGO_URI__" alt="Kayfa logo" />
                      </div>

                      <h1 class="header-title">Employee Attrition Analytics</h1>
                      <div class="header-sub">Understanding why employees leave and what HR can do about it.</div>

                      <div class="header-desc">
                        A deep-dive into 74,498 employee records across five industries.
                        Analyzing the key drivers behind attrition, from compensation and
                        satisfaction to remote work and career growth, to help HR leaders
                        make smarter retention decisions.
                      </div>

                      <div class="header-meta">
                        <div class="meta-item">
                          <div class="meta-label">Employees</div>
                          <div class="meta-value">74,498</div>
                        </div>
                        <div class="meta-item">
                          <div class="meta-label">Attrition Rate</div>
                          <div class="meta-value">47.5%</div>
                        </div>
                        <div class="meta-item">
                          <div class="meta-label">Left</div>
                          <div class="meta-value">35,370</div>
                        </div>
                        <div class="meta-item">
                          <div class="meta-label">Avg Income</div>
                          <div class="meta-value">7,299</div>
                        </div>
                        <div class="meta-item">
                          <div class="meta-label">Avg Tenure</div>
                          <div class="meta-value">15.7</div>
                        </div>
                        <div class="meta-item">
                          <div class="meta-label">Avg Age</div>
                          <div class="meta-value">38.5</div>
                        </div>
                      </div>
                    </div>
                  </section>
                </body>
                </html>
                """
            ).replace("__LOGO_URI__", logo_uri),
            height=710,
            scrolling=False,
        )

    @staticmethod
    def render_metric_row(metrics: dict[str, str]) -> None:
        cols = st.columns(6)
        metric_specs = [
            ("Employees", metrics["employees"]),
            ("Attrition Rate", metrics["attrition_rate"]),
            ("Left", metrics["left"]),
            ("Avg Income", metrics["avg_income"]),
            ("Avg Tenure", metrics["avg_tenure"]),
            ("Avg Age", metrics["avg_age"]),
        ]
        for col, (label, value) in zip(cols, metric_specs):
            with col:
                st.metric(label, value)

    @staticmethod
    def render_footer() -> None:
        logo_uri = AttritionDashboard._asset_to_data_uri(HEADER_LOGO_PATH)
        st.markdown(
            f"""
            <div class="page-footer">
                <span>Kayfa &middot; AI &amp; Data Analytics Internship Program</span>
                <img class="footer-logo" src="{logo_uri}" alt="Kayfa logo" />
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def build_filter_panel(df: pd.DataFrame) -> dict[str, list[str]]:
        st.sidebar.image(str(LOGO_PATH), use_container_width=True)
        st.sidebar.markdown("### Filters")
        st.sidebar.caption("All filters apply to every chart on the page.")

        filters = {
            "job_role": st.sidebar.multiselect("Job role", sorted(df["job_role"].dropna().unique().tolist()), default=sorted(df["job_role"].dropna().unique().tolist())),
            "gender": st.sidebar.multiselect("Gender", sorted(df["gender"].dropna().unique().tolist()), default=sorted(df["gender"].dropna().unique().tolist())),
            "age_group": st.sidebar.multiselect(
                "Age group",
                ["<30 (Young)", "30-44 (Mid-Career)", "45+ (Senior)"],
                default=["<30 (Young)", "30-44 (Mid-Career)", "45+ (Senior)"],
            ),
            "tenure_group": st.sidebar.multiselect(
                "Tenure group",
                ["0-2 years (New)", "2-5 years (Established)", "5-10 years (Loyal)", "10+ years (Veteran)"],
                default=["0-2 years (New)", "2-5 years (Established)", "5-10 years (Loyal)", "10+ years (Veteran)"],
            ),
            "marital_status": st.sidebar.multiselect(
                "Marital status",
                sorted(df["marital_status"].dropna().unique().tolist()),
                default=sorted(df["marital_status"].dropna().unique().tolist()),
            ),
            "job_level_label": st.sidebar.multiselect(
                "Job level",
                ["Entry", "Mid", "Senior"],
                default=["Entry", "Mid", "Senior"],
            ),
        }
        return filters

    @staticmethod
    def render_section_intro(title: str, subtitle: str) -> None:
        st.markdown(f"<div class='section-title'>{title}</div>", unsafe_allow_html=True)
        st.markdown(f"<div class='section-subtitle'>{subtitle}</div>", unsafe_allow_html=True)

    @staticmethod
    def render_question_intro(title: str, subtitle: str) -> None:
        st.markdown(
            f"<div class='question-title'>{title}</div><div class='section-subtitle'>{subtitle}</div>",
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_answer_panel(title: str, answer: str, why_it_matters: str, so_what: str, note: str | None = None) -> None:
        note_html = ""
        if note:
            note_html = f"<p style='margin-top:0.85rem;color:{KayfaBrand.MUTED};font-size:0.92rem;'><strong>Note:</strong> {note}</p>"
        st.markdown(
            f"""
            <div class="insight-card" style="border-left:5px solid {KayfaBrand.PRIMARY};">
                <div class="insight-kicker">Answer</div>
                <h4>{title}</h4>
                <p><strong>What we found:</strong> {answer}</p>
                <p style="margin-top:0.7rem;"><strong>Why it matters:</strong> {why_it_matters}</p>
                <p style="margin-top:0.85rem;color:{KayfaBrand.MUTED};font-size:0.92rem;"><strong>So what?</strong> {so_what}</p>
                {note_html}
            </div>
            """,
            unsafe_allow_html=True,
        )

    @staticmethod
    def render_insight_cards(insights: list[tuple[str, str, str]]) -> None:
        cols = st.columns(2)
        for index, (title, body, solution) in enumerate(insights):
            with cols[index % 2]:
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <div class="insight-kicker">Finding</div>
                        <h4>{title}</h4>
                        <p>{body}</p>
                        <p style="margin-top:0.85rem;color:{KayfaBrand.MUTED};font-size:0.92rem;"><strong>Solution:</strong> {solution.replace('Solution: ', '')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    def run(self) -> None:
        self.inject_styles()
        self.render_header()

        filters = self.build_filter_panel(self.analyzer.display_df)
        filtered_df = self.analyzer.filtered(filters)
        if filtered_df.empty:
            st.warning("No rows match the current filter selection. Please widen at least one filter.")
            return
        full_df = filtered_df

        tabs = st.tabs(["Overview", "Comparison & Segmentations", "Synthesis & Decision-Making", "Business Decisions"])

        with tabs[0]:
            self.render_section_intro(
                "Overview",
                "Key retention headlines from the full cleaned dataset.",
            )
            self.render_question_intro(
                "The headline",
                "What share of employees left overall, and which job role is losing the most people?",
            )
            q1_overall = full_df["attrition"].mean() * 100
            job_role_rates = full_df.groupby("job_role", observed=False)["attrition"].mean().sort_values(ascending=False)
            top_role_name = job_role_rates.index[0]
            top_role_rate = job_role_rates.iloc[0] * 100
            top_role_departures = int(full_df.groupby("job_role", observed=False)["attrition"].sum().sort_values(ascending=False).iloc[0])
            top_role_size = int(full_df["job_role"].value_counts().loc[top_role_name])

            left, right = st.columns([1.15, 0.85])
            with left:
                st.plotly_chart(self.analyzer.chart_attrition_pie(full_df), use_container_width=True, config={"displayModeBar": False})
                st.plotly_chart(self.analyzer.chart_job_role_attrition(full_df), use_container_width=True, config={"displayModeBar": False})
            with right:
                self.render_answer_panel(
                    "The headline",
                    (
                        f"{q1_overall:.1f}% of employees left overall. "
                        f"{top_role_name} has the highest attrition rate at {top_role_rate:.1f}% and contributes {top_role_departures:,} departures out of {top_role_size:,} employees in that role."
                    ),
                    (
                        f"The company has a broad retention problem, but {top_role_name} is the first place leadership should look because it combines the highest rate with meaningful scale."
                    ),
                    (
                        f"Start a focused retention review in {top_role_name}: workload, manager quality, career progression, and compensation should be checked before spreading effort elsewhere."
                    ),
                    note="If leadership cares about absolute headcount lost, this role is also one of the largest contributors to total departures."
                )
                st.markdown(
                    f"""
                    <div class="soft-panel" style="margin-top:1rem;">
                        <div class="meta-label" style="color:{KayfaBrand.MUTED};">Business takeaway</div>
                        <div style="font-size:1.05rem;line-height:1.7;color:{KayfaBrand.TEXT};font-weight:600;">
                            Education is the highest-risk role by attrition rate. Technology has the largest number of departures because it is the biggest group, so leadership should watch both the rate and the volume.
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)
            self.render_question_intro(
                "Overtime",
                "Are employees who work overtime more likely to leave, and by how much versus those who do not?",
            )
            overtime_rates = full_df.groupby("overtime", observed=False)["attrition"].mean().reindex([0, 1]).mul(100)
            overtime_no = overtime_rates.loc[0]
            overtime_yes = overtime_rates.loc[1]
            overtime_lift = overtime_yes - overtime_no
            overtime_ratio = overtime_yes / overtime_no if overtime_no else 0

            left, right = st.columns([1.15, 0.85])
            with left:
                st.plotly_chart(self.analyzer.chart_attrition_by_overtime(full_df), use_container_width=True, config={"displayModeBar": False})
            with right:
                self.render_answer_panel(
                    "Overtime",
                    (
                        f"Employees who work overtime are more likely to leave: {overtime_yes:.1f}% attrition versus {overtime_no:.1f}% for those who do not, a gap of {overtime_lift:.1f} percentage points or about {overtime_ratio:.2f}x higher risk."
                    ),
                    "That size of gap strongly suggests workload pressure or burnout rather than random noise.",
                    "HR should treat overtime as a workload issue: rebalance staffing, reduce chronic overtime, and have managers intervene early when overtime becomes the norm.",
                    note="This is an association, not proof of causality, but the difference is large enough to justify action."
                )

            st.markdown("<div style='height:0.35rem;'></div>", unsafe_allow_html=True)
            self.render_question_intro(
                "Remote work",
                "Does offering remote work appear to keep people, and how large is the effect?",
            )
            remote_rates = full_df.groupby("remote_work", observed=False)["attrition"].mean().reindex([0, 1]).mul(100)
            remote_no = remote_rates.loc[0]
            remote_yes = remote_rates.loc[1]
            remote_gap = remote_no - remote_yes
            remote_share = full_df["remote_work"].mean() * 100

            left, right = st.columns([1.15, 0.85])
            with left:
                st.plotly_chart(self.analyzer.chart_attrition_by_remote_work(full_df), use_container_width=True, config={"displayModeBar": False})
            with right:
                self.render_answer_panel(
                    "Remote work",
                    (
                        f"Remote work appears to help retention: remote employees show {remote_yes:.1f}% attrition versus {remote_no:.1f}% for non-remote staff, a gap of {remote_gap:.1f} percentage points."
                    ),
                    (
                        f"The effect looks large, but only about {remote_share:.1f}% of staff work remotely, so this is based on a relatively small subgroup."
                    ),
                    (
                        "Leadership can say flexible work is associated with lower attrition, but should avoid overclaiming causality. The safe move is to expand flexibility where roles allow it and validate the pattern with more rollout data."
                    ),
                    note="The small remote-work share means this is a promising signal, not a final policy verdict."
                )

        with tabs[1]:
            self.render_section_intro(
                "Comparison & Segmentations",
                "Compare pay, tenure, engagement, and life stage to show where attrition concentrates.",
            )

            self.render_question_intro(
                "Pay fairness",
                "Within the same job level, do lower-paid employees leave more often, and where does the pay effect flatten?",
            )
            left, right = st.columns([1.15, 0.85])
            with left:
                st.plotly_chart(self.analyzer.chart_pay_fairness(full_df), use_container_width=True, config={"displayModeBar": False})
            with right:
                self.render_answer_panel(
                    "Pay fairness",
                    (
                        "Yes. Within every job level, the lowest-paid employees have the highest attrition. "
                        "The retention benefit is strongest at the bottom of the band and then flattens once pay reaches the middle of the range."
                    ),
                    (
                        "That means pay is still part of the problem, but only targeted pay fixes are likely to move the needle. "
                        "Broad raises would be expensive without much extra retention benefit."
                    ),
                    (
                        "Protect the floor pay bands first. Fix the lowest-paid employees inside each job level, keep clear minimum-to-median spreads, and use targeted band corrections instead of across-the-board raises."
                    ),
                    note="The data shows diminishing returns once employees move out of the lowest band."
                )

            st.markdown("---")

            self.render_question_intro(
                "The retention timeline",
                "At what stage of tenure is attrition highest, and where should retention effort be aimed?",
            )
            left, right = st.columns([1.15, 0.85])
            with left:
                st.plotly_chart(self.analyzer.chart_retention_timeline(full_df), use_container_width=True, config={"displayModeBar": False})
            with right:
                peak_year = int(
                    full_df.groupby("years_at_company", observed=False)["attrition"].mean().mul(100).idxmax()
                )
                peak_year_rate = float(
                    full_df.groupby("years_at_company", observed=False)["attrition"].mean().mul(100).max()
                )
                tenure_group_rates = (
                    full_df.groupby("tenure_group", observed=False)["attrition"].mean().sort_values(ascending=False).mul(100)
                )
                worst_tenure_group = tenure_group_rates.index[0]
                worst_tenure_rate = tenure_group_rates.iloc[0]
                self.render_answer_panel(
                    "The retention timeline",
                    (
                        f"Attrition peaks around year {peak_year} at {peak_year_rate:.1f}%, and the highest tenure-group risk is {worst_tenure_group} at {worst_tenure_rate:.1f}%."
                    ),
                    (
                        "The real risk window is early-to-mid tenure, not long-tenure veterans. The first 10 years carry the heaviest churn, with a clear drop after employees become veterans."
                    ),
                    (
                        "Put the most retention effort into onboarding and mid-career growth, especially the 2-10 year window. Long-tenure employees still matter, but they are not the main leak."
                    ),
                    note="The peak at year 9 is the sharpest single point, but the broader message is that the first decade is the danger zone."
                )

            st.markdown("---")

            self.render_question_intro(
                "Engagement warning signs",
                "Which mix of Job Satisfaction and Work-Life Balance is the strongest early-warning sign that someone will leave?",
            )
            left, right = st.columns([1.15, 0.85])
            with left:
                st.plotly_chart(self.analyzer.chart_engagement_heatmap(full_df), use_container_width=True, config={"displayModeBar": False})
            with right:
                engagement = (
                    full_df.groupby(["work_life_balance_label", "job_satisfaction_label"], observed=False)["attrition"]
                    .mean()
                    .mul(100)
                    .reset_index(name="attrition_rate")
                    .sort_values("attrition_rate", ascending=False)
                )
                top_engagement = engagement.iloc[0]
                top_engagement_count = int(
                    full_df[
                        (full_df["work_life_balance_label"] == top_engagement["work_life_balance_label"])
                        & (full_df["job_satisfaction_label"] == top_engagement["job_satisfaction_label"])
                    ].shape[0]
                )
                self.render_answer_panel(
                    "Engagement warning signs",
                    (
                        f"The strongest warning sign is {top_engagement['work_life_balance_label']} work-life balance paired with {top_engagement['job_satisfaction_label']} job satisfaction, where attrition reaches {top_engagement['attrition_rate']:.1f}% across {top_engagement_count:,} employees."
                    ),
                    (
                        "The combo tells managers to watch for both workload strain and disengagement. Poor work-life balance is already dangerous, and low satisfaction makes the risk even worse."
                    ),
                    (
                        "Managers should intervene when workload gets poor or satisfaction starts slipping: rebalance work, reduce overload, and have a direct check-in before the employee becomes a leaver."
                    ),
                    note="The danger signal is the combination, not either metric alone."
                )

            st.markdown("---")

            self.render_question_intro(
                "Life stage",
                "Do age, marital status, and number of dependents change who leaves?",
            )
            st.plotly_chart(self.analyzer.chart_life_stage(full_df), use_container_width=True, config={"displayModeBar": False})
            life_stage = (
                full_df.groupby(["age_group", "marital_status", "number_of_dependents"], observed=False)["attrition"]
                .mean()
                .mul(100)
                .reset_index(name="attrition_rate")
                .sort_values("attrition_rate", ascending=False)
            )
            top_life = life_stage.iloc[0]
            top_life_count = int(
                full_df[
                    (full_df["age_group"] == top_life["age_group"])
                    & (full_df["marital_status"] == top_life["marital_status"])
                    & (full_df["number_of_dependents"] == top_life["number_of_dependents"])
                ].shape[0]
            )
            self.render_answer_panel(
                "Life stage",
                (
                    f"The highest-risk life stage is {top_life['age_group']}, {top_life['marital_status'].lower()}, with {int(top_life['number_of_dependents'])} dependents. That exact profile leaves at {top_life['attrition_rate']:.1f}% and represents {top_life_count:,} employees."
                ),
                (
                    "Age and relationship status clearly matter, but marital status is the strongest signal. Single employees are the main risk group, especially in the younger age bands."
                ),
                (
                    "Retain them with career acceleration, manager support, and flexibility for early-life transitions. Generic perks will not be enough if they feel stuck or unsupported."
                ),
                note="The dependents pattern is weaker than the age-and-marital signal, so don't over-focus on family status alone."
            )

        with tabs[2]:
            self.render_section_intro(
                "Synthesis & Decision-Making",
                "Pull the evidence together and translate it into the most useful next actions.",
            )

            # Career stagnation
            self.render_question_intro(
                "Career stagnation",
                "Does limited growth line up with leaving?",
            )
            left, right = st.columns([1.15, 0.85])
            with left:
                st.plotly_chart(self.analyzer.chart_career_stagnation(full_df), use_container_width=True, config={"displayModeBar": False})
            with right:
                promotions_0_2 = full_df[full_df["number_of_promotions"].isin([0, 1, 2])]["attrition"].mean() * 100
                promotions_3_plus = full_df[full_df["number_of_promotions"].isin([3, 4])]["attrition"].mean() * 100
                entry_rate = full_df[full_df["job_level"] == 0]["attrition"].mean() * 100
                senior_rate = full_df[full_df["job_level"] == 2]["attrition"].mean() * 100
                self.render_answer_panel(
                    "Career stagnation",
                    (
                        f"The pattern is consistent: attrition is {promotions_0_2:.1f}% for employees with 0-2 promotions, then drops to {promotions_3_plus:.1f}% once employees reach 3+ promotions. "
                        f"Entry-level roles also sit at {entry_rate:.1f}% attrition, far above senior roles at {senior_rate:.1f}%."
                    ),
                    (
                        "Employees who do not see movement are much more likely to leave. The signal is strongest in promotions and job level, with smaller but still positive pressure from missing leadership and innovation opportunities."
                    ),
                    (
                        "Build visible internal mobility: clearer promotion criteria, role rotations, stretch assignments, and faster movement out of entry-level tracks."
                    ),
                    note="The biggest gap is between employees who have moved a few times and those who are still stuck near the bottom of the ladder."
                )

            st.markdown("---")

            # Highest-risk profile
            self.render_question_intro(
                "Highest-risk profile",
                "What combination of factors creates the single riskiest employee profile?",
            )
            left, right = st.columns([1.15, 0.85])
            profile_mask = (
                (full_df["age_group"] == "<30 (Young)")
                & (full_df["marital_status"] == "Single")
                & (full_df["job_level"] == 0)
                & (full_df["overtime"] == 1)
            )
            profile_df = full_df[profile_mask]
            profile_rate = profile_df["attrition"].mean() * 100
            profile_count = len(profile_df)
            baseline_rate = full_df["attrition"].mean() * 100
            with left:
                st.plotly_chart(
                    self.analyzer.chart_profile_vs_company(baseline_rate, profile_rate),
                    use_container_width=True,
                    config={"displayModeBar": False},
                )
            with right:
                self.render_answer_panel(
                    "Highest-risk profile",
                    (
                        f"The strongest high-risk profile I found is a young, single, entry-level employee who works overtime. "
                        f"That group leaves at {profile_rate:.1f}% versus {baseline_rate:.1f}% company-wide, and it covers {profile_count:,} employees."
                    ),
                    (
                        "This is not a fringe cluster. It is large enough to matter, and the combination of youth, single status, entry level, and overtime points to early-career pressure."
                    ),
                    (
                        "Treat this as a priority cohort for onboarding, manager coaching, workload control, and early career-path support."
                    ),
                    note="This profile is based on four visible factors, which makes it strong enough to act on but still simple enough to communicate."
                )

            st.markdown("---")

            # What moves the needle
            self.render_question_intro(
                "What moves the needle",
                "If HR could improve only one thing next quarter, where would the data point?",
            )
            left, right = st.columns([1.15, 0.85])
            with left:
                st.plotly_chart(self.analyzer.chart_action_levers(full_df), use_container_width=True, config={"displayModeBar": False})
            with right:
                poor_wlb_rate = full_df[full_df["work_life_balance"] == 0]["attrition"].mean() * 100
                good_wlb_rate = full_df[full_df["work_life_balance"] == 2]["attrition"].mean() * 100
                poor_wlb_count = int((full_df["work_life_balance"] == 0).sum())
                rough_prevented = round((poor_wlb_rate - good_wlb_rate) / 100 * poor_wlb_count)
                self.render_answer_panel(
                    "What moves the needle",
                    (
                        "Among the controllable drivers, poor work-life balance has the largest attrition lift above the company average, ahead of no remote work and overtime. "
                        f"Poor work-life balance sits at {poor_wlb_rate:.1f}% attrition versus {full_df['attrition'].mean() * 100:.1f}% company-wide."
                    ),
                    (
                        "This is the clearest next-quarter lever because it is both material and operationally fixable. Work-life balance also connects directly to the overtime findings from the first tab."
                    ),
                    (
                        f"Focus the next quarter on workload redesign, manager accountability, and protected time off. If the poor work-life balance group moved up to the good-work-life-balance rate, that would prevent roughly {rough_prevented:,} departures across this population."
                    ),
                    note="Other risk signals such as entry-level status are strong, but they are slower to change than day-to-day workload."
                )

        with tabs[3]:
            if filtered_df.empty:
                st.warning("No rows match the current filter selection. Please widen at least one filter.")
            else:
                self.render_section_intro(
                    "Insights",
                    "Business takeaways supported by the strongest patterns in the data.",
                )
                self.render_insight_cards(self.analyzer.top_insights(filtered_df))

        self.render_footer()


def main() -> None:
    st.set_page_config(
        page_title="Kayfa | Employee Attrition Dashboard",
        page_icon=str(ICON_PATH),
        layout="wide",
        initial_sidebar_state="expanded",
    )
    if not DATA_PATH.exists():
        st.error(f"Could not find the cleaned dataset at: {DATA_PATH}")
        return

    raw_df = load_data(str(DATA_PATH))
    analyzer = AttritionAnalyzer(raw_df)
    dashboard = AttritionDashboard(analyzer)
    dashboard.run()


if __name__ == "__main__":
    main()
