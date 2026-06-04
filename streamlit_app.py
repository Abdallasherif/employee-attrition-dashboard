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

    def chart_work_life_balance(self, df: pd.DataFrame) -> go.Figure:
        order = ["Poor", "Fair", "Good", "Excellent"]
        series = (
            df.groupby("work_life_balance_label", observed=False)["attrition"]
            .mean()
            .reindex(order)
            .mul(100)
            .reset_index(name="attrition_rate")
        )
        fig = go.Figure(
            go.Bar(
                x=series["work_life_balance_label"],
                y=series["attrition_rate"],
                marker_color=[KayfaBrand.ACCENT, KayfaBrand.AMBER, KayfaBrand.PRIMARY_LIGHT, KayfaBrand.TEAL],
                text=series["attrition_rate"].map(lambda x: f"{x:.1f}%"),
                textposition="outside",
            )
        )
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title_text="Attrition Rate (%)")
        fig.update_xaxes(title_text="")
        return self.style_figure(fig, title="Attrition by Work-Life Balance")

    def chart_job_satisfaction(self, df: pd.DataFrame) -> go.Figure:
        order = ["Low", "Medium", "High", "Very High"]
        series = (
            df.groupby("job_satisfaction_label", observed=False)["attrition"]
            .mean()
            .reindex(order)
            .mul(100)
            .reset_index(name="attrition_rate")
        )
        fig = go.Figure(
            go.Bar(
                x=series["job_satisfaction_label"],
                y=series["attrition_rate"],
                marker_color=[KayfaBrand.ACCENT, KayfaBrand.AMBER, KayfaBrand.PRIMARY_LIGHT, KayfaBrand.TEAL],
                text=series["attrition_rate"].map(lambda x: f"{x:.1f}%"),
                textposition="outside",
            )
        )
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title_text="Attrition Rate (%)")
        fig.update_xaxes(title_text="")
        return self.style_figure(fig, title="Attrition by Job Satisfaction")

    def chart_promotions(self, df: pd.DataFrame) -> go.Figure:
        series = (
            df.groupby("number_of_promotions", observed=False)["attrition"]
            .mean()
            .mul(100)
            .reset_index(name="attrition_rate")
        )
        fig = px.line(
            series,
            x="number_of_promotions",
            y="attrition_rate",
            markers=True,
        )
        fig.update_traces(line=dict(color=KayfaBrand.PRIMARY, width=3), marker=dict(size=10))
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title_text="Attrition Rate (%)")
        fig.update_xaxes(title_text="Number of Promotions")
        return self.style_figure(fig, title="Attrition vs Promotions")

    def chart_income_box(self, df: pd.DataFrame) -> go.Figure:
        fig = px.box(
            df,
            x="attrition_label",
            y="monthly_income",
            color="attrition_label",
            color_discrete_map={"Stayed": KayfaBrand.PRIMARY, "Left": KayfaBrand.ACCENT},
        )
        fig.update_layout(showlegend=False)
        fig.update_xaxes(title_text="")
        fig.update_yaxes(title_text="Monthly Income")
        return self.style_figure(fig, title="Income Distribution: Stayed vs Left")

    def chart_age_tenure(self, df: pd.DataFrame) -> go.Figure:
        age_order = ["<30 (Young)", "30-44 (Mid-Career)", "45+ (Senior)"]
        tenure_order = ["0-2 years (New)", "2-5 years (Established)", "5-10 years (Loyal)", "10+ years (Veteran)"]

        age_attrition = (
            df.groupby("age_group", observed=False)["attrition"]
            .mean()
            .reindex(age_order)
            .mul(100)
            .reset_index(name="attrition_rate")
        )
        tenure_attrition = (
            df.groupby("tenure_group", observed=False)["attrition"]
            .mean()
            .reindex(tenure_order)
            .mul(100)
            .reset_index(name="attrition_rate")
        )

        fig = make_subplots(
            rows=1,
            cols=2,
            subplot_titles=("Attrition by Age Group", "Attrition by Tenure Group"),
            horizontal_spacing=0.12,
        )
        fig.add_trace(
            go.Bar(
                x=age_attrition["age_group"],
                y=age_attrition["attrition_rate"],
                marker_color=KayfaBrand.PRIMARY,
                text=age_attrition["attrition_rate"].map(lambda x: f"{x:.1f}%"),
                textposition="outside",
                name="Age Group",
            ),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Bar(
                x=tenure_attrition["tenure_group"],
                y=tenure_attrition["attrition_rate"],
                marker_color=KayfaBrand.PRIMARY_DARK,
                text=tenure_attrition["attrition_rate"].map(lambda x: f"{x:.1f}%"),
                textposition="outside",
                name="Tenure Group",
            ),
            row=1,
            col=2,
        )
        fig.update_layout(showlegend=False)
        fig.update_yaxes(title_text="Attrition Rate (%)", row=1, col=1)
        fig.update_yaxes(title_text="Attrition Rate (%)", row=1, col=2)
        fig.update_xaxes(title_text="", row=1, col=1)
        fig.update_xaxes(title_text="", row=1, col=2)
        return self.style_figure(fig, height=480, title="Career Stage Retention Snapshot")

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

        role = (
            df.groupby("job_role", observed=False)["attrition"].mean().sort_values(ascending=False).head(1)
        )
        role_name = role.index[0]
        role_rate = role.iloc[0] * 100

        wlb = (
            df.groupby("work_life_balance_label", observed=False)["attrition"].mean().sort_values(ascending=False).head(1)
        )
        wlb_name = wlb.index[0]
        wlb_rate = wlb.iloc[0] * 100

        sat = (
            df.groupby("job_satisfaction_label", observed=False)["attrition"].mean().sort_values(ascending=False).head(1)
        )
        sat_name = sat.index[0]
        sat_rate = sat.iloc[0] * 100

        age = df.groupby("age_group", observed=False)["attrition"].mean().sort_values(ascending=False).head(1)
        age_name = age.index[0]
        age_rate = age.iloc[0] * 100

        tenure = df.groupby("tenure_group", observed=False)["attrition"].mean().sort_values(ascending=False).head(1)
        tenure_name = tenure.index[0]
        tenure_rate = tenure.iloc[0] * 100

        return [
            (
                f"Retention priority: {role_name}",
                f"{role_name} shows the strongest churn signal at {role_rate:.1f}%, so it should be the first group for a focused retention review.",
                "Solution: review role-specific workload, manager support, and career-path visibility.",
            ),
            (
                f"Workload signal: {wlb_name}",
                f"{wlb_name} work-life balance is linked to {wlb_rate:.1f}% attrition, which points to workload and flexibility as practical levers.",
                "Solution: improve flexibility, rebalance demand, and track workload through pulse checks.",
            ),
            (
                f"Engagement gap: {sat_name}",
                f"Employees in the {sat_name.lower()} satisfaction band are leaving most often at {sat_rate:.1f}%, suggesting a clear engagement issue.",
                "Solution: strengthen feedback loops, recognition, and manager follow-up.",
            ),
            (
                f"Career-stage focus: {age_name}",
                f"The {age_name} segment has the highest age-group attrition at {age_rate:.1f}%, indicating a retention gap in that career stage.",
                "Solution: add mentoring, growth checkpoints, and targeted development plans.",
            ),
            (
                f"Tenure risk: {tenure_name}",
                f"The {tenure_name.lower()} group records {tenure_rate:.1f}% attrition, so onboarding and growth support matter here.",
                "Solution: strengthen onboarding, 90-day check-ins, and early-career coaching.",
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

                .insight-card {{
                    background: var(--card);
                    border: 1px solid var(--border);
                    border-radius: 18px;
                    padding: 1rem 1rem 0.95rem;
                    box-shadow: 0 10px 28px rgba(20,19,43,0.05);
                    height: 100%;
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
                      max-width: 1100px;
                    }

                    .brand-bar {
                      display: flex;
                      align-items: center;
                      margin-bottom: 40px;
                    }

                    .brand-tagline {
                      font-size: 11px;
                      font-weight: 500;
                      letter-spacing: 0.16em;
                      text-transform: uppercase;
                      color: rgba(255,255,255,0.42);
                      text-align: left;
                      line-height: 1.6;
                    }

                    .brand-logo {
                      margin-left: auto;
                      width: 210px;
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
                        width: 150px;
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
                        <div class="brand-tagline">BY Abdalla Omar</div>
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
    def render_insight_cards(insights: list[tuple[str, str, str]]) -> None:
        cols = st.columns(2)
        for index, (title, body, solution) in enumerate(insights):
            with cols[index % 2]:
                st.markdown(
                    f"""
                    <div class="insight-card">
                        <div class="insight-kicker">Insight {index + 1}</div>
                        <h4>{title}</h4>
                        <p>{body}</p>
                        <p style="margin-top:0.85rem;color:{KayfaBrand.MUTED};font-size:0.92rem;"><strong>Solution:</strong> {solution.replace('Solution: ', '')}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                if index % 2 == 1 and index < len(insights) - 1:
                    st.write("")

    def run(self) -> None:
        self.inject_styles()
        self.render_header()

        filters = self.build_filter_panel(self.analyzer.display_df)
        filtered_df = self.analyzer.filtered(filters)

        if filtered_df.empty:
            st.warning("No rows match the current filter selection. Please widen at least one filter.")
            return

        tabs = st.tabs(["Overview", "Drivers", "Segments", "Insights", "Data"])

        with tabs[0]:
            self.render_section_intro(
                "Overview",
                "A quick read on attrition, role distribution, and the main career-stage patterns.",
            )
            left, right = st.columns([1, 1])
            with left:
                st.plotly_chart(self.analyzer.chart_attrition_pie(filtered_df), use_container_width=True, config={"displayModeBar": False})
            with right:
                st.plotly_chart(self.analyzer.chart_job_role_attrition(filtered_df), use_container_width=True, config={"displayModeBar": False})
            st.plotly_chart(self.analyzer.chart_age_tenure(filtered_df), use_container_width=True, config={"displayModeBar": False})

        with tabs[1]:
            self.render_section_intro(
                "Drivers",
                "The key retention levers: work-life balance, satisfaction, promotions, and income.",
            )
            left, right = st.columns(2)
            with left:
                st.plotly_chart(self.analyzer.chart_work_life_balance(filtered_df), use_container_width=True, config={"displayModeBar": False})
                st.plotly_chart(self.analyzer.chart_promotions(filtered_df), use_container_width=True, config={"displayModeBar": False})
            with right:
                st.plotly_chart(self.analyzer.chart_job_satisfaction(filtered_df), use_container_width=True, config={"displayModeBar": False})
                st.plotly_chart(self.analyzer.chart_income_box(filtered_df), use_container_width=True, config={"displayModeBar": False})

        with tabs[2]:
            self.render_section_intro(
                "Segments",
                "Work-pattern signals and a closer look at which employee groups deserve priority attention.",
            )
            st.plotly_chart(self.analyzer.chart_remote_overtime(filtered_df), use_container_width=True, config={"displayModeBar": False})
            top_role = filtered_df.groupby("job_role", observed=False)["attrition"].mean().sort_values(ascending=False).index[0]
            top_age = filtered_df.groupby("age_group", observed=False)["attrition"].mean().sort_values(ascending=False).index[0]
            top_tenure = filtered_df.groupby("tenure_group", observed=False)["attrition"].mean().sort_values(ascending=False).index[0]
            c1, c2, c3 = st.columns(3)
            card_style = """
                background: linear-gradient(180deg, rgba(255,255,255,0.96) 0%, rgba(247,247,255,0.94) 100%);
                border: 1px solid rgba(20,19,43,0.10);
                border-radius: 20px;
                padding: 18px 18px 16px;
                box-shadow: 0 14px 30px rgba(20,19,43,0.07);
                min-height: 112px;
                position: relative;
                overflow: hidden;
            """
            label_style = """
                font-size: 11px;
                font-weight: 700;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: #6B6A82;
                margin-bottom: 10px;
            """
            value_style = f"""
                font-size: 1.55rem;
                line-height: 1.05;
                font-weight: 700;
                color: {KayfaBrand.TEXT};
                overflow: hidden;
                text-overflow: ellipsis;
                white-space: nowrap;
            """
            top_cards = [
                ("Highest attrition role", top_role),
                ("Highest age-group risk", top_age),
                ("Highest tenure risk", top_tenure),
            ]
            for col, (label, value) in zip([c1, c2, c3], top_cards):
                with col:
                    st.markdown(
                        f"""
                        <div style="{card_style}">
                            <div style="position:absolute; inset:0 auto auto 0; width:100%; height:5px; background: linear-gradient(90deg, {KayfaBrand.PRIMARY} 0%, {KayfaBrand.PRIMARY_LIGHT} 100%);"></div>
                            <div style="{label_style}">{label}</div>
                            <div style="{value_style}">{value}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

        with tabs[3]:
            self.render_section_intro(
                "Insights",
                "A concise business Insights with solutions.",
            )
            self.render_insight_cards(self.analyzer.top_insights(filtered_df))

        with tabs[4]:
            self.render_section_intro(
                "Data",
                "Preview the filtered rows and download the cleaned dataset for further analysis.",
            )
            preview_cols = [
                "age",
                "gender",
                "age_group",
                "tenure_group",
                "job_role",
                "job_level_label",
                "monthly_income",
                "work_life_balance_label",
                "job_satisfaction_label",
                "overtime_label",
                "remote_work_label",
                "attrition_label",
            ]
            available_cols = [col for col in preview_cols if col in filtered_df.columns]
            st.dataframe(filtered_df[available_cols].head(100), use_container_width=True, hide_index=True)

            csv_data = filtered_df.to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Download filtered data",
                data=csv_data,
                file_name="kayfa_attrition_filtered.csv",
                mime="text/csv",
                use_container_width=True,
            )


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
