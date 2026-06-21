#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Streamlit dashboard за визуализация на базата:

Tables:
- prompts
- generations
- labels

Показва:
- общ брой prompts/generations/labels
- разпределение по категории
- разпределение по demographic_group
- разпределение по model/system_version
- label distributions
- transparency_score диагностика
- pair_id / bias pair преглед
- joined таблица Prompt → Generation → Label

Run:
    streamlit run scripts/visualize_db.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pandas as pd
import streamlit as st
import plotly.express as px
from sqlalchemy import text
from dotenv import load_dotenv

from bank_ethics.db.base import SessionLocal


load_dotenv()

st.set_page_config(
    page_title="Bank Ethics DB Dashboard",
    page_icon="🏦",
    layout="wide",
)


LABEL_COLUMNS = [
    "unsafe",
    "privacy_violation",
    "privacy_severity",
    "bias",
    "manipulation",
    "transparency_score",
    "honesty_score",
    "policy_alignment",
    "accessibility_score",
    "missing_human_escalation",
]


TRANSPARENCY_MAP = {
    0: "0 = violation",
    1: "1 = minor issue",
    2: "2 = compliant",
}


def read_sql_df(db, sql: str) -> pd.DataFrame:
    conn = db.connection()
    return pd.read_sql_query(text(sql), conn)


@st.cache_data(ttl=10)
def load_data():
    db = SessionLocal()

    try:
        prompts = read_sql_df(
            db,
            """
            SELECT
                id,
                category,
                demographic_group,
                pair_id,
                source,
                text AS prompt_text,
                prompt_hash,
                created_at
            FROM prompts
            ORDER BY created_at DESC
            """
        )

        generations = read_sql_df(
            db,
            """
            SELECT
                id,
                prompt_id,
                model_name,
                system_version,
                temperature,
                answer,
                created_at
            FROM generations
            ORDER BY created_at DESC
            """
        )

        labels = read_sql_df(
            db,
            """
            SELECT
                id,
                gen_id,
                judge_model,
                judge_version,
                unsafe,
                privacy_violation,
                privacy_severity,
                bias,
                manipulation,
                transparency_score,
                honesty_score,
                policy_alignment,
                accessibility_score,
                missing_human_escalation,
                raw_json,
                created_at
            FROM labels
            ORDER BY created_at DESC
            """
        )

        joined = read_sql_df(
            db,
            """
            SELECT
                p.id AS prompt_id,
                p.category,
                p.demographic_group,
                p.pair_id,
                p.source,
                p.text AS prompt_text,
                p.prompt_hash,
                p.created_at AS prompt_created_at,

                g.id AS generation_id,
                g.model_name,
                g.system_version,
                g.temperature,
                g.answer,
                g.created_at AS generation_created_at,

                l.id AS label_id,
                l.judge_model,
                l.judge_version,
                l.unsafe,
                l.privacy_violation,
                l.privacy_severity,
                l.bias,
                l.manipulation,
                l.transparency_score,
                l.honesty_score,
                l.policy_alignment,
                l.accessibility_score,
                l.missing_human_escalation,
                l.raw_json,
                l.created_at AS label_created_at
            FROM prompts p
            LEFT JOIN generations g
                ON g.prompt_id = p.id
            LEFT JOIN labels l
                ON l.gen_id = g.id
            ORDER BY
                p.created_at DESC,
                g.created_at DESC,
                l.created_at DESC
            """
        )

    finally:
        db.close()

    for df in [prompts, generations, labels, joined]:
        for col in df.columns:
            if "created_at" in col:
                df[col] = pd.to_datetime(df[col], errors="coerce")

    if "transparency_score" in joined.columns:
        joined["transparency_label"] = joined["transparency_score"].map(TRANSPARENCY_MAP)
        joined["transparency_violation_bin"] = (joined["transparency_score"] == 0).astype("Int64")
        joined["transparency_issue_bin"] = (joined["transparency_score"].isin([0, 1])).astype("Int64")

    return prompts, generations, labels, joined


def value_counts_df(df: pd.DataFrame, col: str) -> pd.DataFrame:
    if df.empty or col not in df.columns:
        return pd.DataFrame(columns=[col, "count"])

    out = (
        df[col]
        .fillna("NULL")
        .astype(str)
        .value_counts()
        .reset_index()
    )
    out.columns = [col, "count"]
    return out


def plot_bar(df: pd.DataFrame, x: str, y: str, title: str):
    if df.empty:
        st.info("Няма данни за тази графика.")
        return

    fig = px.bar(df, x=x, y=y, title=title, text=y)
    fig.update_layout(xaxis_title=x, yaxis_title=y)
    st.plotly_chart(fig, use_container_width=True)


def apply_sidebar_filters(joined: pd.DataFrame) -> pd.DataFrame:
    df = joined.copy()

    st.sidebar.header("Филтри")

    only_labeled = st.sidebar.checkbox("Само labeled generations", value=False)

    if only_labeled:
        df = df[df["label_id"].notna()]

    def multiselect_filter(label: str, col: str):
        nonlocal df

        if col not in df.columns:
            return

        values = sorted(df[col].dropna().astype(str).unique().tolist())
        selected = st.sidebar.multiselect(label, values)

        if selected:
            df = df[df[col].astype(str).isin(selected)]

    multiselect_filter("Category", "category")
    multiselect_filter("Demographic group", "demographic_group")
    multiselect_filter("Pair ID", "pair_id")
    multiselect_filter("Model name", "model_name")
    multiselect_filter("System version", "system_version")
    multiselect_filter("Judge version", "judge_version")

    search_text = st.sidebar.text_input("Търси в prompt/answer")

    if search_text:
        mask_prompt = df["prompt_text"].fillna("").str.contains(search_text, case=False, regex=False)
        mask_answer = df["answer"].fillna("").str.contains(search_text, case=False, regex=False)
        df = df[mask_prompt | mask_answer]

    return df


def render_overview(prompts, generations, labels, filtered):
    st.subheader("Общ преглед")

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Prompts", len(prompts))
    col2.metric("Generations", len(generations))
    col3.metric("Labels", len(labels))
    col4.metric("Unique pair_id", prompts["pair_id"].dropna().nunique() if "pair_id" in prompts.columns else 0)

    st.divider()

    c1, c2 = st.columns(2)

    with c1:
        cat_counts = value_counts_df(filtered.drop_duplicates("prompt_id"), "category")
        plot_bar(cat_counts, "category", "count", "Prompts by category")

    with c2:
        demo_counts = value_counts_df(filtered.drop_duplicates("prompt_id"), "demographic_group")
        plot_bar(demo_counts, "demographic_group", "count", "Prompts by demographic group")

    c3, c4 = st.columns(2)

    with c3:
        model_counts = value_counts_df(filtered.drop_duplicates("generation_id"), "model_name")
        plot_bar(model_counts, "model_name", "count", "Generations by model")

    with c4:
        sys_counts = value_counts_df(filtered.drop_duplicates("generation_id"), "system_version")
        plot_bar(sys_counts, "system_version", "count", "Generations by system version")


def render_label_distributions(filtered):
    st.subheader("Label distributions")

    labeled = filtered[filtered["label_id"].notna()].copy()

    if labeled.empty:
        st.warning("Няма labels за избраните филтри.")
        return

    st.markdown(
        """
        **Transparency score mapping:**

        - `0 = violation`
        - `1 = minor issue`
        - `2 = compliant`

        Затова бинарният label `transparency_violation_bin` трябва да бъде:
        `1` само когато `transparency_score == 0`.
        """
    )

    c1, c2 = st.columns(2)

    with c1:
        if "transparency_score" in labeled.columns:
            transparency_counts = (
                labeled["transparency_score"]
                .value_counts(dropna=False)
                .sort_index()
                .reset_index()
            )
            transparency_counts.columns = ["transparency_score", "count"]
            transparency_counts["meaning"] = transparency_counts["transparency_score"].map(TRANSPARENCY_MAP)

            st.write("Transparency score counts")
            st.dataframe(transparency_counts, use_container_width=True)

            fig = px.bar(
                transparency_counts,
                x="meaning",
                y="count",
                title="Transparency score distribution",
                text="count",
            )
            st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "transparency_violation_bin" in labeled.columns:
            bin_counts = (
                labeled["transparency_violation_bin"]
                .value_counts(dropna=False)
                .sort_index()
                .reset_index()
            )
            bin_counts.columns = ["transparency_violation_bin", "count"]

            st.write("Derived transparency_violation_bin counts")
            st.dataframe(bin_counts, use_container_width=True)

            fig = px.bar(
                bin_counts,
                x="transparency_violation_bin",
                y="count",
                title="Derived transparency_violation_bin distribution",
                text="count",
            )
            st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("All label distributions")

    existing_label_cols = [c for c in LABEL_COLUMNS if c in labeled.columns]

    if not existing_label_cols:
        st.info("Няма налични label колони за визуализация.")
    else:
        for index in range(0, len(existing_label_cols), 2):
            row_cols = st.columns(2)

            for offset, label_col in enumerate(existing_label_cols[index:index + 2]):
                with row_cols[offset]:
                    label_counts = (
                        labeled[label_col]
                        .value_counts(dropna=False)
                        .sort_index()
                        .reset_index()
                    )
                    label_counts.columns = [label_col, "count"]

                    fig = px.bar(
                        label_counts,
                        x=label_col,
                        y="count",
                        title=f"Distribution of {label_col}",
                        text="count",
                    )
                    fig.update_layout(xaxis_title=label_col, yaxis_title="count")
                    st.plotly_chart(fig, use_container_width=True)

    st.divider()

    st.subheader("Label means by category")

    by_category = (
        labeled
        .groupby("category", dropna=False)[existing_label_cols]
        .mean(numeric_only=True)
        .reset_index()
    )

    st.dataframe(by_category, use_container_width=True)


def render_pair_view(filtered):
    st.subheader("Pair-level / bias преглед")

    if "pair_id" not in filtered.columns:
        st.warning("Няма pair_id колона.")
        return

    pair_df = filtered[filtered["pair_id"].notna()].copy()

    if pair_df.empty:
        st.warning("Няма записи с pair_id.")
        return

    pair_summary = (
        pair_df
        .groupby("pair_id", dropna=False)
        .agg(
            rows=("prompt_id", "count"),
            prompts=("prompt_id", "nunique"),
            generations=("generation_id", "nunique"),
            labels=("label_id", "nunique"),
            categories=("category", lambda x: ", ".join(sorted(set(x.dropna().astype(str))))),
            demographic_groups=("demographic_group", lambda x: ", ".join(sorted(set(x.dropna().astype(str))))),
            max_bias=("bias", "max"),
            avg_bias=("bias", "mean"),
        )
        .reset_index()
        .sort_values(["max_bias", "avg_bias"], ascending=False)
    )

    st.dataframe(pair_summary, use_container_width=True)

    selected_pair = st.selectbox(
        "Избери pair_id",
        pair_summary["pair_id"].astype(str).tolist(),
    )

    selected_rows = pair_df[pair_df["pair_id"].astype(str) == selected_pair].copy()

    st.subheader(f"Rows for pair_id = {selected_pair}")

    show_cols = [
        "category",
        "demographic_group",
        "system_version",
        "prompt_text",
        "answer",
        "bias",
        "transparency_score",
        "unsafe",
        "privacy_violation",
        "manipulation",
        "missing_human_escalation",
    ]

    show_cols = [c for c in show_cols if c in selected_rows.columns]

    st.dataframe(
        selected_rows[show_cols],
        use_container_width=True,
        height=500,
    )


def render_browser(filtered):
    st.subheader("Prompt / Generation browser")

    if filtered.empty:
        st.warning("Няма редове след филтрите.")
        return

    browse_df = filtered.copy()
    browse_df["short_prompt"] = browse_df["prompt_text"].fillna("").str.slice(0, 120)
    browse_df["short_answer"] = browse_df["answer"].fillna("").str.slice(0, 120)

    display_cols = [
        "prompt_id",
        "generation_id",
        "label_id",
        "category",
        "demographic_group",
        "pair_id",
        "model_name",
        "system_version",
        "judge_version",
        "short_prompt",
        "short_answer",
    ]

    display_cols = [c for c in display_cols if c in browse_df.columns]

    st.dataframe(
        browse_df[display_cols],
        use_container_width=True,
        height=450,
    )

    st.divider()

    selected_generation = st.selectbox(
        "Избери generation_id за детайлен преглед",
        browse_df["generation_id"].dropna().astype(str).unique().tolist(),
    )

    if selected_generation:
        row = browse_df[browse_df["generation_id"].astype(str) == selected_generation].iloc[0]

        st.markdown("### Prompt")
        st.write(row.get("prompt_text", ""))

        st.markdown("### Answer")
        st.write(row.get("answer", ""))

        st.markdown("### Metadata")

        meta_cols = [
            "category",
            "demographic_group",
            "pair_id",
            "source",
            "model_name",
            "system_version",
            "temperature",
            "judge_model",
            "judge_version",
        ]

        meta = {
            col: row.get(col)
            for col in meta_cols
            if col in row.index
        }

        st.json(meta)

        st.markdown("### Labels")

        label_meta = {
            col: row.get(col)
            for col in LABEL_COLUMNS
            if col in row.index
        }

        st.json(label_meta)

        if "raw_json" in row.index and pd.notna(row.get("raw_json")):
            with st.expander("Raw judge JSON"):
                st.code(row.get("raw_json"), language="json")


def render_raw_tables(prompts, generations, labels, filtered):
    st.subheader("Raw tables")

    table_name = st.selectbox(
        "Избери таблица",
        ["joined_filtered", "prompts", "generations", "labels"],
    )

    if table_name == "joined_filtered":
        df = filtered
    elif table_name == "prompts":
        df = prompts
    elif table_name == "generations":
        df = generations
    else:
        df = labels

    st.dataframe(df, use_container_width=True, height=600)

    csv_data = df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label=f"Download {table_name}.csv",
        data=csv_data,
        file_name=f"{table_name}.csv",
        mime="text/csv",
    )


def main():
    st.title("🏦 Bank Ethics Database Dashboard")

    prompts, generations, labels, joined = load_data()

    if st.sidebar.button("Refresh data"):
        st.cache_data.clear()
        st.rerun()

    filtered = apply_sidebar_filters(joined)

    tab1, tab2, tab3, tab4, tab5 = st.tabs(
        [
            "Overview",
            "Labels",
            "Pairs / Bias",
            "Browser",
            "Raw / Export",
        ]
    )

    with tab1:
        render_overview(prompts, generations, labels, filtered)

    with tab2:
        render_label_distributions(filtered)

    with tab3:
        render_pair_view(filtered)

    with tab4:
        render_browser(filtered)

    with tab5:
        render_raw_tables(prompts, generations, labels, filtered)


if __name__ == "__main__":
    main()