import json
import requests
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

API_BASE = st.sidebar.text_input("API base URL", value="http://localhost:8000")

st.title("Causal LLM Evaluator")
st.caption("Effect sizes with 95% CI — not just averages")

# ── Run experiment ────────────────────────────────────────────────────────────
st.header("Run experiment")

demo_path = st.text_input(
    "Experiment JSON path",
    value="demo/creator_experiment.json",
)

col1, col2 = st.columns([3, 1])
with col2:
    run_btn = st.button("Run", type="primary", use_container_width=True)

if run_btn:
    try:
        with open(demo_path) as f:
            payload = json.load(f)
        with st.spinner("Running randomised trial…"):
            resp = requests.post(f"{API_BASE}/experiment", json=payload, timeout=300)
            resp.raise_for_status()
            result = resp.json()
        st.success(f"Experiment `{result['experiment_id']}` complete")
        st.session_state["last_result"] = result
    except FileNotFoundError:
        st.error(f"File not found: {demo_path}")
    except requests.HTTPError as e:
        st.error(f"API error: {e.response.text}")

# ── Fetch past result ─────────────────────────────────────────────────────────
st.header("Load past result")
exp_id_input = st.text_input("Experiment ID")
if st.button("Fetch"):
    resp = requests.get(f"{API_BASE}/results/{exp_id_input}", timeout=10)
    if resp.ok:
        st.session_state["last_result"] = resp.json()
    else:
        st.error(resp.json().get("detail", "Not found"))

# ── Display result ────────────────────────────────────────────────────────────
result = st.session_state.get("last_result")
if result:
    st.divider()
    st.subheader(f"Experiment `{result['experiment_id']}`")
    st.info(result["interpretation"])

    effects = result["effects"]
    if not effects:
        st.warning("No treatment variants to display.")
        st.stop()

    df = pd.DataFrame(effects)
    df = df.sort_values("ate", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, max(3, len(df) * 1.2)))

    for i, row in df.iterrows():
        color = "#2ecc71" if (row["significant"] and row["ate"] > 0) else \
                "#e74c3c" if (row["significant"] and row["ate"] < 0) else "#95a5a6"
        ax.barh(i, row["ate"], color=color, height=0.4, zorder=3)
        ax.plot(
            [row["ci_lower"], row["ci_upper"]], [i, i],
            color="black", linewidth=1.5, zorder=4,
        )
        ax.plot([row["ci_lower"], row["ci_upper"]], [i, i], "|", color="black", markersize=8)
        label = f"  p={row['p_value']:.3f}  d={row['cohens_d']:.2f}"
        ax.text(max(row["ci_upper"], row["ate"]) + 0.005, i, label, va="center", fontsize=8)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df["variant_id"])
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=2)
    ax.set_xlabel("Average Treatment Effect (ATE) vs control")
    ax.set_title("Forest plot — prompt variant effects")
    ax.grid(axis="x", alpha=0.3, zorder=1)

    legend_patches = [
        mpatches.Patch(color="#2ecc71", label="Significant positive"),
        mpatches.Patch(color="#e74c3c", label="Significant negative"),
        mpatches.Patch(color="#95a5a6", label="Not significant"),
    ]
    ax.legend(handles=legend_patches, loc="lower right", fontsize=8)
    plt.tight_layout()
    st.pyplot(fig)

    st.subheader("Raw effect estimates")
    st.dataframe(df, use_container_width=True)
