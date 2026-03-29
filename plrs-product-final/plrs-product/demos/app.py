"""
demos/app.py
============
PLRS — Production demo application.

Run:
    streamlit run demos/app.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import streamlit as st

# Allow running from repo root or demos/ directory
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from plrs.curriculum.loader import load_dag
from plrs.pipeline import PLRSPipeline

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PLRS · Logic Engine",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS — instrument panel aesthetic ───────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@300;400;500&family=Syne:wght@400;600;700;800&display=swap');

/* ── Global reset ── */
html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
    background-color: #0a0e1a;
    color: #c8d0e0;
}

/* ── Hide Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding: 1.5rem 2rem 2rem 2rem; max-width: 1400px; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1221;
    border-right: 1px solid #1e2a40;
}
[data-testid="stSidebar"] .stMarkdown p {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    color: #4a5568;
    letter-spacing: 0.08em;
}

/* ── Header band ── */
.plrs-header {
    display: flex;
    align-items: baseline;
    gap: 1rem;
    padding-bottom: 1rem;
    border-bottom: 1px solid #1e2a40;
    margin-bottom: 1.5rem;
}
.plrs-title {
    font-size: 1.75rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #e8edf5;
}
.plrs-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #3d8bcd;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 2px 8px;
    border: 1px solid #1e3a5f;
    border-radius: 2px;
}

/* ── Stat cards ── */
.stat-row {
    display: flex;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
}
.stat-card {
    flex: 1;
    background: #0d1221;
    border: 1px solid #1e2a40;
    border-radius: 4px;
    padding: 0.9rem 1rem;
    position: relative;
    overflow: hidden;
}
.stat-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: var(--accent, #3d8bcd);
}
.stat-card.green::before  { --accent: #22c55e; }
.stat-card.amber::before  { --accent: #f59e0b; }
.stat-card.red::before    { --accent: #ef4444; }
.stat-card.blue::before   { --accent: #3d8bcd; }
.stat-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #4a5568;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-bottom: 0.25rem;
}
.stat-value {
    font-size: 1.6rem;
    font-weight: 700;
    color: #e8edf5;
    line-height: 1;
}
.stat-sub {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #4a5568;
    margin-top: 0.2rem;
}

/* ── Recommendation cards ── */
.rec-card {
    background: #0d1221;
    border: 1px solid #1e2a40;
    border-radius: 4px;
    padding: 0.9rem 1rem;
    margin-bottom: 0.5rem;
    position: relative;
    transition: border-color 0.15s;
}
.rec-card:hover { border-color: #2a4a6e; }
.rec-card.approved  { border-left: 3px solid #22c55e; }
.rec-card.challenging { border-left: 3px solid #f59e0b; }
.rec-card.vetoed    { border-left: 3px solid #ef4444; opacity: 0.6; }

.rec-title {
    font-size: 0.95rem;
    font-weight: 700;
    color: #e8edf5;
    margin-bottom: 0.15rem;
}
.rec-meta {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #4a5568;
    letter-spacing: 0.06em;
}
.rec-reason {
    font-size: 0.75rem;
    color: #8899aa;
    margin-top: 0.35rem;
    padding-top: 0.35rem;
    border-top: 1px solid #1e2a40;
}
.score-bar-wrap {
    background: #131a2e;
    border-radius: 2px;
    height: 3px;
    margin-top: 0.5rem;
    overflow: hidden;
}
.score-bar {
    height: 100%;
    border-radius: 2px;
    background: var(--bar-color, #3d8bcd);
    transition: width 0.4s ease;
}

/* ── Section headers ── */
.section-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.14em;
    text-transform: uppercase;
    color: #4a5568;
    border-bottom: 1px solid #1e2a40;
    padding-bottom: 0.4rem;
    margin-bottom: 0.75rem;
    margin-top: 1.25rem;
}

/* ── What-if panel ── */
.unlock-chip {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    background: #131a2e;
    border: 1px solid #1e3a5f;
    border-radius: 2px;
    padding: 2px 7px;
    margin: 2px 3px 2px 0;
    color: #3d8bcd;
}
.blocked-chip {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    background: #1a1010;
    border: 1px solid #3f1e1e;
    border-radius: 2px;
    padding: 2px 7px;
    margin: 2px 3px 2px 0;
    color: #ef4444;
}

/* ── Mastery bar in slider area ── */
.mastery-mini {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #4a5568;
}

/* ── Tab styling ── */
.stTabs [data-baseweb="tab-list"] {
    gap: 0;
    border-bottom: 1px solid #1e2a40;
    background: transparent;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    letter-spacing: 0.08em;
    color: #4a5568;
    padding: 0.5rem 1.25rem;
    border-bottom: 2px solid transparent;
}
.stTabs [aria-selected="true"] {
    color: #3d8bcd;
    border-bottom-color: #3d8bcd;
    background: transparent;
}

/* ── Sliders ── */
.stSlider [data-testid="stTickBarMin"],
.stSlider [data-testid="stTickBarMax"] { display: none; }
</style>
""", unsafe_allow_html=True)


# ── Data loading ──────────────────────────────────────────────────────────────

@st.cache_resource
def load_pipelines():
    maps = ROOT / "data" / "knowledge_maps"
    pipelines = {}
    for domain, fname in [("math", "math_dag.json"), ("cs", "cs_dag.json")]:
        path = maps / fname
        if path.exists():
            pipelines[domain] = PLRSPipeline(load_dag(path))
    return pipelines

pipelines = load_pipelines()


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 🧠 PLRS")
    st.markdown('<p style="font-family:\'DM Mono\',monospace;font-size:0.65rem;color:#4a5568;letter-spacing:0.1em;">LOGIC ENGINE v0.1.0</p>', unsafe_allow_html=True)
    st.markdown("---")

    domain_label = st.selectbox("Curriculum", ["Nigerian SS Mathematics", "CS Fundamentals"], label_visibility="visible")
    domain_key   = "math" if "Mathematics" in domain_label else "cs"
    pipeline     = pipelines[domain_key]
    curriculum   = pipeline.curriculum

    st.markdown("---")
    threshold      = st.slider("Mastery threshold", 0.50, 0.90, 0.70, 0.05,
                                help="Above this = topic mastered")
    soft_threshold = st.slider("Challenging threshold", 0.20, 0.65, 0.50, 0.05,
                                help="Above this but below mastery = challenging")
    top_n          = st.slider("Top N recommendations", 3, 10, 5)

    pipeline.threshold      = threshold
    pipeline.soft_threshold = soft_threshold
    pipeline.top_n          = top_n
    pipeline.constraint_layer.curriculum = curriculum
    pipeline.ranker = pipeline.ranker.__class__(curriculum)

    st.markdown("---")
    st.markdown('<p>NODES</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:1.2rem;font-weight:700;color:#e8edf5;">{curriculum.num_nodes}</p>', unsafe_allow_html=True)
    st.markdown('<p>EDGES</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:1.2rem;font-weight:700;color:#e8edf5;">{curriculum.num_edges}</p>', unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="plrs-header">
    <span class="plrs-title">Logic Engine</span>
    <span class="plrs-sub">Personalized Learning · Constraint-Aware</span>
</div>
""", unsafe_allow_html=True)


# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["RECOMMENDATIONS", "WHAT-IF SIMULATOR", "CURRICULUM MAP"])


# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — RECOMMENDATIONS
# ══════════════════════════════════════════════════════════════════════════════

with tab1:
    col_left, col_right = st.columns([1, 1.4], gap="large")

    with col_left:
        st.markdown('<div class="section-label">Learner Profile</div>', unsafe_allow_html=True)

        mode = st.radio("Input mode", ["Manual sliders", "Simulate student"], horizontal=True, label_visibility="collapsed")

        mastery_scores = {}

        if mode == "Manual sliders":
            nodes = curriculum.nodes
            for node in nodes:
                label = curriculum.label(node)
                level = curriculum.level(node)
                val = st.slider(
                    f"{label}",
                    0.0, 1.0, 0.0, 0.05,
                    key=f"mastery_{node}",
                    help=f"Level: {level} | Prerequisites: {', '.join(curriculum.prerequisites(node)) or 'none'}"
                )
                mastery_scores[node] = val

        else:
            seq_len = st.slider("Sequence length", 10, 200, 50, key="seq_len")
            seed    = st.number_input("Student seed", 1, 9999, 42, key="seed")
            np.random.seed(int(seed))
            n_skills = 5736
            sim_skills   = np.random.randint(0, n_skills, seq_len).tolist()
            sim_corrects = np.random.randint(0, 2, seq_len).tolist()

            # Map simulated skills to curriculum topics via activity type distribution
            ACTIVITY_TO_DOMAIN = {
                "math": {
                    "oucontent": "algebraic_expressions", "forumng": "statistics_basic",
                    "homepage": "whole_numbers", "subpage": "plane_shapes",
                    "resource": "indices", "url": "number_bases",
                    "ouwiki": "proportion_variation", "glossary": "algebraic_factorization",
                    "quiz": "quadratic_equations",
                },
                "cs": {
                    "oucontent": "programming_concepts", "forumng": "ethics_technology",
                    "homepage": "computer_basics", "subpage": "html_basics",
                    "resource": "networking_fundamentals", "url": "internet_basics",
                    "ouwiki": "cloud_basics", "glossary": "intro_databases",
                    "quiz": "python_basics",
                },
            }
            activity_types = list(ACTIVITY_TO_DOMAIN[domain_key].keys())
            activity_probs = [0.38, 0.20, 0.15, 0.10, 0.06, 0.04, 0.03, 0.02, 0.02]
            mapping = ACTIVITY_TO_DOMAIN[domain_key]

            topic_scores: dict[str, float] = {}
            for skill_id, correct in zip(sim_skills, sim_corrects):
                act_idx = skill_id % 100
                cumulative = 0
                thresholds = [int(p * 100) for p in activity_probs]
                thresholds[-1] += 100 - sum(thresholds)
                act = activity_types[-1]
                for a, thresh in zip(activity_types, thresholds):
                    cumulative += thresh
                    if act_idx < cumulative:
                        act = a
                        break
                topic_id = mapping.get(act)
                if topic_id and topic_id in curriculum.nodes:
                    prob = float(correct) * 0.6 + 0.1 + np.random.random() * 0.3
                    topic_scores[topic_id] = max(topic_scores.get(topic_id, 0.0), min(prob, 1.0))

            mastery_scores = {n: 0.0 for n in curriculum.nodes}
            mastery_scores.update(topic_scores)

            st.success(f"Simulated {seq_len} interactions → {len(topic_scores)} topics mapped")
            if topic_scores:
                st.markdown('<div class="section-label">Mapped Mastery Signal</div>', unsafe_allow_html=True)
                for tid, score in sorted(topic_scores.items(), key=lambda x: -x[1]):
                    label = curriculum.label(tid)
                    pct = int(score * 100)
                    color = "#22c55e" if score >= threshold else "#f59e0b" if score >= soft_threshold else "#ef4444"
                    st.markdown(f"""
                    <div style="margin-bottom:6px;">
                        <div style="display:flex;justify-content:space-between;font-size:0.72rem;color:#8899aa;margin-bottom:2px;">
                            <span>{label}</span>
                            <span style="font-family:'DM Mono',monospace;">{pct}%</span>
                        </div>
                        <div class="score-bar-wrap">
                            <div class="score-bar" style="width:{pct}%;--bar-color:{color};"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

        run = st.button("⚡ Generate Recommendations", type="primary", use_container_width=True)

    with col_right:
        if run or mode == "Simulate student":
            results = pipeline.recommend_from_mastery(mastery_scores)
            summary = results["mastery_summary"]
            stats   = results["stats"]

            # ── Stat cards ────────────────────────────────────────────
            mastery_pct = int(summary["mastery_rate"] * 100)
            vrate_pct   = int(stats["prerequisite_violation_rate"] * 100)

            st.markdown(f"""
            <div class="stat-row">
                <div class="stat-card blue">
                    <div class="stat-label">Mastered</div>
                    <div class="stat-value">{summary['mastered']}<span style="font-size:0.9rem;color:#4a5568;">/{summary['total_topics']}</span></div>
                    <div class="stat-sub">{mastery_pct}% mastery rate</div>
                </div>
                <div class="stat-card green">
                    <div class="stat-label">Approved</div>
                    <div class="stat-value">{stats['approved_count']}</div>
                    <div class="stat-sub">ready to learn</div>
                </div>
                <div class="stat-card amber">
                    <div class="stat-label">Challenging</div>
                    <div class="stat-value">{stats['challenging_count']}</div>
                    <div class="stat-sub">partial prereqs</div>
                </div>
                <div class="stat-card red">
                    <div class="stat-label">Violation rate</div>
                    <div class="stat-value">{vrate_pct}<span style="font-size:0.9rem;color:#4a5568;">%</span></div>
                    <div class="stat-sub">blocked topics</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            # ── Approved ──────────────────────────────────────────────
            if results["approved"]:
                st.markdown('<div class="section-label">✅ Approved Recommendations</div>', unsafe_allow_html=True)
                for i, rec in enumerate(results["approved"]):
                    score_pct = int(rec["score"] * 100)
                    mastery_pct_rec = int(rec["mastery"] * 100)
                    downstream = rec["downstream_count"]
                    st.markdown(f"""
                    <div class="rec-card approved">
                        <div class="rec-title">{i+1}. {rec['topic_label']}</div>
                        <div class="rec-meta">
                            score: {rec['score']:.3f} &nbsp;·&nbsp;
                            mastery: {mastery_pct_rec}% &nbsp;·&nbsp;
                            unlocks: {downstream} topic{'s' if downstream != 1 else ''}
                        </div>
                        <div class="rec-reason">{rec['reasoning']}</div>
                        <div class="score-bar-wrap">
                            <div class="score-bar" style="width:{score_pct}%;--bar-color:#22c55e;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            else:
                st.markdown('<div class="section-label">✅ Approved Recommendations</div>', unsafe_allow_html=True)
                st.markdown('<div style="color:#4a5568;font-size:0.8rem;font-family:\'DM Mono\',monospace;">No approved topics — lower threshold or adjust mastery.</div>', unsafe_allow_html=True)

            # ── Challenging ───────────────────────────────────────────
            if results["challenging"]:
                st.markdown('<div class="section-label">⚠️ Challenging</div>', unsafe_allow_html=True)
                for rec in results["challenging"]:
                    score_pct = int(rec["score"] * 100)
                    unmet = ", ".join(rec["unmet_prerequisites"]) or "—"
                    st.markdown(f"""
                    <div class="rec-card challenging">
                        <div class="rec-title">{rec['topic_label']}</div>
                        <div class="rec-meta">score: {rec['score']:.3f} &nbsp;·&nbsp; strengthen: {unmet}</div>
                        <div class="rec-reason">{rec['reasoning']}</div>
                        <div class="score-bar-wrap">
                            <div class="score-bar" style="width:{score_pct}%;--bar-color:#f59e0b;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            # ── Vetoed sample ─────────────────────────────────────────
            if results["vetoed"]:
                with st.expander(f"❌ Vetoed topics ({stats['vetoed_count']} total)"):
                    for rec in results["vetoed"]:
                        unmet = ", ".join(rec["unmet_prerequisites"]) or "—"
                        st.markdown(f"""
                        <div class="rec-card vetoed">
                            <div class="rec-title">{rec['topic_label']}</div>
                            <div class="rec-meta">blocked by: {unmet}</div>
                        </div>
                        """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="height:300px;display:flex;align-items:center;justify-content:center;
                        border:1px dashed #1e2a40;border-radius:4px;color:#2a3a50;">
                <div style="text-align:center;">
                    <div style="font-size:2rem;margin-bottom:0.5rem;">⚡</div>
                    <div style="font-family:'DM Mono',monospace;font-size:0.7rem;letter-spacing:0.1em;">
                        SET MASTERY LEVELS<br>THEN GENERATE
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — WHAT-IF SIMULATOR
# ══════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-label">Prerequisite Impact Simulator</div>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.8rem;color:#8899aa;">Select a topic to see what it unlocks and what blocks it.</p>', unsafe_allow_html=True)

    node_options = {curriculum.label(n): n for n in curriculum.nodes}
    selected_label = st.selectbox("Select topic", list(node_options.keys()), key="whatif_topic")
    selected_id    = node_options[selected_label]

    col_a, col_b = st.columns(2, gap="large")

    wi = pipeline.what_if(selected_id)

    with col_a:
        st.markdown('<div class="section-label">🔓 What This Unlocks</div>', unsafe_allow_html=True)

        if wi["direct_unlocks"]:
            st.markdown("**Directly unlocks:**")
            chips = "".join(f'<span class="unlock-chip">{u["label"]}</span>' for u in wi["direct_unlocks"])
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#4a5568;font-size:0.8rem;">No direct successors — this is a leaf node.</span>', unsafe_allow_html=True)

        if wi["all_unlocks"]:
            st.markdown(f"**All downstream topics ({wi['total_unlocked']}):**")
            chips = "".join(f'<span class="unlock-chip">{u["label"]}</span>' for u in wi["all_unlocks"])
            st.markdown(chips, unsafe_allow_html=True)

        st.markdown(f"""
        <div class="stat-card blue" style="margin-top:1rem;max-width:200px;">
            <div class="stat-label">Total Unlocked</div>
            <div class="stat-value">{wi['total_unlocked']}</div>
            <div class="stat-sub">downstream topics</div>
        </div>
        """, unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-label">🔒 What Blocks This</div>', unsafe_allow_html=True)

        if wi["blocked_by"]:
            st.markdown("**Prerequisites required:**")
            chips = "".join(f'<span class="blocked-chip">{b["label"]}</span>' for b in wi["blocked_by"])
            st.markdown(chips, unsafe_allow_html=True)
        else:
            st.markdown('<span style="color:#22c55e;font-size:0.8rem;font-family:\'DM Mono\',monospace;">No prerequisites — this is a root topic.</span>', unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — CURRICULUM MAP
# ══════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown('<div class="section-label">Curriculum Knowledge Graph</div>', unsafe_allow_html=True)

    col_info, col_table = st.columns([1, 2], gap="large")

    with col_info:
        st.markdown(f"""
        <div class="stat-card blue" style="margin-bottom:0.75rem;">
            <div class="stat-label">Domain</div>
            <div style="font-size:0.9rem;font-weight:700;color:#e8edf5;">{curriculum.domain}</div>
        </div>
        <div class="stat-card green" style="margin-bottom:0.75rem;">
            <div class="stat-label">Topics</div>
            <div class="stat-value">{curriculum.num_nodes}</div>
        </div>
        <div class="stat-card amber">
            <div class="stat-label">Prerequisite Edges</div>
            <div class="stat-value">{curriculum.num_edges}</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown('<div class="section-label" style="margin-top:1rem;">Root Topics</div>', unsafe_allow_html=True)
        roots = [n for n in curriculum.nodes if not curriculum.prerequisites(n)]
        for r in roots:
            st.markdown(f'<span class="unlock-chip">{curriculum.label(r)}</span>', unsafe_allow_html=True)

        st.markdown('<div class="section-label">Leaf Topics</div>', unsafe_allow_html=True)
        leaves = [n for n in curriculum.nodes if not curriculum.successors(n)]
        for l in leaves:
            st.markdown(f'<span class="blocked-chip">{curriculum.label(l)}</span>', unsafe_allow_html=True)

    with col_table:
        st.markdown('<div class="section-label">All Topics</div>', unsafe_allow_html=True)

        import pandas as pd
        rows = []
        for node in curriculum.nodes:
            prereqs = curriculum.prerequisites(node)
            succs   = curriculum.successors(node)
            desc    = curriculum.descendants(node)
            rows.append({
                "Topic": curriculum.label(node),
                "Level": curriculum.level(node),
                "Prerequisites": len(prereqs),
                "Unlocks (direct)": len(succs),
                "Total Downstream": len(desc),
            })
        df = pd.DataFrame(rows).sort_values("Total Downstream", ascending=False)
        st.dataframe(
            df,
            use_container_width=True,
            height=500,
            hide_index=True,
        )
