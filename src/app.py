"""
Predicate Device Intelligence Platform
Streamlit Dashboard — Portfolio Demo
Author: Saurabh | AI Product Manager Portfolio
"""

import json, os, sys
import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from pipeline import (
    extract_510k_intelligence, compare_predicates,
    match_new_device, DEMO_EXTRACTION
)

# ─── PAGE CONFIG ──────────────────────────────────────────────
st.set_page_config(
    page_title="Predicate Device Intelligence",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

NAVY  = "#1E3A5F"
BLUE  = "#2563EB"
TEAL  = "#0E7490"
AMBER = "#B45309"
GREEN = "#15803D"
RED   = "#DC2626"
GRAY  = "#4A5568"

st.markdown("""
<style>
  [data-testid="stSidebar"] { background: #0f1117; }
  [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
  .kpi-card {
    background: white; border-radius: 6px; padding: 1.1rem 1.4rem;
    border-left: 4px solid #2563EB;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07); margin-bottom: 0.5rem;
  }
  .kpi-card.green { border-left-color: #15803D; }
  .kpi-card.amber { border-left-color: #B45309; }
  .kpi-card.teal  { border-left-color: #0E7490; }
  .kpi-card.red   { border-left-color: #DC2626; }
  .kpi-num   { font-size: 1.9rem; font-weight: 700; color: #1E3A5F; line-height: 1.1; }
  .kpi-label { font-size: 0.75rem; color: #4A5568; text-transform: uppercase; letter-spacing: 0.06em; margin-top: 0.2rem; }
  .chip {
    display: inline-block; padding: 0.2rem 0.65rem;
    border-radius: 10px; font-size: 0.72rem; font-weight: 600;
    margin: 0.12rem; white-space: nowrap;
  }
  .chip-red    { background: #FEE2E2; color: #991B1B; }
  .chip-amber  { background: #FEF3C7; color: #92400E; }
  .chip-green  { background: #DCFCE7; color: #166534; }
  .chip-blue   { background: #DBEAFE; color: #1E40AF; }
  .chip-teal   { background: #CFFAFE; color: #155E75; }
  .chip-navy   { background: #1E3A5F; color: white; }
  .chip-gray   { background: #F1F5F9; color: #374151; }
  .sec-head {
    font-size: 1rem; font-weight: 700; color: #1E3A5F;
    border-bottom: 2px solid #DBEAFE; padding-bottom: 0.35rem;
    margin: 1.1rem 0 0.7rem 0;
  }
  .insight-box {
    background: #F8FAFC; border: 1px solid #E2E8F0; border-radius: 5px;
    padding: 0.7rem 0.9rem; margin: 0.3rem 0; font-size: 0.87rem;
  }
  .insight-box.critical { border-left: 3px solid #DC2626; }
  .insight-box.important { border-left: 3px solid #B45309; }
  .insight-box.supporting { border-left: 3px solid #15803D; }
  .guardrail {
    background: #FFF7ED; border: 1px solid #FED7AA; border-radius: 5px;
    padding: 0.7rem 0.9rem; margin: 0.3rem 0; font-size: 0.87rem;
  }
  .risk-row {
    background: white; border: 1px solid #E2E8F0; border-radius: 4px;
    padding: 0.6rem 0.9rem; margin: 0.3rem 0; font-size: 0.87rem;
  }
  .score-bar-bg {
    background: #E2E8F0; border-radius: 4px; height: 8px; margin-top: 4px;
  }
  .accelerator {
    background: #F0FDF4; border: 1px solid #BBF7D0; border-radius: 5px;
    padding: 0.6rem 0.9rem; margin: 0.3rem 0; font-size: 0.87rem;
  }
</style>
""", unsafe_allow_html=True)


# ─── LOAD DATA ────────────────────────────────────────────────
@st.cache_data
def load_db():
    p = os.path.join(os.path.dirname(__file__), "..", "data", "samples", "predicate_510k_database.json")
    with open(p) as f:
        data = json.load(f)
    return {d["k_number"]: d for d in data}

db = load_db()

# ─── SIDEBAR ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 Predicate Intel")
    st.markdown("*510(k) regulatory intelligence for EP devices*")
    st.divider()
    page = st.radio("Navigate", [
        "🔍 510(k) Deep Dive",
        "⚖️ Predicate Comparison",
        "🎯 New Device Matcher",
        "📋 Regulatory Strategy",
        "ℹ️ About"
    ])
    st.divider()
    st.markdown("**Demo Mode** ✅")
    st.markdown("5 real EP 510(k)s pre-loaded. Add Anthropic API key for live LLM analysis.")
    api_key = st.text_input("Anthropic API Key (optional)", type="password", placeholder="sk-ant-...")
    st.divider()
    st.markdown("**Portfolio Project**")
    st.markdown("Built by Saurabh · AI PM")
    st.markdown("[GitHub](#) · [LinkedIn](#)")


def chip(text, style="chip-blue"):
    return f'<span class="chip {style}">{text}</span>'

def sig_chip(sig):
    cls = {"Critical": "chip-red", "Important": "chip-amber", "Supporting": "chip-green"}.get(sig, "chip-gray")
    return chip(sig, cls)

def priority_chip(p):
    cls = {"High": "chip-red", "Medium": "chip-amber", "Low": "chip-green"}.get(p, "chip-gray")
    return chip(p + " Priority", cls)


# ═══════════════════════════════════════════════════════════════
# PAGE 1 — 510(k) DEEP DIVE
# ═══════════════════════════════════════════════════════════════
if page == "🔍 510(k) Deep Dive":
    st.title("🔍 510(k) Deep Dive")
    st.markdown("Select a predicate device to extract its regulatory intelligence, SE strategy, and performance benchmarks.")

    col_sel, col_main = st.columns([1, 2.4])

    with col_sel:
        opts = {f"{v['k_number']} — {v['device_name']}": k for k, v in db.items()}
        sel_label = st.selectbox("Select 510(k)", list(opts.keys()))
        sel_k = opts[sel_label]
        device = db[sel_k]

        use_demo = not bool(api_key)
        if st.button("🚀 Analyze 510(k)", use_container_width=True, type="primary"):
            with st.spinner("Extracting regulatory intelligence..."):
                st.session_state["deep_dive"] = extract_510k_intelligence(sel_k, api_key, use_demo)
                st.session_state["deep_dive_k"] = sel_k
        st.divider()

        st.markdown("**Device Summary**")
        st.markdown(f"**K-Number:** `{device['k_number']}`")
        st.markdown(f"**Applicant:** {device['applicant']}")
        st.markdown(f"**Decision:** {device['decision']}")
        st.markdown(f"**Decision Date:** {device['decision_date']}")
        st.markdown(f"**Device Type:** {device['device_type']}")
        st.markdown(f"**Energy:** {device['energy_modality']}")
        st.markdown(f"**Indication:** {device['indication']}")

        st.divider()
        with st.expander("📄 Indications for Use"):
            st.markdown(f"*{device['indications_for_use']}*")
        with st.expander("🔗 Predicates Used"):
            for p in device["predicate_devices"]:
                st.markdown(f"- `{p}`")

    with col_main:
        # Auto-load on first render
        if "deep_dive" not in st.session_state or st.session_state.get("deep_dive_k") != sel_k:
            st.session_state["deep_dive"] = extract_510k_intelligence(sel_k, None, True)
            st.session_state["deep_dive_k"] = sel_k

        ins = st.session_state["deep_dive"]

        # SE Strategy card
        se = ins.get("substantial_equivalence_strategy", {})
        se_color = {"single": GREEN, "split": AMBER, "multiple": BLUE}.get(se.get("predicate_approach",""), GRAY)
        st.markdown(f"""
        <div style="background:{se_color}15; border:1px solid {se_color}40; border-radius:6px; padding:0.9rem 1.2rem; margin-bottom:1rem;">
          <span style="font-weight:700; color:{se_color}; font-size:1.05rem;">
            SE Strategy: {se.get('predicate_approach','').upper()} PREDICATE
          </span>
          {'&nbsp;&nbsp;<span class="chip chip-red">Clinical Data Required</span>' if se.get("clinical_data_required") else '&nbsp;&nbsp;<span class="chip chip-green">Bench-Only Path</span>'}
          <br><span style="color:#374151; font-size:0.88rem; margin-top:0.4rem; display:block;">{se.get('key_argument','')}</span>
        </div>
        """, unsafe_allow_html=True)

        if se.get("clinical_data_rationale"):
            st.caption(f"**Clinical data rationale:** {se['clinical_data_rationale']}")

        tab1, tab2, tab3, tab4 = st.tabs(["📊 Benchmarks", "🧪 Testing", "⚠️ Risks & Accelerators", "💡 Design Implications"])

        with tab1:
            st.markdown('<div class="sec-head">Performance Benchmarks Established by this 510(k)</div>', unsafe_allow_html=True)
            benchmarks = ins.get("performance_benchmarks", [])
            if benchmarks:
                # Chart
                bdf = pd.DataFrame(benchmarks)
                fig = go.Figure()
                colors = {"Critical": RED, "Important": AMBER, "Supporting": GREEN}
                for sig in ["Critical", "Important", "Supporting"]:
                    subset = bdf[bdf["significance"] == sig]
                    if not subset.empty:
                        fig.add_trace(go.Bar(
                            x=subset["parameter"], y=[1]*len(subset),
                            name=sig, marker_color=colors[sig], opacity=0.8,
                            text=subset["value"], textposition="inside",
                            hovertext=subset.apply(lambda r: f"{r['parameter']}<br>{r['value']}<br>{r['standard']}", axis=1)
                        ))
                fig.update_layout(
                    barmode="stack", height=220, showlegend=True,
                    plot_bgcolor="white", paper_bgcolor="white",
                    title="Benchmarks by Significance", title_font_color=NAVY,
                    xaxis=dict(showticklabels=False),
                    yaxis=dict(showticklabels=False),
                    margin=dict(l=0, r=0, t=40, b=0),
                    legend=dict(orientation="h", y=-0.05)
                )
                st.plotly_chart(fig, use_container_width=True)

                for b in benchmarks:
                    box_cls = b["significance"].lower()
                    st.markdown(
                        f'<div class="insight-box {box_cls}">'
                        f'{sig_chip(b["significance"])} {chip(b["standard"], "chip-teal")} '
                        f'<strong>{b["parameter"]}</strong>: {b["value"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        with tab2:
            st.markdown('<div class="sec-head">Testing Strategy</div>', unsafe_allow_html=True)
            tests = ins.get("testing_strategy", [])

            # Group by category
            cat_map = {}
            for t in tests:
                cat_map.setdefault(t["category"], []).append(t)

            cat_colors = {
                "Electrical Safety": BLUE, "Mechanical": NAVY, "Biocompatibility": GREEN,
                "Sterilization": TEAL, "Software": AMBER, "Bench Performance": GRAY,
                "Clinical": RED
            }
            for cat, items in cat_map.items():
                color = cat_colors.get(cat, GRAY)
                st.markdown(f'<span style="font-weight:700; color:{color}; font-size:0.9rem;">{cat}</span>', unsafe_allow_html=True)
                for t in items:
                    req_badge = chip("Required for Clearance", "chip-red") if t.get("required_for_clearance") else chip("Supporting", "chip-gray")
                    st.markdown(
                        f'<div class="insight-box">'
                        f'{req_badge} {chip(t["standard"], "chip-teal")}'
                        f'<br><strong>{t["test_name"]}</strong>: {t["result_summary"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        with tab3:
            col_r, col_a = st.columns(2)
            with col_r:
                st.markdown('<div class="sec-head">⚠️ Regulatory Risk Flags</div>', unsafe_allow_html=True)
                for flag in ins.get("regulatory_risk_flags", []):
                    st.markdown(
                        f'<div class="risk-row">'
                        f'<strong style="color:{RED}">Issue:</strong> {flag["issue"]}<br>'
                        f'<strong style="color:{AMBER}">Impact:</strong> {flag["impact"]}<br>'
                        f'<strong style="color:{GREEN}">Mitigation used:</strong> {flag["mitigation_used"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )
            with col_a:
                st.markdown('<div class="sec-head">🚀 Clearance Accelerators</div>', unsafe_allow_html=True)
                for acc in ins.get("clearance_accelerators", []):
                    st.markdown(
                        f'<div class="accelerator">'
                        f'<strong style="color:{GREEN}">{acc["factor"]}</strong><br>'
                        f'{acc["description"]}'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        with tab4:
            st.markdown('<div class="sec-head">💡 Design & Regulatory Implications</div>', unsafe_allow_html=True)
            area_colors = {"Design": BLUE, "Testing": TEAL, "Regulatory Strategy": NAVY, "Clinical": RED, "Software": AMBER}
            for impl in ins.get("design_implications", []):
                color = area_colors.get(impl["area"], GRAY)
                st.markdown(
                    f'<div class="insight-box">'
                    f'<span style="font-weight:700; color:{color}; font-size:0.8rem;">{impl["area"].upper()}</span><br>'
                    f'<strong>Insight:</strong> {impl["insight"]}<br>'
                    f'<strong style="color:{BLUE}">Action:</strong> {impl["action"]}'
                    f'</div>',
                    unsafe_allow_html=True
                )

            # Competitive intel
            ci = ins.get("competitive_intelligence", {})
            if ci:
                st.markdown('<div class="sec-head">⚡ Competitive Intelligence</div>', unsafe_allow_html=True)
                st.markdown(f'**Market Position:** {ci.get("market_position","")}')
                st.markdown(f'**Performance Bar Set:** {ci.get("performance_bar_set","")}')
                st.markdown(f'**Strategic Notes:** {ci.get("strategic_notes","")}')


# ═══════════════════════════════════════════════════════════════
# PAGE 2 — PREDICATE COMPARISON
# ═══════════════════════════════════════════════════════════════
elif page == "⚖️ Predicate Comparison":
    st.title("⚖️ Predicate Comparison")
    st.markdown("Compare multiple predicates side-by-side to identify the strongest SE strategy.")

    k_numbers = list(db.keys())
    selected = st.multiselect("Select 510(k)s to compare (2–4 recommended)", k_numbers,
                               default=["K221847", "K213291", "K192516"],
                               format_func=lambda k: f"{k} — {db[k]['device_name']}")

    if len(selected) < 2:
        st.warning("Select at least 2 predicate devices to compare.")
    else:
        use_demo = not bool(api_key)
        result = compare_predicates(selected, api_key, use_demo)

        # ── PREDICATE RECOMMENDATION ──────────────────────
        rec = result.get("predicate_selection_recommendation", {})
        st.markdown(f"""
        <div style="background:#DBEAFE; border:1px solid #93C5FD; border-radius:6px; padding:1rem 1.3rem; margin-bottom:1.2rem;">
          <span style="font-weight:700; color:{NAVY}; font-size:1.1rem;">
            Recommended Strategy: {rec.get('strategy','').upper()} PREDICATE
          </span><br>
          <span style="font-size:0.88rem; color:#374151;">{rec.get('rationale','')}</span><br><br>
          <strong>Intended Use Predicate:</strong> {chip(rec.get('intended_use_predicate',''), 'chip-navy')}
          &nbsp;<strong>Tech Char Predicate(s):</strong>
          {''.join([chip(k, 'chip-teal') for k in (rec.get('tech_char_predicate') or [])])}
        </div>
        """, unsafe_allow_html=True)

        tab1, tab2, tab3 = st.tabs(["📊 Performance Targets", "🧪 Test Plan", "⏱️ Timeline & Risks"])

        with tab1:
            st.markdown('<div class="sec-head">Consolidated Performance Targets for New Device</div>', unsafe_allow_html=True)
            targets = result.get("performance_targets", [])
            if targets:
                tdf = pd.DataFrame(targets)
                risk_colors = {"High": "#FEE2E2", "Medium": "#FEF3C7", "Low": "#DCFCE7"}
                for _, row in tdf.iterrows():
                    bg = risk_colors.get(row["risk_if_missed"].split()[0], "#F8FAFC")
                    st.markdown(
                        f'<div class="insight-box" style="background:{bg};">'
                        f'{chip(row["risk_if_missed"].split()[0] + " Risk", "chip-" + {"High":"red","Medium":"amber","Low":"green"}.get(row["risk_if_missed"].split()[0],"gray"))}'
                        f' {chip(row["source_k_number"], "chip-teal")}'
                        f'<br><strong>{row["parameter"]}</strong>'
                        f'<br>Minimum: <code>{row["minimum_threshold"]}</code> &nbsp; Target: <code>{row["recommended_target"]}</code>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        with tab2:
            st.markdown('<div class="sec-head">Critical Test Sequence</div>', unsafe_allow_html=True)
            tests = result.get("critical_test_list", [])
            if tests:
                # Gantt-style chart
                fig = go.Figure()
                y_labels = [f"P{t['priority']} — {t['test']}" for t in tests]
                starts = [0] * len(tests)
                durations = [t["estimated_weeks"] for t in tests]
                colors_list = [BLUE if i < 3 else (TEAL if i < 6 else GRAY) for i in range(len(tests))]

                for i, t in enumerate(tests):
                    fig.add_trace(go.Bar(
                        x=[t["estimated_weeks"]],
                        y=[y_labels[i]],
                        orientation="h",
                        marker_color=colors_list[i],
                        text=f"{t['estimated_weeks']}w",
                        textposition="inside",
                        hovertext=f"{t['test']}<br>{t['standard']}<br>{t['rationale']}",
                        showlegend=False
                    ))

                fig.update_layout(
                    title="Estimated Testing Timeline (weeks)",
                    xaxis_title="Weeks", height=360,
                    plot_bgcolor="white", paper_bgcolor="white",
                    title_font_color=NAVY, barmode="overlay",
                    margin=dict(l=0, r=10, t=40, b=0),
                    yaxis=dict(autorange="reversed")
                )
                st.plotly_chart(fig, use_container_width=True)

                total_weeks = max(t["estimated_weeks"] for t in tests)
                st.info(f"Critical path: **{total_weeks} weeks** (longest single test). Parallel testing recommended to compress timeline.")

                for t in tests:
                    p_label = f"P{t['priority']}"
                    st.markdown(
                        f'<div class="insight-box">'
                        f'{chip(p_label, "chip-navy")} {chip(t["standard"], "chip-teal")}'
                        f'<br><strong>{t["test"]}</strong> ({t["estimated_weeks"]}w)'
                        f'<br><span style="color:{GRAY}; font-size:0.82rem;">{t["rationale"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

        with tab3:
            col_tl, col_risk = st.columns(2)
            with col_tl:
                tl = result.get("timeline_estimate", {})
                st.markdown('<div class="sec-head">⏱️ FDA Review Timeline Estimate</div>', unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(f'<div class="kpi-card green"><div class="kpi-num">{tl.get("optimistic_days","—")}</div><div class="kpi-label">Optimistic Days</div></div>', unsafe_allow_html=True)
                with c2:
                    st.markdown(f'<div class="kpi-card amber"><div class="kpi-num">{tl.get("realistic_days","—")}</div><div class="kpi-label">Realistic Days</div></div>', unsafe_allow_html=True)
                st.markdown("**Factors that extend review:**")
                for f in tl.get("risk_factors_that_extend", []):
                    st.markdown(f"- {f}")

            with col_risk:
                st.markdown('<div class="sec-head">⚠️ Regulatory Risks</div>', unsafe_allow_html=True)
                for risk in result.get("regulatory_risks", []):
                    lh_cls = {"High": "chip-red", "Medium": "chip-amber", "Low-Medium": "chip-amber", "Low": "chip-green"}.get(risk["likelihood"], "chip-gray")
                    st.markdown(
                        f'<div class="risk-row">'
                        f'{chip(risk["likelihood"], lh_cls)}'
                        f' <strong>{risk["risk"]}</strong><br>'
                        f'<span style="color:{GRAY}; font-size:0.82rem;">Mitigation: {risk["mitigation_strategy"]}</span>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

            # Q-Sub topics
            st.markdown('<div class="sec-head">📬 Recommended Pre-Submission (Q-Sub) Topics</div>', unsafe_allow_html=True)
            for q in result.get("q_submission_topics", []):
                st.markdown(
                    f'<div class="insight-box important">'
                    f'<strong>Q:</strong> {q["question"]}<br>'
                    f'<span style="color:{AMBER};">Why important: {q["why_important"]}</span><br>'
                    f'<span style="color:{GRAY}; font-size:0.82rem;">Precedent: {q["precedent"]}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )


# ═══════════════════════════════════════════════════════════════
# PAGE 3 — NEW DEVICE MATCHER
# ═══════════════════════════════════════════════════════════════
elif page == "🎯 New Device Matcher":
    st.title("🎯 New Device Matcher")
    st.markdown("Describe your device in development — get instant predicate match scores and fastest clearance path.")

    col_in, col_out = st.columns([1, 1.8])

    with col_in:
        st.markdown("**Describe Your Device**")
        device_type = st.selectbox("Device Type", ["Ablation Catheter", "Diagnostic Mapping Catheter", "Ablation System", "Mapping System", "Other"])
        energy = st.selectbox("Energy Modality", ["RF (Radiofrequency)", "Pulsed Field (PFA)", "Cryoablation", "Laser", "Diagnostic Only"])
        indication = st.multiselect("Target Indication(s)", ["Paroxysmal AF", "Persistent AF", "Atrial Flutter", "SVT", "VT", "General EP Mapping"], default=["Paroxysmal AF"])
        features = st.multiselect("Key Features", [
            "Contact force sensing", "Irrigated tip", "Multi-electrode array",
            "Omnipolar technology", "Bidirectional steering", "Integrated mapping",
            "Remote monitoring", "AI-assisted annotation"
        ], default=["Contact force sensing", "Irrigated tip"])
        custom = st.text_area("Additional differentiating features (optional)", placeholder="e.g. 8-electrode circular array, MRI-conditional design, novel electrode coating...")

        if st.button("🎯 Find Best Predicates", use_container_width=True, type="primary"):
            desc = f"Device type: {device_type}. Energy: {energy}. Indications: {', '.join(indication)}. Features: {', '.join(features)}. {custom}"
            use_demo = not bool(api_key)
            with st.spinner("Matching against predicate database..."):
                st.session_state["match"] = match_new_device(desc, api_key, use_demo)

    with col_out:
        if "match" not in st.session_state:
            st.session_state["match"] = match_new_device("", None, True)

        m = st.session_state["match"]

        # Fastest path banner
        fp = m.get("fastest_path", "")
        if fp:
            st.markdown(f"""
            <div style="background:#DCFCE7; border:1px solid #86EFAC; border-radius:6px; padding:0.9rem 1.2rem; margin-bottom:1rem;">
              <span style="font-weight:700; color:{GREEN}; font-size:1rem;">⚡ Fastest Path to Clearance</span><br>
              <span style="font-size:0.88rem; color:#374151;">{fp}</span>
            </div>
            """, unsafe_allow_html=True)

        # Match scores
        st.markdown('<div class="sec-head">Predicate Match Scores</div>', unsafe_allow_html=True)
        scores = m.get("predicate_match_scores", [])

        if scores:
            # Radar / bar comparison
            score_df = pd.DataFrame(scores)
            fig = go.Figure()
            role_colors = {"Primary": GREEN, "Split-IntendedUse": BLUE, "Split-TechChar": TEAL, "Supporting": AMBER, "Not Recommended": GRAY}
            for _, row in score_df.iterrows():
                color = role_colors.get(row["recommended_role"], GRAY)
                fig.add_trace(go.Bar(
                    name=row["k_number"],
                    x=["Intended Use Match", "Tech Char Match", "Overall Fit"],
                    y=[row["intended_use_match"], row["tech_char_match"], row["overall_fit"]],
                    marker_color=color,
                    text=[f'{row["intended_use_match"]}/10', f'{row["tech_char_match"]}/10', f'{row["overall_fit"]}/10'],
                    textposition="outside"
                ))
            fig.update_layout(
                barmode="group", height=300,
                plot_bgcolor="white", paper_bgcolor="white",
                title="Predicate Match Scores (0–10)", title_font_color=NAVY,
                yaxis=dict(range=[0, 12], title="Score /10"),
                margin=dict(l=0, r=0, t=40, b=0),
                legend=dict(orientation="h", y=-0.15)
            )
            st.plotly_chart(fig, use_container_width=True)

            for s in sorted(scores, key=lambda x: x["overall_fit"], reverse=True):
                role = s["recommended_role"]
                role_cls = {"Primary": "chip-green", "Split-IntendedUse": "chip-blue", "Split-TechChar": "chip-teal", "Supporting": "chip-amber", "Not Recommended": "chip-gray"}.get(role, "chip-gray")
                score_pct = int(s["overall_fit"] * 10)
                color = role_colors.get(role, GRAY)
                st.markdown(
                    f'<div class="insight-box">'
                    f'{chip(role, role_cls)} <strong>{s["k_number"]}</strong> — {s["device_name"]}'
                    f'<div class="score-bar-bg"><div style="background:{color}; width:{score_pct}%; height:8px; border-radius:4px;"></div></div>'
                    f'<span style="font-size:0.78rem; color:{GRAY};">Overall fit: {s["overall_fit"]}/10</span><br>'
                    f'<span style="color:{GREEN}; font-size:0.8rem;">✓ {" · ".join(s["strengths"][:2])}</span><br>'
                    f'<span style="color:{RED}; font-size:0.8rem;">⚠ {" · ".join(s["gaps"][:1])}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

        # Key gaps
        st.markdown('<div class="sec-head">⚠️ Key Gaps to Address Before Submission</div>', unsafe_allow_html=True)
        for gap in m.get("key_gaps", []):
            st.markdown(f'<div class="guardrail">⚠️ {gap}</div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 4 — REGULATORY STRATEGY
# ═══════════════════════════════════════════════════════════════
elif page == "📋 Regulatory Strategy":
    st.title("📋 Regulatory Strategy Builder")
    st.markdown("Full design guardrails and regulatory roadmap synthesized from the predicate database.")

    result = compare_predicates(list(db.keys()), None, True)

    # Guardrails
    st.markdown("### 🔒 Design Guardrails — What R&D Must Design To")
    st.markdown("*Based on analysis of all 5 predicate 510(k)s. These are non-negotiable unless you can justify deviation with data.*")

    flex_colors = {"None": RED, "Low": AMBER, "Medium": GREEN}
    for g in result.get("design_guardrails", []):
        flex = g.get("flexibility", "Medium").split()[0]
        color = flex_colors.get(flex, GRAY)
        st.markdown(
            f'<div class="guardrail" style="border-left: 3px solid {color};">'
            f'<span style="font-weight:700; color:{NAVY};">{g["constraint"]}</span>'
            f'&nbsp;&nbsp;{chip("Flexibility: " + g["flexibility"], "chip-" + {"None":"red","Low":"amber","Medium":"green"}.get(flex,"gray"))}'
            f'<br><span style="color:{GRAY}; font-size:0.85rem;">{g["reason"]}</span>'
            f'</div>',
            unsafe_allow_html=True
        )

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        # Q-Sub topics
        st.markdown("### 📬 Pre-Submission Questions for FDA")
        st.markdown("*Raise these in a Q-Sub meeting 6–9 months before planned 510(k) submission.*")
        for i, q in enumerate(result.get("q_submission_topics", []), 1):
            st.markdown(
                f'<div class="insight-box important">'
                f'<strong>Q{i}:</strong> {q["question"]}<br>'
                f'<span style="color:{AMBER}; font-size:0.82rem;">Why: {q["why_important"]}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

    with col2:
        # Timeline visualization
        st.markdown("### ⏱️ 510(k) Review Timeline Model")
        tl = result.get("timeline_estimate", {})
        fig = go.Figure()
        phases = ["Device Design\n& V&V Testing", "510(k) Preparation\n& Submission", "FDA Review\n(Optimistic)", "FDA Review\n(Realistic)", "Clearance\nGranted"]
        durations = [52, 12, tl.get("optimistic_days", 90)//7, tl.get("realistic_days", 130)//7, 2]
        colors_list = [NAVY, TEAL, GREEN, AMBER, BLUE]
        fig.add_trace(go.Bar(
            x=phases, y=durations,
            marker_color=colors_list,
            text=[f"{d}w" for d in durations],
            textposition="outside"
        ))
        fig.update_layout(
            height=300, plot_bgcolor="white", paper_bgcolor="white",
            title="Weeks per Phase (representative)", title_font_color=NAVY,
            yaxis_title="Weeks", margin=dict(l=0, r=0, t=40, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("### 📊 Key Stats from Predicate Analysis")
        k1, k2 = st.columns(2)
        with k1:
            st.markdown(f'<div class="kpi-card"><div class="kpi-num">3/5</div><div class="kpi-label">510(k)s Used Split Predicate</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-card green"><div class="kpi-num">2/5</div><div class="kpi-label">Bench-Only Clearance</div></div>', unsafe_allow_html=True)
        with k2:
            st.markdown(f'<div class="kpi-card amber"><div class="kpi-num">141d</div><div class="kpi-label">Fastest Review (K183021)</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="kpi-card red"><div class="kpi-num">2× AI</div><div class="kpi-label">Requests — Novel Energy (K213291)</div></div>', unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════
# PAGE 5 — ABOUT
# ═══════════════════════════════════════════════════════════════
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
## Predicate Device Intelligence Platform

**Product:** AI-powered regulatory intelligence tool that analyzes FDA 510(k) summaries for EP predicate devices — extracting SE strategies, performance benchmarks, testing approaches, and design implications to help R&D teams get to clearance faster.

**Problem Statement:**
EP device R&D teams spend weeks manually reading 510(k) summaries to understand predicate strategies, performance benchmarks, and testing requirements. This analysis happens late in development — after design decisions are already locked — resulting in costly redesigns, failed bench tests, and FDA additional information requests that extend review by months.

**The insight:** If R&D teams had predicate intelligence at the design phase, they could design to clearance from day one — not just design to spec and discover regulatory gaps later.

**Solution Architecture:**
1. **510(k) Database** — structured repository of EP predicate summaries (integrates with FDA DERA API in production)
2. **LLM Extraction Layer** — Claude analyzes each 510(k) and extracts SE strategy, benchmarks, testing, risks, and design implications
3. **Comparison Engine** — aggregates intelligence across multiple predicates to produce consolidated targets and test plans
4. **Matcher** — maps new device description to best predicate candidates with fit scores
5. **Strategy Builder** — synthesizes database into design guardrails and FDA pre-submission topics

**AI PM Competencies Demonstrated:**
- **Evals:** Extraction validated against known 510(k) outcomes — SE strategy correctly identified for all 5 predicates
- **Prompting:** Three distinct system prompts for extraction, comparison, and device matching — each engineered for regulatory precision
- **RAG Basics:** Architecture designed for grounding against live FDA DERA API (510(k) database) and FDA guidance documents
- **Building:** 4-page Streamlit app with working LLM extraction, comparison engine, match scoring, and strategy builder
- **Model Tradeoffs:** Claude Sonnet chosen for complex regulatory document reasoning; structured JSON output reliability critical
- **Latency & Cost:** ~$0.06/510(k) analysis; full predicate set (5 devices) = $0.30 per analysis run
- **Failure Modes:** Conservative extraction prompt; hallucination risk mitigated by requiring direct citation of K-numbers and standards

**Business Case:**
A single FDA additional information request costs 30–45 days of review time and typically $50,000–$200,000 in R&D carrying costs. If this tool prevents one AI request per program, it pays for itself 100x over. At scale (20 EP programs/year), the value is in the millions.
        """)

    with col2:
        st.markdown("### Project Info")
        st.info("""
**Built by:** Saurabh

**Role:** AI Product Manager (Portfolio)

**Stack:**
- Python 3.11
- Streamlit
- Plotly
- Anthropic Claude API
- Pandas

**Domain:**
EP MedTech · FDA 510(k)
Regulatory Affairs · R&D

**Predicate Database:**
5 real EP 510(k) summaries
(K221847, K213291, K192516,
K201823, K183021)

**Status:** Portfolio Demo
        """)
        st.markdown("### Links")
        st.markdown("🔗 [GitHub Repository](#)")
        st.markdown("🔗 [LinkedIn Profile](#)")
        st.markdown("🌐 [thecuriousdetour.ca](#)")
