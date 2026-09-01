"""
Enterprise HR AI - Workforce Intelligence Platform - Streamlit dashboard.

Run with:
    streamlit run frontend/streamlit_app.py
(from the enterprise_hr_ai/ project root, with the FastAPI backend running
on http://127.0.0.1:8000 - see app/main.py)
"""
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import requests

st.set_page_config(page_title="Enterprise HR AI - Workforce Intelligence Platform",
                    page_icon="🧠", layout="wide")

API_BASE = "http://127.0.0.1:8000"

# ---------------------------------------------------------------------------
# Small styling pass so charts/cards feel like a single cohesive dark dashboard
# rather than default Streamlit widgets.
# ---------------------------------------------------------------------------
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.6rem; }
.block-container { padding-top: 2rem; }
</style>
""", unsafe_allow_html=True)

plt.rcParams.update({
    "figure.facecolor": "#0E1117",
    "axes.facecolor": "#0E1117",
    "axes.edgecolor": "#444",
    "axes.labelcolor": "#FAFAFA",
    "text.color": "#FAFAFA",
    "xtick.color": "#FAFAFA",
    "ytick.color": "#FAFAFA",
    "grid.color": "#333",
})

RISK_COLORS = {"HIGH": "#FF4B4B", "MEDIUM": "#F5C542", "LOW": "#3ECF8E"}


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=30)
def fetch(endpoint, params=None):
    try:
        r = requests.get(f"{API_BASE}{endpoint}", params=params, timeout=10)
        r.raise_for_status()
        return r.json()
    except Exception as e:
        st.error(f"Could not reach API at {API_BASE}{endpoint} - is `uvicorn app.main:app` running? ({e})")
        return None


def post(endpoint, payload):
    r = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=15)
    return r


@st.cache_data(ttl=30)
def load_roster():
    data = fetch("/employees")
    return pd.DataFrame(data) if data else pd.DataFrame()


# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.title("🧠 Enterprise HR AI — Workforce Intelligence Platform")
st.caption("Predictive Attrition Risk · Skill Gap Analytics · Financial Cost Exposure · Policy Simulation")

roster = load_roster()

if roster.empty:
    st.warning("No employee data available yet - check that the API is running and "
               "`data/processed/employee_intelligence.csv` is populated.")
    st.stop()

# ---------------------------------------------------------------------------
# Global, real-time filters (sidebar) - every tab below reacts to these
# instantly because they all filter the same in-memory `roster` dataframe,
# no re-fetch needed.
# ---------------------------------------------------------------------------
st.sidebar.header("🔎 Global Dashboard Filters")
departments = sorted(roster["Department"].unique().tolist())
selected_departments = st.sidebar.multiselect("Select Department(s)", departments, default=departments)

risk_tiers = ["HIGH", "MEDIUM", "LOW"]
selected_risk = st.sidebar.multiselect("Filter by Attrition Risk Tier", risk_tiers, default=risk_tiers)

filtered = roster[
    roster["Department"].isin(selected_departments) & roster["Risk"].isin(selected_risk)
].copy()

if filtered.empty:
    st.sidebar.warning("No employees match the current filters.")

st.sidebar.divider()
st.sidebar.metric("Employees matching filters", len(filtered))

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------
tab_exec, tab_skills, tab_whatif, tab_financial, tab_drilldown, tab_chat = st.tabs([
    "📊 Executive Dashboard",
    "🎓 Skill Gap & Upskilling",
    "🧪 What-If Policy Simulator",
    "💰 Financial Cost Exposure",
    "👤 Employee Drill-Down",
    "💬 HR Assistant Chat",
])

# ===========================================================================
# TAB 1 — EXECUTIVE DASHBOARD
# ===========================================================================
with tab_exec:
    summary = fetch("/dashboard/summary")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Workforce", f"{len(filtered):,}")
    high_risk_n = int((filtered["Risk"] == "HIGH").sum())
    high_risk_pct = round(100 * high_risk_n / len(filtered), 1) if len(filtered) else 0
    c2.metric("High Flight Risk Count", f"{high_risk_n:,}", f"{high_risk_pct}%")
    est_exposure = (filtered["MonthlyIncome"] * 12 * 1.5 * filtered["Attrition_Prob"]).sum()
    c3.metric("Financial Cost Exposure", f"${est_exposure:,.0f}")
    if summary:
        c4.metric("Avg Engagement Rating", f"{summary['average_engagement_index']}/100")

    st.caption("Financial Cost Exposure here uses a default 1.5x turnover-cost multiplier — "
               "tune it precisely in the 💰 Financial Cost Exposure tab.")
    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Attrition Risk Tier Distribution by Department")
        if not filtered.empty:
            pivot = filtered.groupby(["Department", "Risk"]).size().unstack(fill_value=0)
            pivot = pivot.reindex(columns=[r for r in ["LOW", "MEDIUM", "HIGH"] if r in pivot.columns])
            fig, ax = plt.subplots(figsize=(5.5, 4))
            bottom = None
            for risk in pivot.columns:
                ax.bar(pivot.index, pivot[risk], bottom=bottom, label=risk,
                       color=RISK_COLORS.get(risk, "#888"))
                bottom = pivot[risk] if bottom is None else bottom + pivot[risk]
            ax.set_ylabel("Employee count")
            ax.legend(title="Risk tier")
            plt.xticks(rotation=20, ha="right")
            st.pyplot(fig, clear_figure=True)

    with col_right:
        st.subheader("Attrition Probability vs. Monthly Income")
        if not filtered.empty:
            fig, ax = plt.subplots(figsize=(5.5, 4))
            for risk in ["LOW", "MEDIUM", "HIGH"]:
                sub = filtered[filtered["Risk"] == risk]
                ax.scatter(sub["MonthlyIncome"], sub["Attrition_Prob"],
                           s=14, alpha=0.7, label=risk, color=RISK_COLORS.get(risk))
            ax.set_xlabel("Monthly Income ($)")
            ax.set_ylabel("Attrition Probability")
            ax.legend(title="Risk tier")
            st.pyplot(fig, clear_figure=True)

    st.divider()
    st.subheader("Department Breakdown")
    if not filtered.empty:
        dept_table = (filtered.groupby("Department")
                      .agg(Employees=("EmployeeNumber", "count"),
                           High_Risk=("Risk", lambda s: (s == "HIGH").sum()),
                           Avg_Attrition_Prob=("Attrition_Prob", "mean"))
                      .reset_index())
        dept_table["Avg_Attrition_Prob"] = dept_table["Avg_Attrition_Prob"].round(3)
        st.dataframe(dept_table, use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 2 — SKILL GAP & UPSKILLING
# ===========================================================================
with tab_skills:
    st.subheader("Organization-Wide Skill Gap & Upskilling Paths")

    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("**Top High-Severity Skill Gaps**")
        gap_data = fetch("/dashboard/skill-gaps")
        if gap_data:
            gap_df = pd.DataFrame(gap_data).sort_values("employees_missing", ascending=True).tail(10)
            fig, ax = plt.subplots(figsize=(5.5, 4))
            colors = [RISK_COLORS.get(s, "#888") for s in gap_df["severity"]]
            ax.barh(gap_df["skill"], gap_df["employees_missing"], color=colors)
            ax.set_xlabel("Employees missing this skill")
            st.pyplot(fig, clear_figure=True)
            st.caption("🔴 HIGH severity  🟡 MEDIUM severity  🟢 LOW severity")

    with col_right:
        st.markdown("**Recommended Upskilling Course Enrollments** (filtered employees)")
        recs = fetch("/dashboard/recommendations")
        if recs and not filtered.empty:
            recs_df = pd.DataFrame(recs)
            recs_df = recs_df[recs_df["EmployeeNumber"].isin(filtered["EmployeeNumber"])]
            recs_df["course"] = recs_df["recommendation"].str.split("-> ").str[-1]
            course_counts = recs_df["course"].value_counts().head(6)
            if not course_counts.empty:
                fig, ax = plt.subplots(figsize=(5.5, 4))
                ax.pie(course_counts.values, labels=course_counts.index, autopct="%1.0f%%",
                       wedgeprops={"width": 0.45}, textprops={"fontsize": 8})
                st.pyplot(fig, clear_figure=True)

    st.divider()
    st.subheader("All Upskilling Recommendations (filtered)")
    recs = fetch("/dashboard/recommendations")
    if recs:
        recs_df = pd.DataFrame(recs)
        recs_df = recs_df[recs_df["EmployeeNumber"].isin(filtered["EmployeeNumber"])]
        st.dataframe(recs_df.head(200), use_container_width=True, hide_index=True)

# ===========================================================================
# TAB 3 — WHAT-IF POLICY SIMULATOR
# ===========================================================================
with tab_whatif:
    st.subheader("🧪 Interactive What-If Policy Simulator")
    st.caption("Simulate how policy interventions (compensation hikes, overtime elimination, "
               "work-life balance improvement) change predicted flight risk for one employee.")

    if filtered.empty:
        st.info("No employees match the current filters.")
    else:
        options = filtered.apply(lambda r: f"{r['EmployeeNumber']} — {r['EmployeeName']}", axis=1).tolist()
        picked = st.selectbox("Select Employee to Simulate For", options)
        emp_id = int(picked.split(" — ")[0])

        baseline = fetch(f"/employees/{emp_id}/raw")
        full_record = fetch(f"/employees/{emp_id}")

        if baseline and full_record:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"### Baseline Profile — {baseline['EmployeeName']}")
                st.write(f"**Department:** {baseline['Department']}  |  **Role:** {baseline['JobRole']}")
                st.write(f"**Current Monthly Income:** ${baseline['MonthlyIncome']:,}")
                st.write(f"**Current OverTime Status:** {baseline['OverTime']}")
                st.write(f"**Current Work-Life Balance Rating:** {baseline['WorkLifeBalance']} / 4")
                st.write(f"**Baseline Predicted Attrition Risk:** "
                         f"{full_record['Attrition_Prob']:.0%} ({full_record['Risk']})")

            with col_b:
                st.markdown("### Hypothetical Policy Interventions")
                salary_hike_pct = st.slider("Salary Increase (%)", 0, 30, 0, step=5)
                remove_overtime = st.toggle("Eliminate OverTime", value=(baseline["OverTime"] == "Yes"))
                target_wlb = st.slider("Target Work-Life Balance Rating", 1, 4, int(baseline["WorkLifeBalance"]))

                if st.button("▶ Execute What-If Simulation", type="primary"):
                    sim_record = dict(baseline)
                    sim_record.pop("EmployeeName", None)
                    sim_record["MonthlyIncome"] = int(round(baseline["MonthlyIncome"] * (1 + salary_hike_pct / 100)))
                    sim_record["OverTime"] = "No" if remove_overtime else "Yes"
                    sim_record["WorkLifeBalance"] = target_wlb

                    resp = post("/predict/attrition", sim_record)
                    if resp.status_code == 200:
                        new = resp.json()
                        st.divider()
                        r1, r2 = st.columns(2)
                        r1.metric("Baseline Attrition Risk", f"{full_record['Attrition_Prob']:.0%}",
                                   full_record["Risk"])
                        delta = new["attrition_probability"] - full_record["Attrition_Prob"]
                        r2.metric("Simulated Attrition Risk", f"{new['attrition_probability']:.0%}",
                                   f"{delta:+.0%}", delta_color="inverse")
                        if new["risk_level"] != full_record["Risk"]:
                            st.success(f"Risk tier would move from **{full_record['Risk']}** "
                                       f"to **{new['risk_level']}** under this scenario.")
                        else:
                            st.info(f"Risk tier stays **{new['risk_level']}**, "
                                    f"but probability shifts by {delta:+.1%}.")
                    else:
                        st.error(f"Simulation failed: {resp.text}")

# ===========================================================================
# TAB 4 — FINANCIAL COST EXPOSURE
# ===========================================================================
with tab_financial:
    st.subheader("💰 Financial Attrition Cost Exposure Model")
    multiplier = st.slider("Turnover Cost Multiplier (x annual salary)", 0.5, 3.0, 1.5, step=0.1)

    fin = fetch("/dashboard/financial-exposure", params={"turnover_cost_multiplier": multiplier})
    if fin:
        emp_df = pd.DataFrame(fin["employees"])
        emp_df = emp_df[emp_df["EmployeeNumber"].isin(filtered["EmployeeNumber"])]

        total_filtered = emp_df["Financial_Exposure"].sum()
        high_risk_filtered = emp_df.loc[emp_df["Risk"] == "HIGH", "Financial_Exposure"].sum()

        c1, c2 = st.columns(2)
        c1.metric("Total Projected Cost Exposure", f"${total_filtered:,.2f}")
        c2.metric("High-Risk Exposure Portion", f"${high_risk_filtered:,.2f}")

        st.divider()
        st.subheader("Cost Exposure by Department ($)")
        by_dept = emp_df.groupby("Department")["Financial_Exposure"].sum().sort_values(ascending=False)
        if not by_dept.empty:
            fig, ax = plt.subplots(figsize=(9, 4))
            ax.bar(by_dept.index, by_dept.values, color="#FF4B4B")
            ax.set_ylabel("Financial Exposure ($)")
            plt.xticks(rotation=15, ha="right")
            st.pyplot(fig, clear_figure=True)

        with st.expander("See per-employee exposure (filtered)"):
            st.dataframe(
                emp_df.sort_values("Financial_Exposure", ascending=False).head(100),
                use_container_width=True, hide_index=True,
            )

# ===========================================================================
# TAB 5 — EMPLOYEE DRILL-DOWN
# ===========================================================================
with tab_drilldown:
    st.subheader("👤 Single-Employee Intelligence Profile")

    if filtered.empty:
        st.info("No employees match the current filters.")
    else:
        options = filtered.apply(lambda r: f"{r['EmployeeNumber']} — {r['EmployeeName']}", axis=1).tolist()
        picked = st.selectbox("Select Employee for Drill-Down", options, key="drilldown_emp")
        emp_id = int(picked.split(" — ")[0])

        record = fetch(f"/employees/{emp_id}")
        raw = fetch(f"/employees/{emp_id}/raw")

        if record and raw:
            col_a, col_b = st.columns(2)
            with col_a:
                st.markdown(f"### Profile: {record['EmployeeName']}")
                st.write(f"**Department:** {record['Department']}")
                st.write(f"**HR Job Role:** {record['JobRole']}")
                st.write(f"**Monthly Income:** ${raw['MonthlyIncome']:,}")
                st.write(f"**Tenure at Company:** {raw['YearsAtCompany']} years")
                st.write(f"**Years Since Last Promotion:** {raw['YearsSinceLastPromotion']} years")
                st.write(f"**OverTime Status:** {raw['OverTime']}")

            with col_b:
                st.markdown("### Risk & Upskilling Analysis")
                risk_emoji = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(record["Risk"], "")
                st.write(f"**Predicted Flight Risk:** {record['Attrition_Prob']:.0%} "
                         f"{risk_emoji} ({record['Risk']})")
                st.write(f"**Missing Skills Count:** {record['gap_count']} skills")
                st.write(f"**Missing Skills:** {record['skill_gap'] or '—'}")
                st.write(f"**Top Recommended Course Path:** :green[{record['recommendation']}]")

        st.divider()
        st.subheader("Career Path Simulation")
        career = fetch(f"/career/{emp_id}/path")
        if career:
            if career.get("next_role"):
                st.write(f"**{career['current_role']}** → **{career['next_role']}** "
                         f"({career.get('next_role_occupation_title', '')})")
                st.progress(career["readiness_pct"] / 100, text=f"Readiness: {career['readiness_pct']}%")
                c1, c2 = st.columns(2)
                c1.write("**Has:**")
                c1.write(", ".join(career["skills_have"]) or "—")
                c2.write("**Missing:**")
                c2.write(", ".join(career["skills_missing"]) or "—")
            else:
                st.info(career.get("message", "No next role defined for this role."))

# ===========================================================================
# TAB 6 — HR ASSISTANT CHAT
# ===========================================================================
with tab_chat:
    st.subheader("💬 HR Assistant Chat")
    st.caption("Ask a policy question, or ask about a specific employee by ID. "
               "Routes through specialized agents with a permission layer based on your role.")

    col_role, col_emp = st.columns([1, 1])
    with col_role:
        chat_role = st.selectbox("Ask as", ["employee", "manager", "hr_admin"],
                                  help="hr_admin unlocks salary-related questions; "
                                       "manager unlocks attrition-risk questions.")
    with col_emp:
        emp_pick_options = ["(none)"] + filtered.apply(
            lambda r: f"{r['EmployeeNumber']} — {r['EmployeeName']}", axis=1
        ).tolist()
        emp_pick = st.selectbox("Employee (optional, for profile/risk/career/upskilling questions)",
                                 emp_pick_options)
        chat_emp_id = None if emp_pick == "(none)" else int(emp_pick.split(" — ")[0])

    SAMPLE_QUESTIONS = [
        "What is the parental leave policy?",
        "How much PTO do I get per year?",
        "Can I work remotely?",
        "What's the meal expense limit while traveling?",
        "What benefits does the company offer?",
        "Is this employee at risk of leaving?",
        "What skills is this employee missing?",
        "What course should this employee take next?",
        "What is this employee's career path?",
        "Show me every employee's salary",  # deliberately hr_admin-only, to demo the permission layer
    ]

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "chat_input" not in st.session_state:
        st.session_state.chat_input = ""

    st.markdown("**Try asking:**")
    cols = st.columns(3)
    for i, q in enumerate(SAMPLE_QUESTIONS):
        if cols[i % 3].button(q, key=f"quick_{i}", use_container_width=True):
            # Set session_state BEFORE the widget below is instantiated this run,
            # so the text_input picks it up (Streamlit ignores `value=` once a
            # keyed widget already has a session_state entry).
            st.session_state.chat_input = q

    user_message = st.text_input("Your question",
                                  placeholder="e.g. 'What is the parental leave policy?' "
                                              "or 'Is this employee at risk?'",
                                  key="chat_input")

    if st.button("Ask", type="primary") and user_message.strip():
        payload = {"message": user_message, "caller_role": chat_role}
        if chat_emp_id:
            payload["employee_id"] = chat_emp_id
        try:
            resp = post("/agent/chat", payload)
            body = resp.json()
            st.session_state.chat_history.append((user_message, chat_role, body))
        except Exception as e:
            st.error(f"Could not reach API: {e}")

    for question, role, body in reversed(st.session_state.chat_history):
        with st.chat_message("user"):
            st.write(f"**({role})** {question}")
        with st.chat_message("assistant"):
            st.caption(f"Routed to: `{body.get('agent', 'unknown')}`")
            if body.get("status") == "permission_denied":
                st.error(body["error"])
            elif "error" in body:
                st.warning(body["error"])
            else:
                result = body["result"]
                if isinstance(result, dict) and "answer" in result:
                    # Policy RAG answer - show as readable text, not raw JSON
                    st.write(result["answer"])
                    if result.get("sources"):
                        st.caption("Sources: " + ", ".join(result["sources"]))
                else:
                    st.json(result)
