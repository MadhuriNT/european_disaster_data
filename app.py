import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(
    page_title="Europe Natural Disasters 1975–2025",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
    .metric-card {
        background: #1a1d2e;
        border: 1px solid #2d3152;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 600; color: #7c83ff; }
    .metric-label { font-size: 0.8rem; color: #8b8fa8; margin-top: 4px; }
    .section-header {
        font-size: 1.1rem; font-weight: 600;
        color: #e2e4f0; margin: 1.5rem 0 0.5rem;
        border-left: 3px solid #7c83ff;
        padding-left: 10px;
    }
    .cause-box {
        background: #1a1d2e;
        border-radius: 8px;
        padding: 14px 18px;
        margin: 6px 0;
        border: 1px solid #2d3152;
        font-size: 0.88rem;
        color: #c4c7dc;
    }
    .cause-box b { color: #a5a9ff; }
</style>
""", unsafe_allow_html=True)

# ── Colour palette ───────────────────────────────────────────────────────────
DISASTER_COLORS = {
    "Floods":       "#378ADD",
    "Earthquakes":  "#E24B4A",
    "Storms":       "#EF9F27",
    "Heatwaves":    "#BA7517",
    "Wildfires":    "#639922",
    "Landslides":   "#7F77DD",
    "Droughts":     "#1D9E75",
    "Extreme Cold": "#D4537E",
}

# ── Causes text ──────────────────────────────────────────────────────────────
CAUSES = {
    "Floods":       "Heavy prolonged rainfall, snowmelt, river overflow. Intensified by climate change — warmer air holds ~7 % more moisture per 1 °C (Clausius-Clapeyron effect).",
    "Earthquakes":  "African-Eurasian plate boundary collision. Apennine extensional faults (Italy); North Anatolian fault (Turkey). Frequency is unchanged by climate.",
    "Storms":       "Atlantic extratropical cyclones; rapid cyclogenesis. Medicanes (Mediterranean hurricanes) are increasing post-2000 as sea-surface temperatures rise.",
    "Heatwaves":    "Persistent blocking anticyclones + rising baseline temperatures from climate change. Urban heat islands amplify readings by 2–5 °C inside cities.",
    "Wildfires":    "Longer, hotter, drier summers extend the fire season (now ~3 weeks longer than 1979). Rural land abandonment and invasive species increase fuel loads.",
    "Landslides":   "Intense rainfall on steep terrain; deforestation; permafrost thaw in the Alps destabilising slopes that were previously frozen year-round.",
    "Droughts":     "Blocking anticyclones divert Atlantic rainfall northward. Mediterranean drying trend is well-established; Alpine snowpack declining every decade.",
    "Extreme Cold": "Siberian anticyclone intrusions and polar vortex disruptions. Paradoxically, some events are linked to Arctic warming weakening the vortex.",
}

# ── Annual event counts (EM-DAT / EEA based) ────────────────────────────────
years = list(range(1975, 2026))  # 51 years

raw = {
    "Floods":      [4,5,3,6,4,5,4,5,3,4,5,4,6,5,5,7,6,5,8,7,9,6,7,8,9,8,
                    9,12,7,8,8,9,10,9,10,11,10,11,12,10,11,11,10,12,11,12,14,13,15,14,13],
    "Earthquakes": [3,6,4,2,3,2,3,2,2,2,3,2,3,3,2,4,3,3,2,3,3,2,4,3,3,4,
                    4,3,3,3,4,3,4,4,5,4,5,5,3,4,4,5,4,4,4,3,4,4,5,4,4],
    "Storms":      [3,2,3,3,4,3,3,4,3,3,4,3,5,3,3,6,4,3,4,4,5,4,4,4,5,5,
                    5,5,5,5,5,5,6,5,5,6,5,5,6,6,5,6,5,6,6,6,6,7,7,7,6],
    "Heatwaves":   [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
                    1,1,5,2,2,2,2,2,2,2,2,2,2,2,3,2,2,3,4,3,4,5,5,5,4],
    "Wildfires":   [2,2,2,2,2,2,2,2,2,2,2,2,3,2,5,4,3,3,3,3,3,3,3,3,3,3,
                    3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,6,7,6,5,5],
    "Landslides":  [1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,1,1,2,2,2,1,2,3,2,2,
                    2,2,2,2,2,2,2,2,3,3,3,3,3,3,3,3,3,3,3,3,4,4,4,4,3],
    "Droughts":    [1,1,1,1,1,1,1,1,1,1,1,1,1,1,2,2,3,3,1,1,2,2,1,1,1,2,
                    2,1,4,1,1,2,2,1,1,2,2,3,1,2,2,2,2,4,3,2,3,5,3,4,3],
    "Extreme Cold":[2,2,2,2,3,2,2,2,3,2,4,2,4,2,2,2,2,2,2,2,2,3,2,2,2,2,
                    2,2,1,2,2,2,2,2,2,3,2,3,2,2,2,2,2,3,2,2,2,2,2,2,2],
}

# Notable historical events for annotations
NOTABLE = {
    1976: "Friuli EQ",
    1980: "S.Italy EQ",
    1987: "Great Storm",
    1988: "Spitak EQ",
    1990: "Triple Storms",
    1993: "Rhine Floods",
    1999: "Lothar/Martin",
    2002: "Elbe Floods",
    2003: "Heatwave",
    2009: "L'Aquila EQ",
    2016: "Amatrice EQ",
    2021: "Ahr Floods",
    2022: "Record Drought",
    2023: "Turkey EQ",
}

df = pd.DataFrame(raw, index=years)
df.index.name = "Year"
df["Total"] = df.sum(axis=1)

# ── Sidebar controls ─────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🌍 Controls")
    st.markdown("---")

    year_range = st.slider(
        "Year range",
        min_value=1975, max_value=2025,
        value=(1975, 2025), step=1
    )

    selected_types = st.multiselect(
        "Disaster types",
        options=list(DISASTER_COLORS.keys()),
        default=list(DISASTER_COLORS.keys())
    )

    chart_type = st.selectbox(
        "Chart style",
        ["Stacked Bar", "Line Chart", "Area Chart", "Heatmap"]
    )

    show_notable = st.checkbox("Show major events", value=True)
    show_causes  = st.checkbox("Show causes panel", value=True)

    st.markdown("---")
    st.caption("Sources: EM-DAT · EEA · Munich Re · IPCC AR6")

# ── Filter ───────────────────────────────────────────────────────────────────
mask = (df.index >= year_range[0]) & (df.index <= year_range[1])
dff  = df.loc[mask, selected_types].copy() if selected_types else df.loc[mask, list(DISASTER_COLORS.keys())].copy()
dff["Total"] = dff.sum(axis=1)

# ── Page title ───────────────────────────────────────────────────────────────
st.title("🌍 Natural Disasters in Europe  1975 – 2025")
st.caption(f"Showing {year_range[0]}–{year_range[1]}  ·  {len(selected_types)} disaster type(s) selected")

# ── KPI cards ────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
total_events = int(dff["Total"].sum())
worst_year   = int(dff["Total"].idxmax())
avg_yr       = round(dff["Total"].mean(), 1)
worst_type   = dff[selected_types].sum().idxmax() if selected_types else "—"

for col, val, label in [
    (c1, total_events, "Total events"),
    (c2, worst_year,   "Busiest year"),
    (c3, avg_yr,       "Avg events / year"),
    (c4, worst_type,   "Most frequent type"),
]:
    col.markdown(
        f'<div class="metric-card">'
        f'<div class="metric-value">{val}</div>'
        f'<div class="metric-label">{label}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Main chart ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Annual disaster events</div>', unsafe_allow_html=True)

BG = "#0f1117"

if chart_type == "Stacked Bar":
    melted = dff[selected_types].reset_index().melt(id_vars="Year", var_name="Type", value_name="Events")
    fig = px.bar(melted, x="Year", y="Events", color="Type",
                 color_discrete_map=DISASTER_COLORS, template="plotly_dark")
    fig.update_layout(barmode="stack")

elif chart_type == "Line Chart":
    fig = go.Figure()
    for dtype in selected_types:
        fig.add_trace(go.Scatter(
            x=dff.index, y=dff[dtype], name=dtype,
            line=dict(color=DISASTER_COLORS[dtype], width=2),
            mode="lines+markers", marker_size=4
        ))

elif chart_type == "Area Chart":
    melted = dff[selected_types].reset_index().melt(id_vars="Year", var_name="Type", value_name="Events")
    fig = px.area(melted, x="Year", y="Events", color="Type",
                  color_discrete_map=DISASTER_COLORS, template="plotly_dark")

else:  # Heatmap
    fig = px.imshow(dff[selected_types].T, aspect="auto",
                    color_continuous_scale="Blues",
                    labels=dict(x="Year", y="Type", color="Events"),
                    template="plotly_dark")

fig.update_layout(
    paper_bgcolor=BG, plot_bgcolor=BG,
    legend_title_text="Disaster type", height=420,
    margin=dict(l=10, r=10, t=20, b=30)
)

# Notable event annotations
if show_notable and chart_type != "Heatmap":
    for yr, label in NOTABLE.items():
        if year_range[0] <= yr <= year_range[1] and yr in dff.index:
            fig.add_vline(x=yr, line_dash="dot",
                          line_color="rgba(255,255,255,0.2)", line_width=1)
            fig.add_annotation(
                x=yr, y=dff.loc[yr, "Total"],
                text=label, showarrow=True,
                arrowhead=2, arrowcolor="#7c83ff",
                font=dict(size=9, color="#a5a9ff"),
                bgcolor="#1a1d2e", bordercolor="#2d3152",
                ax=20, ay=-32
            )

st.plotly_chart(fig, use_container_width=True)

# ── Two column row ───────────────────────────────────────────────────────────
left, right = st.columns(2)

with left:
    st.markdown('<div class="section-header">Total by disaster type</div>', unsafe_allow_html=True)
    if selected_types:
        totals = dff[selected_types].sum().reset_index()
        totals.columns = ["Type", "Events"]
        totals = totals.sort_values("Events", ascending=True)
        fig2 = px.bar(totals, x="Events", y="Type", orientation="h",
                      color="Type", color_discrete_map=DISASTER_COLORS,
                      template="plotly_dark")
        fig2.update_layout(paper_bgcolor=BG, plot_bgcolor=BG,
                           showlegend=False, height=300,
                           margin=dict(l=0, r=10, t=10, b=20))
        st.plotly_chart(fig2, use_container_width=True)

with right:
    st.markdown('<div class="section-header">Decade comparison</div>', unsafe_allow_html=True)
    periods = {
        "1975–84": (1975, 1984), "1985–94": (1985, 1994),
        "1995–04": (1995, 2004), "2005–14": (2005, 2014),
        "2015–25": (2015, 2025),
    }
    decade_rows = {}
    for lbl, (s, e) in periods.items():
        pmask = (df.index >= max(s, year_range[0])) & (df.index <= min(e, year_range[1]))
        if pmask.any() and selected_types:
            decade_rows[lbl] = df.loc[pmask, selected_types].sum()
    if decade_rows:
        dec_df = pd.DataFrame(decade_rows).T
        melted3 = dec_df.reset_index().melt(id_vars="index", var_name="Type", value_name="Events")
        fig3 = px.bar(melted3, x="index", y="Events", color="Type",
                      color_discrete_map=DISASTER_COLORS, template="plotly_dark")
        fig3.update_layout(paper_bgcolor=BG, plot_bgcolor=BG, barmode="stack",
                           xaxis_title="", showlegend=False, height=300,
                           margin=dict(l=0, r=10, t=10, b=20))
        st.plotly_chart(fig3, use_container_width=True)

# ── Trend line ───────────────────────────────────────────────────────────────
st.markdown('<div class="section-header">Trend — rolling 5-year average</div>', unsafe_allow_html=True)
roll = dff["Total"].rolling(5, center=True).mean()
fig4 = go.Figure()
fig4.add_trace(go.Bar(x=dff.index, y=dff["Total"], name="Annual total",
                      marker_color="rgba(124,131,255,0.3)"))
fig4.add_trace(go.Scatter(x=roll.index, y=roll, name="5-yr average",
                          line=dict(color="#7c83ff", width=2.5)))

if len(dff) > 2:
    xv = np.array(dff.index)
    yv = dff["Total"].values
    z  = np.polyfit(xv, yv, 1)   # numpy: fit a straight line (y = mx + c)
    p  = np.poly1d(z)
    fig4.add_trace(go.Scatter(
        x=dff.index, y=p(xv),
        name=f"Linear trend (+{z[0]:.2f} events/yr)",
        line=dict(color="#E24B4A", width=1.5, dash="dash")
    ))

fig4.update_layout(template="plotly_dark", paper_bgcolor=BG, plot_bgcolor=BG,
                   height=280, margin=dict(l=10, r=10, t=10, b=30))
st.plotly_chart(fig4, use_container_width=True)

# ── Raw data table ───────────────────────────────────────────────────────────
with st.expander("📊 View raw data table"):
    show_cols = selected_types + ["Total"] if selected_types else ["Total"]
    st.dataframe(
        dff[show_cols].style.background_gradient(cmap="Blues", subset=["Total"]),
        use_container_width=True
    )
    st.download_button(
        "⬇ Download as CSV",
        data=dff[show_cols].to_csv(),
        file_name="europe_disasters.csv",
        mime="text/csv"
    )

# ── Causes panel ─────────────────────────────────────────────────────────────
if show_causes and selected_types:
    st.markdown('<div class="section-header">Causes & context</div>', unsafe_allow_html=True)
    cols = st.columns(2)
    for i, dtype in enumerate(selected_types):
        color = DISASTER_COLORS.get(dtype, "#888")
        cols[i % 2].markdown(
            f'<div class="cause-box">'
            f'<b style="color:{color}">{dtype}</b><br>'
            f'{CAUSES.get(dtype, "—")}'
            f'</div>',
            unsafe_allow_html=True
        )

st.markdown("---")
st.caption("Data: EM-DAT (CRED) · European Environment Agency · Munich Re NatCatSERVICE · IPCC AR6")
