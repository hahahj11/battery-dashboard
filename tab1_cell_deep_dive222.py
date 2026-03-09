import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import trapezoid

# =========================
# Config
# =========================
DISCHARGE_PATH = "discharge_all.csv"
ENRICHED_PATH = "train_table_discharge_enriched_v2.csv"

TARGET_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
EOL_THRESHOLD = 70.0
BASELINE_CYCLES = (1, 5)
WAVE_N_GRID = 250
TEMP_LIMIT = 45.0


# =========================
# Page + Style
# =========================
st.set_page_config(page_title="Cell Deep Dive", layout="wide")
st.markdown(
    """
    <style>
    :root {
        --bg:#eef2ff;
        --panel:#ffffff;
        --line:#d8e0f0;
        --text:#1a2438;
        --muted:#65728b;
        --blue:#2f63c8;
        --blue2:#5f86d9;
        --yellow:#f6be2e;
        --pink:#df5f8b;
    }

    .stApp {
        background:
            radial-gradient(circle at 5% 10%, rgba(99,123,220,0.10) 0%, rgba(99,123,220,0) 30%),
            radial-gradient(circle at 90% 90%, rgba(132,159,228,0.14) 0%, rgba(132,159,228,0) 38%),
            linear-gradient(180deg, #f5f8ff 0%, #edf2ff 100%);
    }

    .dash-header {
        background: linear-gradient(95deg, #35579a, #2a4d90 65%, #244585);
        color: white;
        padding: 16px 20px;
        border-radius: 16px;
        font-weight: 800;
        font-size: 26px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 10px 28px rgba(37, 64, 120, 0.22);
        margin-bottom: 10px;
    }

    .brand {
        font-size: 24px;
        letter-spacing: 1px;
        opacity: 0.95;
    }

    .control-shell {
        background: rgba(255,255,255,0.82);
        border: 1px solid var(--line);
        border-radius: 12px;
        padding: 10px 14px 4px 14px;
        margin-bottom: 12px;
    }

    .metric-card {
        overflow: visible !important;
    }

    .metric-main {
        overflow: visible !important;
    }

    div[data-testid="stHorizontalBlock"] {
        overflow: visible !important;
    }

    div[data-testid="column"] {
        overflow: visible !important;
    }

    .metric-card {
        background: linear-gradient(180deg, #ffffff 0%, #f9fbff 100%);
        border: 1px solid var(--line);
        border-radius: 12px;
        box-shadow: 0 4px 14px rgba(46, 69, 120, 0.09);
        overflow: hidden;
        min-height: 146px;
        display: flex;
        flex-direction: column;
    }

    .metric-main {
        padding: 13px 14px 9px 14px;
        overflow: hidden;
        flex: 1;
    }

    .metric-label {
        color: #35445f;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 4px;
    }

    .metric-value {
        color: var(--text);
        font-size: 30px;
        line-height: 1;
        font-weight: 900;
        letter-spacing: -0.5px;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    .metric-value.small {
        font-size: 30px;
    }

    .metric-sub {
        color: var(--muted);
        font-size: 13px;
        margin-top: 6px;
    }

    .metric-foot {
        background: linear-gradient(90deg, #3f77d9, #2f63c8);
        color: #eaf1ff;
        font-weight: 700;
        font-size: 12px;
        padding: 6px 10px;
        min-height: 28px;
        display: flex;
        align-items: center;
        gap: 6px;
        border-radius: 8px;
        margin: 0 8px 8px 8px;
    }

    .metric-label-row {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 4px;
        flex-wrap: nowrap;
    }

    .metric-label {
        color: #35445f;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 0;
    }

    .panel {
        background: var(--panel);
        border: 1px solid var(--line);
        border-radius: 14px;
        box-shadow: 0 6px 18px rgba(44, 62, 105, 0.08);
        padding: 12px;
        height: 100%;
        overflow: hidden;
    }

    .panel-title {
        font-size: 17px;
        font-weight: 900;
        color: #1f2c46;
        margin-bottom: 8px;
    }

    .chip {
        display: inline-flex;
        align-items: center;
        border-radius: 9px;
        padding: 7px 11px;
        font-weight: 800;
        font-size: 13px;
        margin-right: 8px;
        margin-bottom: 8px;
        border: 1px solid rgba(0,0,0,0.06);
    }
    .chip-wrap {
        display: flex;
        flex-wrap: wrap;
        gap: 8px;
        width: 100%;
        max-width: 100%;
        margin-bottom: 6px;
    }

    .chip.yellow { background: #ffd451; color: #5e4300; }
    .chip.blue { background: #5b82e0; color: #edf3ff; }
    .chip.teal { background: #3f6ec0; color: #edf3ff; }
    .chip.pink { background: #d96592; color: #fff1f7; }

    .note-list {
        margin: 8px 0 8px 0;
        color: #24314a;
        font-size: 14px;
        line-height: 1.55;
        max-width: 100%;
        white-space: normal;
        word-break: keep-all;
        overflow-wrap: anywhere;
        padding-left: 18px;
    }

    .note-list li {
        margin-bottom: 6px;
    }

    .insight-wrap {
        width: 100%;
        max-width: 100%;
        overflow: hidden;
    }

    .small-note {
        color: var(--muted);
        font-size: 12px;
        margin-top: 4px;
    }

    .wave-btn {
        margin-top: 8px;
        background: linear-gradient(90deg, #4f79cc, #3f67bd);
        color: white;
        font-weight: 800;
        border-radius: 8px;
        text-align: center;
        padding: 8px;
    }

    .top1-badge {
        display: inline-block;
        margin-top: 6px;
        padding: 6px 10px;
        border-radius: 10px;
        background: #2f63c8;
        color: #ffffff;
        font-weight: 800;
        font-size: 12px;
    }

    .alert-subbox {
        background: #ffffff;
        border: 1px solid var(--line);
        border-radius: 10px;
        padding: 10px;
        margin-top: 4px;
        overflow: hidden;
    }

    .alert-subbox.lift {
        transform: translateY(-4px);
    }

    .alert-caption {
        color: #5f6f8e;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 6px;
    }

    .panel-head {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 8px;
        margin-bottom: 6px;
        flex-wrap: wrap;
    }

    .panel-head .title {
        font-size: 17px;
        font-weight: 900;
        color: #1f2c46;
    }

    .panel-head .delta {
        font-size: 12px;
        color: #5f6f8e;
        font-weight: 700;
    }

    .action-line {
        margin-top: 8px;
        font-size: 15px;
        color: #1f2c46;
        font-weight: 700;
        line-height: 1.4;
    }

    .alert-alarm {
        background: #fff8f1;
        border: 1px solid #f2a64a;
        border-radius: 14px;
        padding: 14px 16px;
        animation: alarmPulse 2.2s infinite, alarmFloat 3s ease-in-out infinite;
    }

    .alarm-title {
        display: inline-block;
        background: #f39a2a;
        color: white;
        font-size: 13px;
        font-weight: 800;
        border-radius: 10px;
        padding: 6px 10px;
        margin-bottom: 10px;
    }

    .alarm-msg {
        font-size: 14px;
        line-height: 1.5;
        color: #5b3a12;
        font-weight: 700;
    }

    .alert-alarm.temp-pulse {
        animation: tempGlow 2.2s ease-in-out infinite;
        border-color: #ef8c24;
    }

    .metric-label-row {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 4px;
    }

    .info-tip {
        position: relative;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        width: 16px;
        height: 16px;
        border-radius: 50%;
        background: #e8f0ff;
        color: #2f63c8;
        font-size: 11px;
        font-weight: 900;
        border: 1px solid #b8caf1;
        cursor: help;
        flex-shrink: 0;
    }

    .info-tip .tooltip-text {
        visibility: hidden;
        opacity: 0;
        position: absolute;
        bottom: 130%;
        left: 50%;
        transform: translateX(-50%);
        width: 190px;
        background: #1f2c46;
        color: #ffffff;
        text-align: center;
        padding: 10px 12px;
        border-radius: 8px;
        font-size: 14px;
        line-height: 1.5;
        font-weight: 500;
        box-shadow: 0 8px 20px rgba(0,0,0,0.18);
        transition: opacity 0.2s ease;
        z-index: 9999;
        white-space: normal;
    }

    .info-tip .tooltip-text::after {
        content: "";
        position: absolute;
        top: 100%;
        left: 50%;
        margin-left: -5px;
        border-width: 5px;
        border-style: solid;
        border-color: #1f2c46 transparent transparent transparent;
    }

    .info-tip:hover .tooltip-text {
        visibility: visible;
        opacity: 1;
    }

    .section-title-box {
        width: 100%;
        box-sizing: border-box;
        border: 1px solid #d7dfef;
        border-radius: 16px;
        padding: 14px 16px;
        margin-bottom: 10px;
        background: rgba(255,255,255,0.78);
        color: #1f2c46;
        font-size: 17px;
        font-weight: 800;
        line-height: 1.2;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.04);
    }

    .alert-subbox.aligned {
        margin-top: 6px;
    }

    @keyframes tempGlow {
        0% { box-shadow: 0 2px 10px rgba(239,140,36,0.18); }
        50% { box-shadow: 0 6px 18px rgba(239,140,36,0.34); }
        100% { box-shadow: 0 2px 10px rgba(239,140,36,0.18); }
    }

    @keyframes alarmPulse {
    0% {
        box-shadow: 0 0 0 0 rgba(243, 154, 42, 0.35);
    }
    50% {
        box-shadow: 0 0 0 10px rgba(243, 154, 42, 0);
    }
    100% {
        box-shadow: 0 0 0 0 rgba(243, 154, 42, 0);
    }
    }

    /* 살짝 떠있는 느낌 */
    @keyframes alarmFloat {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-2px); }
    100% { transform: translateY(0px); }
    }

    /* 경보 카드 기본 스타일 */
    .alert-alarm {
    background: #fff6ef;
    border: 1px solid #f39a2a;
    border-radius: 12px;
    padding: 14px 16px;
    animation: alarmPulse 2.2s infinite, alarmFloat 3s ease-in-out infinite;
    }

    /* 경보 강조 배지 */
    .alarm-title {
    font-weight: 800;
    color: #d46a00;
    margin-bottom: 4px;
    }

    .alarm-msg {
    font-size: 18px;
    color: #2d3b55;
    }    

    .section-divider {
        width: 100%;
        border-top: 2px dashed #9fb2d9;
        margin: 16px 0 14px 0;
        opacity: 0.9;
    }

    .section-divider::after {
        content: "";
        position: absolute;
        left: 50%;
        top: -5px;
        transform: translateX(-50%);
        width: 40px;
        height: 8px;
        background: #f5f7fb;
    }
    
    .v-divider {
        width: 0;
        height: 300px;
        margin: 36px auto 0 auto;
        border-left: 2px dashed #9fb2d9;
        opacity: 0.95;
    }

    .v-divider-mid {
        width: 0;
        height: 420px;
        margin: 40px auto 0 auto;
        border-left: 2px dashed #9fb2d9;
        opacity: 0.95;
    }

    /* Controls 영역 정리 */
    div[data-testid="stSelectbox"] label,
    div[data-testid="stSlider"] label,
    div[data-testid="stRadio"] label {
        font-size: 13px !important;
        font-weight: 700 !important;
        color: #42526e !important;
        margin-bottom: 4px !important;
    }

    /* Select box */
    div[data-testid="stSelectbox"] > div > div {
        border-radius: 10px !important;
        border: 1px solid #d9e2f2 !important;
        background: rgba(255,255,255,0.78) !important;
        min-height: 44px !important;
    }

    /* Slider 주변 여백 */
    div[data-testid="stSlider"] {
        padding-top: 2px !important;
    }

    /* Radio 버튼 간격 */
    div[role="radiogroup"] {
        gap: 18px !important;
    }
    .metric-value.rul-highlight {
        font-size: 22px;
        font-weight: 800;
        color: #0f766e;
    }
    
    div[data-testid="stSelectbox"] > div > div {
    border-radius: 10px !important;
    border: 1px solid #d9e2f2 !important;
    background: rgba(255,255,255,0.78) !important;
    }    
    </style>
    """,
    unsafe_allow_html=True,
)

# =========================
# Utils
# =========================
@st.cache_data
def load_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)


def get_group_col(df: pd.DataFrame) -> str:
    if "uid" in df.columns:
        return "uid"
    if "cycle_index" in df.columns:
        return "cycle_index"
    return ""


def require_cols(df: pd.DataFrame, cols: list[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        st.error(f"[{name}] required columns are missing: {missing}")
        st.stop()


# =========================
# Build df_cycle (cycle summary)
# =========================
@st.cache_data
def build_cycle_table(df_raw: pd.DataFrame) -> pd.DataFrame:
    if df_raw.empty:
        return pd.DataFrame()

    gcol = get_group_col(df_raw)
    if not gcol:
        st.error("discharge_all.csv must contain either 'uid' or 'cycle_index'.")
        st.stop()

    require_cols(
        df_raw,
        ["battery_id", gcol, "Time", "Voltage_measured", "Current_measured", "Temperature_measured"],
        "discharge_all.csv",
    )

    rows = []
    for (bid, uid), g in df_raw.groupby(["battery_id", gcol], sort=False):
        g = g.sort_values("Time")
        t = g["Time"].to_numpy()
        v = g["Voltage_measured"].to_numpy()
        i = g["Current_measured"].to_numpy()
        temp = g["Temperature_measured"].to_numpy()

        if len(t) < 5 or (t[-1] - t[0]) <= 0:
            continue

        energy_wh = trapezoid(np.abs(v * i), t) / 3600.0
        cap_ah = trapezoid(np.abs(i), t) / 3600.0

        t0 = t[0]
        idx_60 = np.where(t <= t0 + 60)[0]
        vdrop_60 = (v[idx_60[0]] - v[idx_60[-1]]) if len(idx_60) >= 2 else np.nan

        rows.append(
            {
                "battery_id": bid,
                "uid": int(uid),
                "duration_s": float(t[-1] - t[0]),
                "energy_wh": float(energy_wh),
                "cap_ah_proxy": float(cap_ah),
                "temp_max": float(np.max(temp)),
                "v_mean": float(np.mean(v)),
                "vdrop_60s": float(vdrop_60) if vdrop_60 == vdrop_60 else np.nan,
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        return df

    df = df.sort_values(["battery_id", "uid"]).reset_index(drop=True)
    df["cycle_count"] = df.groupby("battery_id").cumcount() + 1
    df["soh_pct_actual"] = df.groupby("battery_id")["cap_ah_proxy"].transform(lambda x: (x / x.iloc[0]) * 100.0)
    df["predicted_soh"] = df["soh_pct_actual"] * 0.985
    return df


# =========================
# Attach EIS resistance (R_total = Re_enriched + Rct_enriched)
# =========================
@st.cache_data
def attach_resistance(df_cycle: pd.DataFrame, df_enriched: pd.DataFrame) -> pd.DataFrame:
    if df_cycle.empty:
        return df_cycle

    if df_enriched.empty:
        df_cycle["R_total_ohm"] = np.nan
        return df_cycle

    if "uid" not in df_enriched.columns:
        for cand in ["cycle_idx", "cycle_index", "uid_discharge", "discharge_uid"]:
            if cand in df_enriched.columns:
                df_enriched = df_enriched.rename(columns={cand: "uid"})
                break

    need = ["battery_id", "uid", "Re_enriched", "Rct_enriched"]
    missing = [c for c in need if c not in df_enriched.columns]
    if missing:
        df_cycle["R_total_ohm"] = np.nan
        return df_cycle

    tmp = df_enriched[need].copy()
    tmp = tmp.dropna(subset=["battery_id", "uid"])
    tmp["uid"] = tmp["uid"].astype(int)
    tmp["Re_enriched"] = pd.to_numeric(tmp["Re_enriched"], errors="coerce")
    tmp["Rct_enriched"] = pd.to_numeric(tmp["Rct_enriched"], errors="coerce")
    tmp["R_total_ohm"] = tmp["Re_enriched"] + tmp["Rct_enriched"]
    tmp = tmp.drop_duplicates(subset=["battery_id", "uid"])

    out = df_cycle.merge(tmp[["battery_id", "uid", "R_total_ohm"]], on=["battery_id", "uid"], how="left")
    out["R_total_ohm"] = pd.to_numeric(out["R_total_ohm"], errors="coerce")
    return out


# =========================
# Waveform baseline vs current (normalized time)
# =========================
def to_grid(g: pd.DataFrame, n_grid=250):
    g = g.sort_values("Time")
    t = g["Time"].to_numpy()
    v = g["Voltage_measured"].to_numpy()
    temp = g["Temperature_measured"].to_numpy()
    if len(t) < 3 or (t[-1] - t[0]) <= 0:
        return None
    tn = (t - t[0]) / (t[-1] - t[0])
    grid = np.linspace(0, 1, n_grid)
    return grid, np.interp(grid, tn, v), np.interp(grid, tn, temp)


def baseline_vs_current(df_raw, battery_id, uid_current, baseline_uids, n_grid=250):
    gcol = get_group_col(df_raw)
    sub = df_raw[df_raw["battery_id"] == battery_id].copy()
    if sub.empty:
        return None

    base_v, base_t = [], []
    grid = None
    for u in baseline_uids:
        g = sub[sub[gcol] == u]
        out = to_grid(g, n_grid=n_grid)
        if out is None:
            continue
        grid, v_i, t_i = out
        base_v.append(v_i)
        base_t.append(t_i)

    if not base_v:
        return None

    vb = np.mean(base_v, axis=0)
    tb = np.mean(base_t, axis=0)

    gcur = sub[sub[gcol] == uid_current]
    out = to_grid(gcur, n_grid=n_grid)
    if out is None:
        return None
    grid, vc, tc = out

    deltas = {
        "dV_end": float(vc[-1] - vb[-1]),
        "dT_end": float(tc[-1] - tb[-1]),
    }
    return grid, vb, vc, tb, tc, deltas


def drop_rate(sub: pd.DataFrame, window=10):
    if len(sub) < window + 1:
        return np.nan
    y = sub["soh_pct_actual"].tail(window).to_numpy()
    x = sub["cycle_count"].tail(window).to_numpy()
    return float(np.polyfit(x, y, 1)[0])


def make_alerts(sub: pd.DataFrame, cur: pd.Series):
    counts = {"Temp Spike": 0, "Energy Drop": 0, "Resistance Jump": 0, "Voltage Sag Jump": 0}
    scores = {"Temp Spike": 0.0, "Energy Drop": 0.0, "Resistance Jump": 0.0, "Voltage Sag Jump": 0.0}
    action_map = {
        "Temp Spike": "온도 급상승: thermal test 우선 권장",
        "Energy Drop": "에너지 감소: 출력/부하 조건 재검토",
        "Resistance Jump": "저항 증가: EIS 원인 분석 우선",
        "Voltage Sag Jump": "Voltage sag 증가: 열화 메커니즘 점검",
    }

    temp_excess = max(0.0, float(cur["temp_max"]) - TEMP_LIMIT)
    scores["Temp Spike"] = temp_excess / max(TEMP_LIMIT, 1.0)
    if temp_excess > 0:
        counts["Temp Spike"] += 1

    base_e = float(sub["energy_wh"].iloc[0])
    if base_e > 0:
        energy_drop_ratio = max(0.0, (base_e - float(cur["energy_wh"])) / base_e)
        scores["Energy Drop"] = max(0.0, energy_drop_ratio - 0.10)
        if energy_drop_ratio > 0.10:
            counts["Energy Drop"] += 1

    sub_r = pd.to_numeric(sub["R_total_ohm"], errors="coerce") if "R_total_ohm" in sub.columns else pd.Series(dtype=float)
    cur_r = pd.to_numeric(pd.Series([cur.get("R_total_ohm", np.nan)]), errors="coerce").iloc[0]
    if sub_r.notna().any() and pd.notna(cur_r):
        base_r = sub_r.dropna().iloc[0]
        if base_r > 0:
            r_jump_ratio = (cur_r - base_r) / base_r
            scores["Resistance Jump"] = max(0.0, r_jump_ratio - 0.20)
            if r_jump_ratio > 0.20:
                counts["Resistance Jump"] += 1

    base_vm = float(sub["v_mean"].iloc[0])
    if base_vm > 0:
        sag_ratio = (base_vm - float(cur["v_mean"])) / base_vm
        scores["Voltage Sag Jump"] = max(0.0, sag_ratio - 0.01)
        if sag_ratio > 0.01:
            counts["Voltage Sag Jump"] += 1

    best_metric = max(scores, key=scores.get)
    if scores[best_metric] <= 0:
        best_metric = "Normal"
        primary_action = "이상 징후 없음: 현재 운전 조건 유지 및 모니터링"
    else:
        primary_action = action_map[best_metric]

    return counts, scores, best_metric, primary_action


def mock_anomaly_profile(battery_id: str, cycle: int):
    labels = ["전압 강하", "용량 감소", "온도 증가", "내부저항 증가"]
    seed = sum(ord(c) for c in battery_id) + int(cycle) * 17
    rng = np.random.default_rng(seed)
    vals = rng.integers(1, 10, len(labels))
    profile = {k: int(v) for k, v in zip(labels, vals)}
    top_label = max(profile, key=profile.get)
    return profile, top_label

def metric_card(label: str, value: str, sub: str, foot: str, small=False, help_text=""):
    value_cls = "metric-value small" if small else "metric-value"

    info_html = ""
    if help_text:
        info_html = (
            f'<span class="info-tip">i'
            f'<span class="tooltip-text">{help_text}</span>'
            f'</span>'
        )

    card_html = (
        f'<div class="metric-card">'
        f'  <div class="metric-main">'
        f'    <div class="metric-label-row">'
        f'      <div class="metric-label">{label}</div>'
        f'      {info_html}'
        f'    </div>'
        f'    <div class="{value_cls}">{value}</div>'
        f'    <div class="metric-sub">{sub}</div>'
        f'  </div>'
        f'  <div class="metric-foot">{foot}</div>'
        f'</div>'
    )

    st.markdown(card_html, unsafe_allow_html=True)

def pct_delta(current: float, baseline: float):
    if pd.isna(current) or pd.isna(baseline) or baseline == 0:
        return np.nan
    return ((current - baseline) / abs(baseline)) * 100.0


def colored_pct_text(delta_pct: float):
    if pd.isna(delta_pct):
        return "N/A"
    color = "#de4b68" if delta_pct > 0 else ("#2b6eea" if delta_pct < 0 else "#64748b")
    return f"<span style='color:{color}; font-weight:800;'>{delta_pct:+.2f}%</span>"


def transparent_fig(fig: go.Figure):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


# =========================
# Load data
# =========================
df_raw = load_csv(DISCHARGE_PATH)
df_enriched = load_csv(ENRICHED_PATH)

if df_raw.empty:
    st.error(f"'{DISCHARGE_PATH}' is missing or empty.")
    st.stop()

if "battery_id" in df_raw.columns:
    df_raw = df_raw[df_raw["battery_id"].isin(TARGET_BATTERIES)].copy()
else:
    st.error("discharge_all.csv must contain 'battery_id'.")
    st.stop()

df_cycle = build_cycle_table(df_raw)
df_cycle = attach_resistance(df_cycle, df_enriched)
if "R_total_ohm" in df_cycle.columns:
    df_cycle["R_total_ohm"] = pd.to_numeric(df_cycle["R_total_ohm"], errors="coerce")

if df_cycle.empty:
    st.error("Could not build cycle summary table from discharge data.")
    st.stop()

# =========================
# Header
# =========================
st.markdown(
    '<div class="dash-header"><div>Cell Deep Dive(개별 셀 분석실)</div><div class="brand">SAMSUNG</div></div>',
    unsafe_allow_html=True,
)

# =========================
# Default state
# =========================
if "battery_id" not in st.session_state:
    st.session_state.battery_id = TARGET_BATTERIES[0]

# 현재 선택된 battery 기준 데이터 먼저 준비
battery_id = st.session_state.battery_id
sub = df_cycle[df_cycle["battery_id"] == battery_id].sort_values("cycle_count").reset_index(drop=True)
max_cycle = int(sub["cycle_count"].max())

if "cycle" not in st.session_state:
    st.session_state.cycle = min(5, max_cycle)

# battery 바뀌면 cycle 범위도 다시 맞춰줘야 함
st.session_state.cycle = min(st.session_state.cycle, max_cycle)
cycle = st.session_state.cycle

if "view_mode" not in st.session_state:
    st.session_state.view_mode = "Trend"
view_mode = st.session_state.view_mode

# 현재 사이클 행
cur = sub[sub["cycle_count"] == cycle].iloc[0]

# Metrics
win = min(20, len(sub))
tail = sub.tail(win)
slope = np.polyfit(tail["cycle_count"], tail["soh_pct_actual"], 1)[0] if win >= 5 else -0.1
rul = np.nan if slope >= 0 else max(0.0, (cur["soh_pct_actual"] - EOL_THRESHOLD) / (-slope))
mae = float(np.mean(np.abs(sub["soh_pct_actual"] - sub["predicted_soh"])))
r_total = pd.to_numeric(cur.get("R_total_ohm", np.nan), errors="coerce")
drop = drop_rate(sub, window=10)
prev_soh_delta = np.nan
if cycle > 1:
    prev_row = sub[sub["cycle_count"] == cycle - 1]
    if not prev_row.empty:
        prev_soh_delta = float(cur["soh_pct_actual"] - prev_row.iloc[0]["soh_pct_actual"])

pred_gap = cur["predicted_soh"] - cur["soh_pct_actual"]
base_temp = sub[sub["cycle_count"].between(BASELINE_CYCLES[0], BASELINE_CYCLES[1])]["temp_max"].mean()
base_r = sub["R_total_ohm"].dropna().iloc[0] if "R_total_ohm" in sub.columns and sub["R_total_ohm"].notna().any() else np.nan
r_delta = np.nan if pd.isna(r_total) or pd.isna(base_r) else r_total - base_r
prev_r = np.nan
if cycle > 1 and "R_total_ohm" in sub.columns:
    prev_r_row = sub[sub["cycle_count"] == cycle - 1]["R_total_ohm"]
    if not prev_r_row.empty:
        prev_r = pd.to_numeric(prev_r_row.iloc[0], errors="coerce")
dr_prev = np.nan if pd.isna(r_total) or pd.isna(prev_r) else (r_total - prev_r)
dr_prev_pct = pct_delta(r_total, prev_r)
dr_prev_text = "N/A" if pd.isna(dr_prev) else f"{dr_prev:+.4f} Ω"

st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)
m1, m2, m3, m4, m5 = st.columns(5, gap="small")

with m1:
    foot_text = "전 사이클 없음" if pd.isna(prev_soh_delta) else f"전 사이클 대비 {prev_soh_delta:+.2f}%p"
    metric_card(
        "SOH",
        f"{cur['soh_pct_actual']:.1f}%",
        "",
        foot_text,
        help_text="배터리 현재 건강 상태"
    )

with m2:
    metric_card(
        "현재 SOH / 예측 SOH",
        f"{cur['soh_pct_actual']:.1f}% / {cur['predicted_soh']:.1f}%",
        f"MAE ±{mae:.2f}%",
        f"예측 편차 {pred_gap:+.2f}%",
        small=True,
        help_text="현재 성능과 예측 성능 비교"
    )

with m3:
    metric_card(
        "남은 수명 (RUL)",
        f"{rul:.0f} cycles",
        f"EOL {EOL_THRESHOLD:.0f}% 기준",
        f"최근 slope {slope:+.3f}",
        small=True,
        help_text="검증 완료까지 남은 예상 수명"
    )

with m4:
    t_delta = np.nan if pd.isna(base_temp) else cur["temp_max"] - base_temp
    metric_card(
        "온도 (°C)",
        f"{cur['temp_max']:.1f}°C",
        f"Baseline {base_temp:.1f}°C",
        f"전사이클 대비 {t_delta:+.2f}°C",
        help_text="배터리 작동 중 최대 온도.<br>높아질수록 열화 의심"
    )

with m5:
    metric_card(
        "내부 저항 (EIS Resistance)",
        "-" if pd.isna(r_total) else f"{r_total:.3f} Ω",
        "R_total",
        f"전사이클 대비 {dr_prev_text}",
        small=True,
        help_text="열화 진행 시 증가하는 저항.<br>증가하면 성능 저하 신호"
    )

st.markdown("<div style='height:10px;'></div>", unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
# =========================
# Controls
# =========================
ctl1, ctl2, ctl3 = st.columns([0.8, 1.8, 0.8])

with ctl1:
    st.selectbox(
        "Battery",
        TARGET_BATTERIES,
        key="battery_id",
    )

# battery 변경 반영
battery_id = st.session_state.battery_id
sub = df_cycle[df_cycle["battery_id"] == battery_id].sort_values("cycle_count").reset_index(drop=True)
max_cycle = int(sub["cycle_count"].max())

# cycle 범위 보정
if "cycle" not in st.session_state:
    st.session_state.cycle = min(5, max_cycle)
st.session_state.cycle = min(st.session_state.cycle, max_cycle)

with ctl2:
    st.slider(
        "Cycle Step",
        1,
        max_cycle,
        key="cycle",
    )

with ctl3:
    st.radio(
        "View",
        ["Trend", "Waveform"],
        horizontal=True,
        key="view_mode",
    )

# 최종 상태값 다시 읽기
battery_id = st.session_state.battery_id
cycle = st.session_state.cycle
view_mode = st.session_state.view_mode

sub = df_cycle[df_cycle["battery_id"] == battery_id].sort_values("cycle_count").reset_index(drop=True)
cur = sub[sub["cycle_count"] == cycle].iloc[0]

# =========================
# Middle row: SOH + Alerts
# =========================
baseline_uids = sub[sub["cycle_count"].between(BASELINE_CYCLES[0], BASELINE_CYCLES[1])]["uid"].astype(int).tolist()
wave = baseline_vs_current(df_raw, battery_id, int(cur["uid"]), baseline_uids, n_grid=WAVE_N_GRID) if view_mode == "Waveform" else None

left_mid, div_mid, right_mid = st.columns([2.2, 0.03, 1.0], gap="medium")

with left_mid:
    st.markdown('<div class="panel"><div class="panel-title">SOH Monitoring & Prediction</div>', unsafe_allow_html=True)
    if view_mode == "Trend":
        sub_view = sub[sub["cycle_count"] <= cycle].copy()
        
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=sub_view["cycle_count"],
                y=sub_view["soh_pct_actual"],
                mode="lines",
                name="Actual SOH",
                line=dict(width=3, color="#2b6eea"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sub_view["cycle_count"],
                y=sub_view["predicted_soh"],
                mode="lines",
                name="Predicted SOH",
                line=dict(width=2, dash="dash", color="#8ea5d3"),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sub_view["cycle_count"],
                y=sub_view["predicted_soh"] + mae,
                mode="lines",
                line=dict(width=0),
                showlegend=False,
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=sub_view["cycle_count"],
                y=sub_view["predicted_soh"] - mae,
                mode="lines",
                line=dict(width=0),
                fill="tonexty",
                fillcolor="rgba(43,110,234,0.10)",
                showlegend=False,
                hoverinfo="skip",
            )
        )

        fig.add_hline(y=EOL_THRESHOLD, line_dash="dot", line_color="#e1627d", annotation_text="EOL Threshold")
        fig.add_vline(x=cycle, line_dash="dot", line_color="#66748f")
        fig.update_layout(
            height=410,
            margin=dict(l=8, r=8, t=8, b=8),
            xaxis_title="Cycle Count",
            yaxis_title="SOH (%)",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        transparent_fig(fig)
        st.plotly_chart(fig, use_container_width=True)
    else:
        if wave is None:
            st.info("Waveform 비교를 위한 baseline/current 파형이 부족합니다.")
        else:
            t_norm, vb, vc, tb, tc, deltas = wave
            w1, w2 = st.columns(2, gap="small")
            with w1:
                fig_wv = go.Figure()
                fig_wv.add_trace(go.Scatter(x=t_norm, y=vb, name="Baseline V", line=dict(dash="dash", color="#8ea5d3")))
                fig_wv.add_trace(go.Scatter(x=t_norm, y=vc, name=f"Current V({cycle})", line=dict(color="#2b6eea", width=3)))
                fig_wv.update_layout(
                    height=410,
                    margin=dict(l=8, r=8, t=8, b=8),
                    xaxis_title="Normalized time",
                    yaxis_title="Voltage (V)",
                    legend=dict(orientation="h", y=1.02, x=0),
                )
                transparent_fig(fig_wv)
                st.plotly_chart(fig_wv, use_container_width=True)
            with w2:
                fig_wt = go.Figure()
                fig_wt.add_trace(go.Scatter(x=t_norm, y=tb, name="Baseline T", line=dict(dash="dash", color="#b9c5de")))
                fig_wt.add_trace(go.Scatter(x=t_norm, y=tc, name=f"Current T({cycle})", line=dict(color="#f59e0b", width=3)))
                fig_wt.update_layout(
                    height=410,
                    margin=dict(l=8, r=8, t=8, b=8),
                    xaxis_title="Normalized time",
                    yaxis_title="Temp (°C)",
                    legend=dict(orientation="h", y=1.02, x=0),
                )
                transparent_fig(fig_wt)
                st.plotly_chart(fig_wt, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

with div_mid:
    st.markdown('<div class="v-divider-mid"></div>', unsafe_allow_html=True)
    
with right_mid:
    profile, top_label = mock_anomaly_profile(battery_id, cycle)
    action_map_demo = {
        "전압 강하": "부하 조건 및 전압 강하 구간 점검",
        "용량 감소": "부하 조건 및 배터리 열화 상태 확인",
        "온도 증가": "내부저항 및 열 관리 상태 점검",
        "내부저항 증가": "EIS 진단 및 셀 열화 상태 점검",
    }

    # 발생 건수 기준 내림차순 정렬
    sorted_items = sorted(profile.items(), key=lambda x: x[1], reverse=True)
    labels_sorted = [k for k, _ in sorted_items]
    values_sorted = [v for _, v in sorted_items]

    top_label = labels_sorted[0]
    top_value = values_sorted[0]

    bar_colors = ["#f39a2a"] + ["#7b9be3"] * (len(labels_sorted) - 1)
    bar_line_colors = ["#d97f18"] + ["#7b9be3"] * (len(labels_sorted) - 1)
    bar_line_widths = [2] + [0] * (len(labels_sorted) - 1)

    ymax = max(5, int(max(profile.values())) + 1)

    fig_a = go.Figure(
        data=[
            go.Bar(
                x=labels_sorted,
                y=values_sorted,
                marker=dict(
                    color=bar_colors,
                    line=dict(color=bar_line_colors, width=bar_line_widths),
                ),
                text=[f"{v}건" for v in values_sorted],
                textposition="inside",
                textfont=dict(color="white", size=14),
                insidetextanchor="middle",
                hovertemplate="%{x}: %{y}건<extra></extra>",
            )
        ]
    )

    fig_a.add_annotation(
        x=top_label,
        y=top_value + 0.25,
        text="TOP1",
        showarrow=False,
        font=dict(color="#f39a2a", size=12),
    )

    fig_a.update_layout(
        height=235,   # 기존 260보다 조금 더 줄임
        margin=dict(l=8, r=8, t=8, b=0),
        yaxis_title="발생 건수",
        xaxis_title="",
        yaxis=dict(
            range=[0, ymax],
            tick0=0,
            dtick=1,
            gridcolor="#e9eef8",
        ),
    )
    transparent_fig(fig_a)
    
    st.markdown(
        '<div class="section-title-box">이상감지 발생 원인</div>',
        unsafe_allow_html=True
    )

    st.plotly_chart(fig_a, use_container_width=True)
    
    alarm_cls = "alert-alarm temp-pulse" if top_label == "온도 증가" else "alert-alarm"
    st.markdown(
        f"""
        <div class="alert-subbox aligned">
          <div class="{alarm_cls}">
            <div class="alarm-title">TOP 1 경보: {top_label} ({top_value}건)</div>
            <div class="alarm-msg">[{top_label}] {action_map_demo[top_label]}</div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)
st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
# =========================
# Bottom row: three charts
# =========================
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

b1, div1, b2, div2, b3 = st.columns([1, 0.03, 1, 0.03, 1])

baseline_mask = sub["cycle_count"].between(BASELINE_CYCLES[0], BASELINE_CYCLES[1])
base_v = sub.loc[baseline_mask, "v_mean"].mean()
base_t = sub.loc[baseline_mask, "temp_max"].mean()
sub_view = sub[sub["cycle_count"] <= cycle].copy()

with b1:
    dv = cur["v_mean"] - base_v
    dv_pct = pct_delta(cur["v_mean"], base_v)
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-head">
            <div class="title">전압 (Voltage) vs Cycle</div>
            <div class="delta">ΔV {dv:+.3f} V | {colored_pct_text(dv_pct)}</div>
          </div>
        """,
        unsafe_allow_html=True,
    )
    fig_v = go.Figure()
    fig_v.add_trace(
        go.Scatter(
            x=sub_view["cycle_count"],
            y=sub_view["v_mean"],
            mode="lines",
            name="Voltage",
            line=dict(color="#2b6eea", width=3),
        )
    )
    fig_v.add_hline(y=base_v, line_dash="dash", line_color="#9aa9c7", annotation_text="Baseline")
    fig_v.add_vline(x=cycle, line_dash="dot", line_color="#66748f")
    fig_v.update_layout(height=255, margin=dict(l=8, r=8, t=8, b=8), xaxis_title="Cycle", yaxis_title="V")
    transparent_fig(fig_v)
    st.plotly_chart(fig_v, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
    
with div1:
    st.markdown('<div class="v-divider"></div>', unsafe_allow_html=True)
    
with b2:
    dt = cur["temp_max"] - base_t
    dt_pct = pct_delta(cur["temp_max"], base_t)
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-head">
            <div class="title">최대 온도 (Max Temp) vs Cycle</div>
            <div class="delta">ΔT {dt:+.2f} °C | {colored_pct_text(dt_pct)}</div>
          </div>
        """,
        unsafe_allow_html=True,
    )
    fig_t = go.Figure()
    fig_t.add_trace(
        go.Scatter(
            x=sub_view["cycle_count"],
            y=sub_view["temp_max"],
            mode="lines",
            name="Temp Max",
            line=dict(color="#f59e0b", width=3),
        )
    )
    fig_t.add_hline(y=base_t, line_dash="dash", line_color="#9aa9c7", annotation_text="Baseline")
    fig_t.add_hline(y=TEMP_LIMIT, line_dash="dot", line_color="#de4b68", annotation_text="Limit")
    fig_t.add_vline(x=cycle, line_dash="dot", line_color="#66748f")
    fig_t.update_layout(height=255, margin=dict(l=8, r=8, t=8, b=8), xaxis_title="Cycle", yaxis_title="°C")
    transparent_fig(fig_t)
    st.plotly_chart(fig_t, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

with div2:
    st.markdown('<div class="v-divider"></div>', unsafe_allow_html=True)
    
with b3:
    dr_pct = pct_delta(r_total, base_r)
    dr_text = "N/A" if pd.isna(r_delta) else f"{r_delta:+.4f} Ω"
    st.markdown(
        f"""
        <div class="panel">
          <div class="panel-head">
            <div class="title">내부저항 vs Cycle</div>
            <div class="delta"> ΔR {dr_text} | {colored_pct_text(dr_pct)}
          </div>
        """,
        unsafe_allow_html=True,
    )
    if "R_total_ohm" not in sub.columns or sub["R_total_ohm"].isna().all():
        st.warning("R_total data is unavailable in enriched table.")
    else:
        fig_r = go.Figure()
        fig_r.add_trace(
            go.Scatter(
                x=sub_view["cycle_count"],
                y=sub_view["R_total_ohm"],
                mode="lines",
                name="R_total",
                line=dict(color="#c27918", width=3),
            )
        )
        fig_r.add_hline(y=base_r, line_dash="dash", line_color="#9aa9c7", annotation_text="Baseline")
        fig_r.add_vline(x=cycle, line_dash="dot", line_color="#66748f")
        fig_r.update_layout(height=255, margin=dict(l=8, r=8, t=8, b=8), xaxis_title="Cycle", yaxis_title="Ω")
        transparent_fig(fig_r)
        st.plotly_chart(fig_r, use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)
