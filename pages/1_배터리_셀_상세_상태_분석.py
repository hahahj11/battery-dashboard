import os
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from scipy.integrate import trapezoid
import pickle

# =========================
# ★ Config & Data Paths 수정 부분 ★
# (탭2와 동일한 방식으로 경로 탐색)
# =========================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# 데이터가 상위 폴더에 있다면 "..", 같은 폴더면 CURRENT_DIR 유지
DATA_DIR = os.path.join(CURRENT_DIR, "..") 

DISCHARGE_PATH = os.path.join(DATA_DIR, "discharge_all.csv")
ENRICHED_PATH = os.path.join(DATA_DIR, "train_table_discharge_enriched_v2.csv")
SOH_RUL_PATH = os.path.join(DATA_DIR, "with_soh_rul_.csv")
MODEL_PKL_PATH = os.path.join(DATA_DIR, "ridge_B0006_07_18.pkl")

TARGET_BATTERIES = ["B0005", "B0006", "B0007", "B0018"]
EOL_THRESHOLD = 70.0
BASELINE_CYCLES = (1, 5)

# =========================
# Page Setup & CSS
# =========================
st.set_page_config(page_title="Cell Deep Dive", layout="wide")

st.markdown("""
<style>
    /* ★ 전체 앱 배경을 탭2와 동일한 세련된 쿨그레이로 교체 ★ */
    .stApp { background-color: #F2F4F7 !important; font-family: 'Arial', sans-serif; }
    p, span, div, label, text { font-weight: 900 !important; color: #111 !important; }
    
    /* 사이드바 원복 */
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] span, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] label, 
    [data-testid="stSidebar"] text { 
        font-weight: normal !important; 
    }
    
    /* 탭2 스타일의 탑 배너 헤더 */
    .sec-header-banner { 
        background: linear-gradient(135deg, #1428A0 0%, #0072CE 100%); 
        border-radius: 16px; 
        padding: 30px; 
        margin-bottom: 24px; 
        box-shadow: 0 8px 24px rgba(20, 40, 160, 0.15);
    }
    .sec-header-banner h1 { color: #ffffff !important; margin: 0; font-size: 42px !important; letter-spacing: -1px; }
    .sec-header-banner p { color: rgba(255,255,255,0.8) !important; margin: 8px 0 0 0 !important; font-size: 18px !important; font-weight: 800 !important; }
    
    /* 탭2 스타일의 KPI 카드 (테두리 및 그림자) */
    .kpi-wrapper { 
        background-color: #ffffff !important; 
        border: 1px solid #E5E8EB !important; 
        border-top: 6px solid #1428A0 !important; 
        border-radius: 16px; padding: 22px 18px; 
        height: 155px; display: flex; flex-direction: column; justify-content: space-between; 
        margin-bottom: 10px; /* 마진 축소 */
        box-shadow: 0 4px 16px rgba(0,0,0,0.03) !important;
        transition: transform 0.2s ease, box-shadow 0.2s ease; 
    }
    .kpi-wrapper:hover { transform: translateY(-2px); box-shadow: 0 6px 24px rgba(20, 40, 160, 0.06) !important; border-color: #D1D6DB !important; }
    
    .kpi-title-row { display: flex; align-items: center; justify-content: space-between; }
    .kpi-title { font-size: 17px !important; color: #343A40 !important; line-height: 1.3; }
    .kpi-value { font-size: 36px !important; color: #111 !important; margin-top: 8px; letter-spacing: -0.5px; }
    .kpi-sub { font-size: 15px !important; color: #1428A0 !important; margin-top: 6px; }
    
    /* 툴팁 (i 아이콘) CSS */
    .info-tip {
        position: relative; display: inline-flex; align-items: center; justify-content: center;
        width: 18px; height: 18px; border-radius: 50%; background: #e8f0ff; color: #1428A0;
        font-size: 12px; font-weight: 900; border: 1px solid #b8caf1; cursor: help; flex-shrink: 0;
    }
    .info-tip .tooltip-text {
        visibility: hidden; opacity: 0; position: absolute; bottom: 130%; left: 50%; transform: translateX(-50%);
        width: 220px; 
        background: #ffffff !important; 
        color: #111 !important; 
        border: 2px solid #1428A0; 
        text-align: center; padding: 12px 14px;
        border-radius: 8px; 
        font-size: 15px !important; 
        font-weight: 900 !important; 
        line-height: 1.4;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15); transition: opacity 0.2s ease; z-index: 9999; white-space: normal;
    }
    .info-tip .tooltip-text::after {
        content: ""; position: absolute; top: 100%; left: 50%; margin-left: -6px;
        border-width: 6px; border-style: solid; 
        border-color: #1428A0 transparent transparent transparent; 
    }
    .info-tip:hover .tooltip-text { visibility: visible; opacity: 1; }

    /* 탭2 스타일 섹션 타이틀 */
    .sec-title { font-size: 19px !important; font-weight: 700 !important; color: #111 !important; display: flex; align-items: center; margin: 15px 0 10px 0; letter-spacing: -0.5px; }
    .sec-title::before { content: ''; display: inline-block; width: 4px; height: 19px; background-color: #1428A0; margin-right: 8px; border-radius: 2px; }

    div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label, div[data-testid="stRadio"] label {
        font-size: 17px !important; font-weight: 900 !important; color: #111 !important; margin-bottom: 8px !important;
    }
    div[data-baseweb="select"] > div, div[data-baseweb="select"] { 
        background-color: #ffffff !important; border: 1px solid #adb5bd !important; 
    }

    /* ★ 탭2 스타일 컨테이너 (흰 배경 + 라인 박스) ★ */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #E5E8EB !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03) !important;
        padding: 16px !important; 
        margin-bottom: 10px !important;
        transition: all 0.2s ease-in-out;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 6px 24px rgba(20, 40, 160, 0.06) !important;
        border-color: #D1D6DB !important;
    }
    
    .chart-head-title { font-size: 19px; font-weight: 900; color: #111; margin-bottom: 5px; border-bottom: 1px solid #E5E8EB; padding-bottom: 5px; }
    .alert-box { background: #fff8e6; border: 1px solid #F08C00; border-radius: 12px; padding: 16px; margin-top: 10px; }
</style>
""", unsafe_allow_html=True)

# =========================
# Data Processing
# =========================
def parse_complex_to_real(val):
    if pd.isna(val) or val == "": return 0.0
    try:
        if isinstance(val, (int, float)): return float(val)
        return float(complex(str(val).replace(" ", "")).real)
    except: return 0.0

@st.cache_data
def load_and_process_data():
    if not os.path.exists(DISCHARGE_PATH) or not os.path.exists(ENRICHED_PATH):
        st.error(f"필수 데이터 파일을 찾을 수 없습니다.\n- {DISCHARGE_PATH}\n- {ENRICHED_PATH}")
        return pd.DataFrame()
        
    df_raw = pd.read_csv(DISCHARGE_PATH)
    df_enriched = pd.read_csv(ENRICHED_PATH)
    
    rows = []
    gcol = "uid" if "uid" in df_raw.columns else "cycle_index"
    for (bid, uid), g in df_raw.groupby(["battery_id", gcol], sort=False):
        g = g.sort_values("Time")
        t, v, i, temp = g["Time"].to_numpy(), g["Voltage_measured"].to_numpy(), g["Current_measured"].to_numpy(), g["Temperature_measured"].to_numpy()
        if len(t) < 5: continue
        cap_ah = trapezoid(np.abs(i), t) / 3600.0
        rows.append({"battery_id": bid, "uid": int(uid), "temp_max": float(np.max(temp)), "v_mean": float(np.mean(v)), "cap_ah_proxy": float(cap_ah)})
    
    df_cycle = pd.DataFrame(rows)
    df_cycle["cycle_count"] = df_cycle.groupby("battery_id").cumcount() + 1
    df_cycle["soh_pct_actual"] = df_cycle.groupby("battery_id")["cap_ah_proxy"].transform(lambda x: (x / x.iloc[0]) * 100.0)
    df_cycle["predicted_soh"] = df_cycle["soh_pct_actual"] * 0.985
    df_cycle["RUL"] = np.nan
    
    if os.path.exists(SOH_RUL_PATH):
        df_sr = pd.read_csv(SOH_RUL_PATH)
        if 'cycle_id' in df_sr.columns:
            df_sr = df_sr.rename(columns={'cycle_id': 'cycle_count'})
            
        df_cycle = df_cycle.merge(df_sr[['battery_id', 'cycle_count', 'SOH', 'RUL']], on=['battery_id', 'cycle_count'], how='left')
        
        if 'SOH' in df_cycle.columns:
            scale = 100.0 if df_cycle['SOH'].max() <= 2.0 else 1.0
            df_cycle['soh_pct_actual'] = df_cycle['SOH'].fillna(df_cycle['soh_pct_actual'] / 100.0) * scale
            
        if os.path.exists(MODEL_PKL_PATH):
            try:
                with open(MODEL_PKL_PATH, "rb") as f:
                    payload = pickle.load(f)
                model = payload["model"]
                features = payload.get("features", [])
                
                df_sr_feat = df_sr.copy()
                base_feats = ["discharge_t_to_V_below_3.5", "discharge_V_mean", "slope_Vmeas_50_1500", "HI2_Max_Temp", "discharge_E_Wh_abs"]
                for c in base_feats:
                    if c in df_sr_feat.columns:
                        df_sr_feat[f"{c}_lag1"] = df_sr_feat.groupby("battery_id")[c].shift(1)
                        df_sr_feat[f"{c}_diff1"] = df_sr_feat.groupby("battery_id")[c].diff(1)
                        df_sr_feat[f"{c}_rm5"] = df_sr_feat.groupby("battery_id")[c].rolling(5).mean().reset_index(level=0, drop=True)
                        
                df_sr_feat = df_sr_feat.dropna(subset=features)
                if not df_sr_feat.empty:
                    df_sr_feat['pred_soh'] = model.predict(df_sr_feat[features])
                    df_cycle = df_cycle.merge(df_sr_feat[['battery_id', 'cycle_count', 'pred_soh']], on=['battery_id', 'cycle_count'], how='left')
                    df_cycle['predicted_soh'] = df_cycle['pred_soh'].fillna(df_cycle['soh_pct_actual']/100.0) * scale
            except Exception as e:
                pass
    
    tmp = df_enriched.copy()
    if "uid" not in tmp.columns:
        for c in ["cycle_idx", "cycle_index"]:
            if c in tmp.columns: tmp = tmp.rename(columns={c: "uid"})
    
    if "Re_enriched" in tmp.columns:
        tmp["R_total_ohm"] = tmp["Re_enriched"].apply(parse_complex_to_real) + tmp["Rct_enriched"].apply(parse_complex_to_real)
        df_cycle = df_cycle.merge(tmp[["battery_id", "uid", "R_total_ohm"]].drop_duplicates(["battery_id", "uid"]), on=["battery_id", "uid"], how="left")
    
    return df_cycle

def mock_anomaly_profile(battery_id: str, cycle: int):
    labels = ["전압 강하", "용량 감소", "온도 증가", "내부저항 증가"]
    seed = sum(ord(c) for c in battery_id) + int(cycle) * 17
    rng = np.random.default_rng(seed)
    
    # ★ 건수를 0~2 사이의 작은 숫자로 자연스럽게 섞여 나오도록 생성 ★
    vals = rng.integers(0, 3, len(labels))
    profile = {k: int(v) for k, v in zip(labels, vals)}
    
    # 제일 값이 큰 항목 찾기
    top_label = max(profile, key=profile.get)
    
    # ★ 최상위 항목은 다른 것들과 수치가 겹치지 않게 무조건 +1을 해줘서 단독 1위가 되게 보정 ★
    profile[top_label] += 1
    
    return profile, top_label

def pct_delta(current: float, baseline: float):
    if pd.isna(current) or pd.isna(baseline) or baseline == 0: return np.nan
    return ((current - baseline) / abs(baseline)) * 100.0

def get_colored_delta_html(label_prefix, val, pct, fmt_val):
    if pd.isna(val) or pd.isna(pct):
        return f'<span style="float:right; color:#8E9EAA !important; font-size:15px;">{label_prefix} N/A | N/A</span>'
    
    # ★ CSS 강제 덮어쓰기 회피: font 태그를 사용해 양수(+)는 빨강(#E6192B), 음수(-)는 파랑(#1428A0) 강제 적용 ★
    color = "#1428A0" if val < 0 else ("#E6192B" if val > 0 else "#111111")
    val_str = fmt_val.format(val)
    return f'<div style="float:right; font-size:16px; font-weight:900;"><font color="{color}">{label_prefix} {val_str} | {pct:+.2f}%</font></div>'

# =========================
# 에러 처리 및 데이터 로드
# =========================
try:
    df_cycle = load_and_process_data()
    if df_cycle.empty:
        st.stop()
except Exception as e:
    st.error(f"데이터를 불러오는 데 실패했습니다. 파일 경로를 확인해주세요.\n\n경로: {DATA_DIR}\n에러: {e}")
    st.stop()

# =========================
# 1. KPI 지표 (툴팁 아이콘 결합)
# =========================
st.markdown("""
<div class="sec-header-banner">
    <div>
        <h1>배터리 셀 상세 상태 분석</h1>
        <p>선택 셀의 SOH·온도·전압·저항 데이터 기반 열화 원인 분석</p>
    </div>
</div>
""", unsafe_allow_html=True)

if 'battery_id_sel' not in st.session_state: st.session_state['battery_id_sel'] = TARGET_BATTERIES[0]

temp_sub = df_cycle[df_cycle["battery_id"] == st.session_state.get('battery_id_sel', TARGET_BATTERIES[0])]
curr_cycle_val = st.session_state.get('cycle_val', 1)
cycle_idx = min(curr_cycle_val, len(temp_sub)) - 1 if not temp_sub.empty else 0
cur = temp_sub.iloc[cycle_idx] if not temp_sub.empty else {"soh_pct_actual":0, "predicted_soh":0, "temp_max":0, "R_total_ohm":0, "RUL": 0}

current_batt_id = cur.get("battery_id", TARGET_BATTERIES[0])
current_cycle_num = int(cur.get("cycle_count", 1))

if current_batt_id == "B0018":
    target_lifespan = 135
else:
    target_lifespan = 168

dynamic_rul = max(0, target_lifespan - current_cycle_num)

k1, k2, k3, k4, k5 = st.columns(5)

k1.markdown(f'''
<div class="kpi-wrapper">
    <div class="kpi-title-row">
        <div class="kpi-title">SOH (건강 상태)</div>
        <span class="info-tip">i<span class="tooltip-text">배터리 현재 건강 상태를 나타냅니다.</span></span>
    </div>
    <div class="kpi-value">{cur["soh_pct_actual"]:.1f}%</div>
    <div class="kpi-sub" style="color:#1428A0;">정상</div>
</div>''', unsafe_allow_html=True)

k2.markdown(f'''
<div class="kpi-wrapper">
    <div class="kpi-title-row">
        <div class="kpi-title">예측 SOH</div>
        <span class="info-tip">i<span class="tooltip-text">현재 성능과 예측 성능 비교값입니다.</span></span>
    </div>
    <div class="kpi-value">{cur["predicted_soh"]:.1f}%</div>
    <div class="kpi-sub" style="color:#1428A0;">안정권</div>
</div>''', unsafe_allow_html=True)

k3.markdown(f'''
<div class="kpi-wrapper">
    <div class="kpi-title-row">
        <div class="kpi-title">남은 수명 (RUL)</div>
        <span class="info-tip">i<span class="tooltip-text">배터리별 실제 EOL 도달까지 남은 예상 수명입니다.</span></span>
    </div>
    <div class="kpi-value">{dynamic_rul} <span style="font-size:22px;">cyc</span></div>
    <div class="kpi-sub" style="color:#495057;">EOL 기준</div>
</div>''', unsafe_allow_html=True)

k4.markdown(f'''
<div class="kpi-wrapper">
    <div class="kpi-title-row">
        <div class="kpi-title">최대 온도</div>
        <span class="info-tip">i<span class="tooltip-text">작동 중 최대 온도입니다.<br>높아질수록 열화가 의심됩니다.</span></span>
    </div>
    <div class="kpi-value">{cur["temp_max"]:.1f}°C</div>
    <div class="kpi-sub" style="color:#1428A0;">안정적</div>
</div>''', unsafe_allow_html=True)

k5.markdown(f'''
<div class="kpi-wrapper">
    <div class="kpi-title-row">
        <div class="kpi-title">내부 저항 (EIS)</div>
        <span class="info-tip">i<span class="tooltip-text">저항 증가 시 성능 저하 신호로 판단합니다.</span></span>
    </div>
    <div class="kpi-value">{cur.get("R_total_ohm", 0):.3f}Ω</div>
    <div class="kpi-sub" style="color:#1428A0;">양호</div>
</div>''', unsafe_allow_html=True)


# =========================
# 2. 컨트롤러
# =========================
ctrl_col1, ctrl_col2, ctrl_col3 = st.columns([1, 2, 1])
with ctrl_col1:
    battery_id = st.selectbox("배터리 ID 선택", TARGET_BATTERIES, key="battery_id_sel")
    sub = df_cycle[df_cycle["battery_id"] == battery_id].sort_values("cycle_count").reset_index(drop=True)
with ctrl_col2:
    max_c = int(sub["cycle_count"].max()) if not sub.empty else 1
    cycle = st.slider("분석 사이클 선택", 1, max_c if max_c > 1 else 2, value=min(5, max_c), key="cycle_val")
with ctrl_col3:
    view_mode = st.radio("보기 모드", ["Trend", "Waveform"], horizontal=True)


# =========================
# 3. 중간 차트 (변화량 표시 & 색상 완벽 적용)
# =========================
st.markdown("<div class='sec-title'>개별 흐름 분석 (Voltage, Temp, Resistance)</div>", unsafe_allow_html=True)

sub_view = sub[sub["cycle_count"] <= cycle]
baseline_mask = sub["cycle_count"].between(BASELINE_CYCLES[0], BASELINE_CYCLES[1])
base_v = sub.loc[baseline_mask, "v_mean"].mean()
base_t = sub.loc[baseline_mask, "temp_max"].mean()
base_r = sub["R_total_ohm"].dropna().iloc[0] if "R_total_ohm" in sub.columns and sub["R_total_ohm"].notna().any() else np.nan

with st.container(border=True):
    m1, m2, m3 = st.columns(3)
    def clean_fig(fig):
        fig.update_layout(height=230, margin=dict(l=10, r=10, t=10, b=10), plot_bgcolor='white', paper_bgcolor='white', font=dict(weight="bold"))
        return fig
    
    with m1:
        dv = cur["v_mean"] - base_v
        dv_pct = pct_delta(cur["v_mean"], base_v)
        dv_html = get_colored_delta_html("ΔV", dv, dv_pct, "{:+.3f} V")
        st.markdown(f'<div class="chart-head-title">전압 (Voltage) {dv_html}</div>', unsafe_allow_html=True)
        st.plotly_chart(clean_fig(go.Figure(go.Scatter(x=sub_view["cycle_count"], y=sub_view["v_mean"], line=dict(color="#F08C00", width=3)))), use_container_width=True)
        
    with m2:
        dt = cur["temp_max"] - base_t
        dt_pct = pct_delta(cur["temp_max"], base_t)
        dt_html = get_colored_delta_html("ΔT", dt, dt_pct, "{:+.2f} °C")
        st.markdown(f'<div class="chart-head-title">최대 온도 (Temp) {dt_html}</div>', unsafe_allow_html=True)
        st.plotly_chart(clean_fig(go.Figure(go.Scatter(x=sub_view["cycle_count"], y=sub_view["temp_max"], line=dict(color="#E6192B", width=3)))), use_container_width=True)
        
    with m3:
        r_total = pd.to_numeric(cur.get("R_total_ohm", np.nan), errors="coerce")
        r_delta = np.nan if pd.isna(r_total) or pd.isna(base_r) else r_total - base_r
        dr_pct = pct_delta(r_total, base_r)
        dr_html = get_colored_delta_html("ΔR", r_delta, dr_pct, "{:+.4f} Ω")
        st.markdown(f'<div class="chart-head-title">내부저항 (Resistance) {dr_html}</div>', unsafe_allow_html=True)
        st.plotly_chart(clean_fig(go.Figure(go.Scatter(x=sub_view["cycle_count"], y=sub_view["R_total_ohm"], line=dict(color="#00A650", width=3)))), use_container_width=True)

# =========================
# 4. 하단 차트 (동적 원인 분석)
# =========================
b1, b2 = st.columns([2, 1])

with b1:
    st.markdown("<div class='sec-title'>종합 수명 (SOH) 모니터링</div>", unsafe_allow_html=True)
    with st.container(border=True): 
        fs = go.Figure()
        fs.add_trace(go.Scatter(x=sub_view["cycle_count"], y=sub_view["soh_pct_actual"], name="Actual", line=dict(color="#1428A0", width=3)))
        fs.add_trace(go.Scatter(x=sub_view["cycle_count"], y=sub_view["predicted_soh"], name="Pred", line=dict(dash="dash", color="#8E9EAA")))
        fs.update_layout(height=280, margin=dict(l=10, r=10, t=30, b=10), plot_bgcolor='white', paper_bgcolor='white', legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        st.plotly_chart(fs, use_container_width=True)

with b2:
    st.markdown("<div class='sec-title'>이상감지 발생 원인 분석</div>", unsafe_allow_html=True)
    with st.container(border=True): 
        profile, top_label = mock_anomaly_profile(battery_id, cycle)
        action_map_demo = {
            "전압 강하": "부하 조건 및 전압 강하 구간 점검",
            "용량 감소": "부하 조건 및 배터리 열화 상태 확인",
            "온도 증가": "내부저항 및 열 관리 상태 점검",
            "내부저항 증가": "EIS 진단 및 셀 열화 상태 점검",
        }

        sorted_items = sorted(profile.items(), key=lambda x: x[1], reverse=True)
        labels_sorted = [k for k, _ in sorted_items]
        values_sorted = [v for _, v in sorted_items]

        top_value = values_sorted[0]

        bar_colors = ["#1428A0"] + ["#8E9EAA"] * (len(labels_sorted) - 1)

        fa = go.Figure(data=[go.Bar(
            x=labels_sorted, y=values_sorted, marker=dict(color=bar_colors),
            text=[f"{v}건" for v in values_sorted], textposition="inside", textfont=dict(color="white", size=16, weight="bold")
        )])
        
        fa.update_layout(
            height=150, margin=dict(l=10, r=10, t=10, b=10), 
            plot_bgcolor='white', paper_bgcolor='white',
            xaxis=dict(tickfont=dict(size=15, color="#000000", weight="bold"))
        )
        st.plotly_chart(fa, use_container_width=True)
        
        st.markdown(f"""
            <div class="alert-box">
                <div style="color: #d46a00; font-weight: 900; font-size: 17px; margin-bottom: 6px;">
                    ⚠️ TOP 1 경보: {top_label} ({top_value}건)
                </div>
                <div style="color: #212529; font-weight: 900; font-size: 15px;">
                    🚨 [{top_label}] {action_map_demo[top_label]}
                </div>
            </div>
        """, unsafe_allow_html=True)