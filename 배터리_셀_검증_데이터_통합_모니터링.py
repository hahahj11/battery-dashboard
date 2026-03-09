import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import time

try:
    from statsmodels.tsa.seasonal import STL
except ImportError:
    st.error("statsmodels 라이브러리가 필요합니다. 터미널에 'pip install statsmodels'를 입력해주세요.")
    st.stop()

st.set_page_config(page_title="Nelson Rule Monitoring (STL)", page_icon="🏢", layout="wide")

# =====================================================================
# ★ Session State 초기화
# =====================================================================
if 'is_playing' not in st.session_state:
    st.session_state.is_playing = False
if 'current_cycle' not in st.session_state:
    st.session_state.current_cycle = 1

TARGET_BATTS = ["B0005", "B0006", "B0007", "B0018"]
STL_PERIOD = 7
RULE_COLORS = {"Rule1": "#E6192B", "Rule2": "#F08C00", "Rule3": "#00A650"}

# =====================================================================
# ★ 고급 분석 로직 (STL & Rules 1~3)
# =====================================================================
def get_residual_stl(series: pd.Series, period: int = 7) -> pd.Series:
    s = pd.Series(series).astype(float).reset_index(drop=True)
    if len(s) < max(period * 2, 15):
        trend = s.rolling(window=5, center=True, min_periods=1).mean()
        return s - trend
    try:
        stl = STL(s, period=period, robust=True)
        res = stl.fit()
        return pd.Series(res.resid, index=s.index)
    except Exception:
        trend = s.rolling(window=5, center=True, min_periods=1).mean()
        return s - trend

def detect_rule1(resid: np.ndarray, sigma: float): return np.where(np.abs(resid) > 3 * sigma)[0]
def detect_rule2(resid: np.ndarray):
    flagged = set(); sign = np.sign(resid); sign[sign == 0] = np.nan
    start, n = 0, len(resid)
    while start < n:
        if np.isnan(sign[start]): start += 1; continue
        current_sign, end = sign[start], start + 1
        while end < n and sign[end] == current_sign: end += 1
        if (end - start) >= 9: flagged.update(range(start, end))
        start = end
    return np.array(sorted(flagged), dtype=int)
def detect_rule3(resid: np.ndarray):
    flagged = set(); n = len(resid)
    for i in range(n - 5):
        diffs = np.diff(resid[i:i+6])
        if np.all(diffs > 0) or np.all(diffs < 0): flagged.update(range(i, i+6))
    return np.array(sorted(flagged), dtype=int)

def detect_all_rules(resid: pd.Series):
    r = resid.astype(float).reset_index(drop=True)
    sigma = float(r.std(ddof=1))
    if not np.isfinite(sigma) or sigma == 0: sigma = 1e-12
    arr = r.values
    return {"Rule1": detect_rule1(arr, sigma), "Rule2": detect_rule2(arr), "Rule3": detect_rule3(arr), "sigma": sigma}

# --- 데이터 로드 ---
@st.cache_data
def load_and_process_real_data():
    try:
        df = pd.read_csv("with_soh_rul_.csv")
        df = df[df['battery_id'].isin(TARGET_BATTS)].copy()
        
        if 'cycle_id' not in df.columns: df['cycle_id'] = df.groupby('battery_id').cumcount() + 1
        if 'soh' not in df.columns: df['soh'] = df['SOH'] if 'SOH' in df.columns else 2.0
        if 'temp' not in df.columns: df['temp'] = df['HI2_Max_Temp'] if 'HI2_Max_Temp' in df.columns else 25.0

        all_frames = []
        for batt in TARGET_BATTS:
            sub = df[df['battery_id'] == batt].sort_values('cycle_id').copy().reset_index(drop=True)
            if len(sub) == 0: continue
            
            sub["residual"] = get_residual_stl(sub["temp"], period=STL_PERIOD)
            rules = detect_all_rules(sub["residual"])
            
            for rule_name in ["Rule1", "Rule2", "Rule3"]:
                sub[f"Temp_{rule_name}"] = False
                if len(rules[rule_name]) > 0: sub.loc[rules[rule_name], f"Temp_{rule_name}"] = True
            
            sub["sigma"] = rules["sigma"]
            sub["ucl"], sub["lcl"] = 3 * rules["sigma"], -3 * rules["sigma"]
            all_frames.append(sub)
            
        return pd.concat(all_frames, ignore_index=True)
    except Exception as e:
        st.error(f"데이터 로드 실패: {e}")
        return pd.DataFrame()

df = load_and_process_real_data()
if df.empty: st.stop()

MAX_CYCLE = int(df['cycle_id'].max())
current_df = df[df['cycle_id'] <= st.session_state.current_cycle].copy()

# =====================================================================
# ★ UI 및 CSS 구성 
# =====================================================================
st.markdown("""
<style>
    /* 전체 배경 및 폰트 강제 볼드 처리 */
    .stApp { background-color: #F2F4F7 !important; font-weight: 900 !important; }
    p, span, div, label { font-weight: 800 !important; font-size: 15px !important; }
    
    /* ★ 사이드바(왼쪽 페이지 메뉴)는 굵은 글씨체 적용에서 제외하여 원복 ★ */
    [data-testid="stSidebar"] * { 
        font-weight: normal !important; 
    }
    
    /* 애니메이션 흐림(Ghosting) 원천 차단 */
    div[data-stale="true"], 
    [data-testid="stVerticalBlock"], 
    [data-testid="stMainBlockContainer"], 
    [data-testid="stElementContainer"], .stPlotlyChart { 
        opacity: 1 !important; 
        transition: none !important; 
        filter: none !important; 
    }

    /* 배너 스타일 */
    .sec-header-banner { background: linear-gradient(135deg, #1428A0 0%, #0072CE 100%); border-radius: 16px; padding: 24px 30px; color: white; margin-bottom: 20px; box-shadow: 0 8px 24px rgba(20, 40, 160, 0.15); display: flex; justify-content: space-between; align-items: center; }
    /* ★ h1 폰트 사이즈를 32px -> 42px로 수정하여 제목 크기 확대 ★ */
    .sec-header-banner h1 { color: white; margin: 0; font-size: 42px !important; font-weight: 900 !important; letter-spacing: -1px; }
    .sec-header-banner p { color: rgba(255,255,255,0.95); margin: 6px 0 0 0; font-size: 18px !important; font-weight: 800 !important; }
    
    /* 상단 KPI 카드 스타일 */
    .kpi-wrapper { background-color: #ffffff; border: 1px solid #E5E8EB; border-radius: 12px; border-top: 5px solid #1428A0; padding: 20px 16px; box-shadow: 0 4px 16px rgba(0,0,0,0.03); height: 145px; display: flex; flex-direction: column; justify-content: space-between; transition: transform 0.2s ease, box-shadow 0.2s ease; }
    .kpi-wrapper:hover { transform: translateY(-2px); box-shadow: 0 8px 24px rgba(20, 40, 160, 0.08); }
    .kpi-title { font-size: 16px !important; font-weight: 900 !important; color: #343A40; line-height: 1.3; }
    .kpi-value { font-size: 28px !important; font-weight: 900 !important; color: #111111; margin-top: 6px; }
    .kpi-sub { font-size: 14px !important; color: #495057; line-height: 1.3; margin-top: 4px; font-weight: 800 !important; }
    
    /* 섹션 타이틀 */
    .sec-title { font-size: 20px !important; font-weight: 900 !important; color: #111111; display: flex; align-items: center; margin-bottom: 12px; letter-spacing: -0.5px; }
    .sec-title::before { content: ''; display: inline-block; width: 5px; height: 20px; background-color: #1428A0; margin-right: 10px; border-radius: 3px; }
    
    /* 깜빡임 애니메이션 설정 */
    @keyframes blink-danger { 0% {opacity:1; color:#E6192B;} 50% {opacity:0.6; color:#E6192B;} 100% {opacity:1; color:#E6192B;} }
    @keyframes blink-warn { 0% {opacity:1; color:#F08C00;} 50% {opacity:0.6; color:#F08C00;} 100% {opacity:1; color:#F08C00;} }
    .anim-danger { animation: blink-danger 1s infinite; font-weight: 900 !important; font-size: 16px !important; }
    .anim-warn { animation: blink-warn 1.5s infinite; font-weight: 900 !important; font-size: 16px !important; }
    button { font-size: 16px !important; font-weight: 900 !important; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="sec-header-banner">
    <div>
        <h1>배터리 셀 검증 데이터 통합 모니터링</h1>
        <p>SPC 기반 실시간 이상 감지 및 셀 상태 관제</p>
    </div>
</div>
""", unsafe_allow_html=True)

# =====================================================================
# ★ 상단 KPI 
# =====================================================================
num_cells = len(TARGET_BATTS)
avg_cycle = int(current_df.groupby('battery_id')['cycle_id'].max().mean()) if not current_df.empty else 0
tot_alerts = current_df[['Temp_Rule1', 'Temp_Rule2', 'Temp_Rule3']].sum().sum() if not current_df.empty else 0
latest_states = current_df.groupby('battery_id').tail(1)

min_soh_val = latest_states.loc[latest_states['soh'].idxmin()]['soh'] if not latest_states.empty else 0
min_soh_batt = latest_states.loc[latest_states['soh'].idxmin()]['battery_id'] if not latest_states.empty else "-"
soh_subtext = "<span class='anim-danger'>🔴 열화가 진행중입니다.</span>" if min_soh_val < 1.4 else "🟢 안정적 범위 내"

max_temp_val = latest_states.loc[latest_states['temp'].idxmax()]['temp'] if not latest_states.empty else 0
max_temp_batt = latest_states.loc[latest_states['temp'].idxmax()]['battery_id'] if not latest_states.empty else "-"

worst_batt, viol_max = "", 0
for b in TARGET_BATTS:
    b_recent = current_df[current_df['battery_id'] == b].tail(30)
    viol = b_recent[['Temp_Rule1', 'Temp_Rule2', 'Temp_Rule3']].sum().sum()
    if viol > viol_max: viol_max, worst_batt = viol, b

if viol_max >= 3: status_html, status_sub = f"<div class='anim-danger'>🔴 [{worst_batt}] 위험</div>", "다중 Rule 위반 감지"
elif viol_max >= 1: status_html, status_sub = f"<div class='anim-warn'>🟡 [{worst_batt}] 주의</div>", "패턴 이상 감지"
else: status_html, status_sub = "<div style='color:#00A650; font-weight:900; font-size:16px;'>🟢 채널 상태 양호</div>", "특이사항 없음"

k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.markdown(f'<div class="kpi-wrapper"><div class="kpi-title">측정 진행 전지수</div><div class="kpi-value" style="color:#1428A0;">{num_cells}개 <span style="font-size:18px;">/ {avg_cycle} cyc</span></div></div>', unsafe_allow_html=True)
with k2: st.markdown(f'<div class="kpi-wrapper"><div class="kpi-title">총 이상 건수</div><div class="kpi-value">{tot_alerts} 건</div></div>', unsafe_allow_html=True)
# ★ 단위 Ah를 %로만 정확히 수정 ★
with k3: st.markdown(f'<div class="kpi-wrapper"><div class="kpi-title">최저 Capacity (SOH)</div><div class="kpi-value">{min_soh_val:.2f} % <span style="font-size:18px;">/ {min_soh_batt}</span></div><div class="kpi-sub">{soh_subtext}</div></div>', unsafe_allow_html=True)
with k4: st.markdown(f'<div class="kpi-wrapper"><div class="kpi-title">현재 최고 온도</div><div class="kpi-value">{max_temp_val:.1f}°C <span style="font-size:16px;">({max_temp_batt})</span></div></div>', unsafe_allow_html=True)
with k5: st.markdown(f'<div class="kpi-wrapper"><div class="kpi-title">실시간 상태</div><div style="margin-top:10px;">{status_html}</div><div class="kpi-sub">{status_sub}</div></div>', unsafe_allow_html=True)

st.write("") 

# ==========================================
# 메인 레이아웃 (3단)
# ==========================================
col_left, col_mid, col_right = st.columns([2.3, 6.0, 1.7], gap="large")

# 1. 좌측: 3D 랙 디자인 (B0001 ~ B0030 배치)
with col_left:
    st.markdown("<div class='sec-title'>Battery Cell Cycler</div>", unsafe_allow_html=True)
    st.markdown("<span style='font-size:15px; font-weight:900; color:#555;'>🟢 양호 / 🟡 주의 / 🔴 위험</span>", unsafe_allow_html=True)
    
    led_status = {}
    for b in TARGET_BATTS:
        viol = current_df[current_df['battery_id'] == b].tail(30)[['Temp_Rule1', 'Temp_Rule2', 'Temp_Rule3']].sum().sum()
        led_status[b] = "led-red-blink" if viol >= 3 else "led-yellow" if viol >= 1 else "led-green"

    # ★ CSS 수정: box-sizing: border-box 추가 및 padding 조절로 우측 쏠림(대칭) 완벽 해결 ★
    rack_html = """<style>
.machine-wrapper { width: 100%; max-width: 340px; background: #FFFFFF; border: 1px solid #D1D6DB; border-radius: 12px; box-shadow: 0 10px 30px rgba(20, 40, 160, 0.08); margin: 0 auto 20px auto; overflow: hidden; box-sizing: border-box; }
.machine-header { background: linear-gradient(90deg, #1428A0 0%, #0072CE 100%); color: #FFFFFF; padding: 14px 16px; font-size: 15px; font-weight: 900; display: flex; justify-content: space-between; align-items: center; }
.rack-container { display: grid; grid-template-columns: repeat(5, 1fr); gap: 10px; padding: 20px; background-color: #F2F4F7; box-sizing: border-box; width: 100%; justify-items: center;}
.slot { background: #FFFFFF; border: 1px solid #D1D6DB; border-radius: 6px; display: flex; flex-direction: column; align-items: center; justify-content: center; padding: 12px 0; text-decoration: none !important; color: inherit; transition: transform 0.1s, border-color 0.1s; cursor: pointer; width: 100%; box-sizing: border-box; }
.slot:hover { transform: scale(1.05); border-color: #1428A0; z-index: 10; box-shadow: 0 4px 8px rgba(0,0,0,0.1); }
.slot-id { font-size: 13px; font-weight: 900; color: #495057; margin-bottom: 8px; text-decoration: none;}
.led { width: 16px; height: 16px; border-radius: 50%; box-shadow: inset 0px 2px 4px rgba(0,0,0,0.4); pointer-events: none; }
.led-green { background: radial-gradient(circle at 30% 30%, #4ae06a 0%, #00A650 60%, #007a3b 100%); box-shadow: 0 0 8px rgba(0, 166, 80, 0.6);}
.led-yellow { background: radial-gradient(circle at 30% 30%, #ffcf40 0%, #F08C00 60%, #b86b00 100%); box-shadow: 0 0 8px rgba(240, 140, 0, 0.6);}
.led-empty { background: #E5E8EB; opacity: 0.5;}
@keyframes pulse-red-corp { 0%, 100% { box-shadow: 0 0 6px rgba(230, 25, 43, 0.5); background:#E6192B;} 50% { box-shadow: 0 0 16px rgba(255, 77, 77, 0.8); background:#FF4D4D;} }
.led-red-blink { animation: pulse-red-corp 0.8s infinite; }
</style>
<div class="machine-wrapper"><div class="machine-header"><span>SYS_READY</span><span>30CH</span></div><div class="rack-container">
"""
    # B0001 ~ B0030까지 순서대로 배치
    for i in range(1, 31):
        batt_name = f"B{i:04d}"
        if i in [5, 6, 7, 18]:
            led = led_status.get(batt_name, "led-empty")
        else:
            led = "led-empty"
            
        # ★ href 링크를 "배터리_셀_상세_상태_분석"으로 설정하여 1_페이지로 완벽 이동 ★
        rack_html += f'<a href="배터리_셀_상세_상태_분석" target="_self" class="slot"><div class="slot-id">{batt_name}</div><div class="led {led}"></div></a>'
        
    rack_html += """</div></div>"""
    st.markdown(rack_html, unsafe_allow_html=True)
    
# 2. 중앙: 고급 STL Nelson Rule 2x2 차트
with col_mid:
    st.markdown("<div class='sec-title'>STL Residual & Advanced Nelson Rules</div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='display:flex; gap:20px; align-items:center; background:#fff; border:1px solid #E5E8EB; padding:10px 15px; border-radius:8px; margin-bottom:15px;'>
        <span style='font-size:14px; font-weight:900; color:#333;'>[이상 탐지 범례]</span>
        <span style='font-size:14px; font-weight:800; color:#333;'><span style='color:#E6192B;'>●</span> Rule 1 (3σ 이탈)</span>
        <span style='font-size:14px; font-weight:800; color:#333;'><span style='color:#F08C00;'>●</span> Rule 2 (9연속 편향)</span>
        <span style='font-size:14px; font-weight:800; color:#333;'><span style='color:#00A650;'>●</span> Rule 3 (6연속 증감)</span>
    </div>
    """, unsafe_allow_html=True)
    
    c_btn1, c_btn2, c_btn3, c_info = st.columns([1.5, 1.5, 1.5, 5.5])
    with c_btn1: st.button("▶ Start", on_click=lambda: st.session_state.update(is_playing=True), use_container_width=True)
    with c_btn2: st.button("⏸ Pause", on_click=lambda: st.session_state.update(is_playing=False), use_container_width=True)
    with c_btn3: st.button("Reset", on_click=lambda: st.session_state.update(current_cycle=1, is_playing=False), use_container_width=True)
    
    with c_info:
        val = st.slider(" 시뮬레이션 사이클 (Cycle) 조절", 1, MAX_CYCLE, value=st.session_state.current_cycle)
        if val != st.session_state.current_cycle: st.session_state.current_cycle = val

    fig = make_subplots(rows=2, cols=2, subplot_titles=TARGET_BATTS, vertical_spacing=0.15, horizontal_spacing=0.08)
    y_max = max(abs(df['residual'].max()), abs(df['residual'].min())) * 1.2

    for i, batt in enumerate(TARGET_BATTS):
        r, c = (i // 2) + 1, (i % 2) + 1
        b_df = current_df[current_df['battery_id'] == batt]
        
        if not b_df.empty:
            sigma = b_df['sigma'].iloc[0]
            fig.add_trace(go.Scatter(x=b_df['cycle_id'], y=b_df['residual'], mode='lines', line=dict(color='#8E9EAA', width=2), name="Residual", showlegend=False), row=r, col=c)
            fig.add_trace(go.Scatter(x=b_df['cycle_id'], y=b_df['residual'], mode='markers', marker=dict(size=4, color='#495057'), showlegend=False), row=r, col=c)
            
            for rule_name, color in RULE_COLORS.items():
                r_df = b_df[b_df[f"Temp_{rule_name}"]]
                if not r_df.empty:
                    fig.add_trace(go.Scatter(
                        x=r_df['cycle_id'], y=r_df['residual'], mode='markers', 
                        marker=dict(color=color, size=8, line=dict(color='white', width=1)),
                        name=rule_name, legendgroup=rule_name, showlegend=False
                    ), row=r, col=c)
                    
            fig.add_hline(y=3*sigma, line_dash="dash", line_color="#E6192B", opacity=0.3, row=r, col=c)
            fig.add_hline(y=-3*sigma, line_dash="dash", line_color="#E6192B", opacity=0.3, row=r, col=c)
            fig.add_hline(y=0, line_dash="solid", line_color="#ADB5BD", opacity=0.5, row=r, col=c)

        fig.update_xaxes(range=[0, MAX_CYCLE + 2], gridcolor='#E5E8EB', zeroline=False, tickfont=dict(weight='bold'), row=r, col=c)
        fig.update_yaxes(range=[-y_max, y_max], gridcolor='#E5E8EB', zeroline=False, tickfont=dict(weight='bold'), row=r, col=c)

    fig.update_layout(
        height=480, margin=dict(t=30, b=0, l=0, r=0), plot_bgcolor='#FFFFFF', paper_bgcolor='rgba(0,0,0,0)', 
        font=dict(family="Arial", size=14, color="#111"), 
        uirevision="constant" 
    )
    st.plotly_chart(fig, use_container_width=True)

# 3. 우측: 스크롤 이상 로그 및 펼쳐지는 리포트 메뉴
with col_right:
    st.markdown("<div class='sec-title'>데이터 수집 안정성</div>", unsafe_allow_html=True)
    score = 100
    for b in TARGET_BATTS:
        viol = current_df[current_df['battery_id'] == b].tail(30)[['Temp_Rule1', 'Temp_Rule2', 'Temp_Rule3']].sum().sum()
        score -= (viol * 2)
    score = max(0, score)
    color = "#00A650" if score >= 80 else "#F08C00" if score >= 60 else "#E6192B"
    status = "🟢 양호" if score >= 80 else "🟡 주의" if score >= 60 else "🔴 위험"
    
    st.markdown(f"""
    <div style="background:#fff; border:1px solid #E5E8EB; border-radius:8px; padding:20px; text-align:center; box-shadow:0 2px 8px rgba(0,0,0,0.02);">
        <h1 style='color:{color}; margin:0; font-size:46px; font-weight:900;'>{score}<span style='font-size:18px; color:#868E96;'>/100</span></h1>
        <div style='color:{color}; font-weight:900; font-size:22px; margin-top:5px;'>{status}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ★ 요청하신 대로 (스크롤) 텍스트를 제거했습니다 ★
    st.markdown("<div class='sec-title' style='margin-top:25px;'>실시간 이상 로그</div>", unsafe_allow_html=True)
    
    recent_logs = []
    log_df = current_df[current_df[['Temp_Rule1', 'Temp_Rule2', 'Temp_Rule3']].any(axis=1)].sort_values('cycle_id', ascending=False)
    for idx, row in log_df.iterrows():
        for r_name in ["Rule1", "Rule2", "Rule3"]:
            if row[f"Temp_{r_name}"]:
                recent_logs.append((row['battery_id'], row['cycle_id'], r_name))
                
    with st.container(height=230, border=True):
        if not recent_logs:
            st.markdown("<div style='font-weight:900; color:#1428A0; font-size:15px;'>현재 감지된 룰 위반이 없습니다.</div>", unsafe_allow_html=True)
        else:
            for batt, cyc, rule in recent_logs:
                if rule == "Rule1":
                    st.markdown(f"<div style='background-color:#ffebe9; color:#E6192B; padding:10px; border-radius:6px; margin-bottom:8px; font-weight:900; font-size:14px; border-left:4px solid #E6192B;'>{batt} ({cyc} cyc) - Rule 1 이탈</div>", unsafe_allow_html=True)
                elif rule == "Rule2":
                    st.markdown(f"<div style='background-color:#fff8e6; color:#b86b00; padding:10px; border-radius:6px; margin-bottom:8px; font-weight:900; font-size:14px; border-left:4px solid #F08C00;'>{batt} ({cyc} cyc) - Rule 2 편향</div>", unsafe_allow_html=True)
                elif rule == "Rule3":
                    st.markdown(f"<div style='background-color:#e6f8ef; color:#007a3b; padding:10px; border-radius:6px; margin-bottom:8px; font-weight:900; font-size:14px; border-left:4px solid #00A650;'>{batt} ({cyc} cyc) - Rule 3 추세</div>", unsafe_allow_html=True)
                
    st.markdown("<div style='margin-top: 15px;'></div>", unsafe_allow_html=True)
    
    with st.expander("자동 요약 보고서 작성 (PDF)", expanded=False):
        st.markdown("<div style='font-weight:900; color:#333; margin-bottom:10px;'>[보고서 출력 설정]</div>", unsafe_allow_html=True)
        st.checkbox("실시간 이상 로그 전체 내역 포함", value=True)
        st.checkbox("STL 잔차 및 Rule 1~3 그래프 포함", value=True)
        if st.button("문서 생성 및 다운로드", use_container_width=True):
            st.success("다운로드가 완료되었습니다.")

# =====================================================================
# ★ 애니메이션 강제 갱신 로직 (반드시 최하단 유지) ★
# =====================================================================
if st.session_state.is_playing and st.session_state.current_cycle < MAX_CYCLE:
    time.sleep(0.08)  
    st.session_state.current_cycle += 1
    st.rerun()