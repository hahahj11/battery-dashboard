import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# --- 페이지 설정 ---
st.set_page_config(page_title="Enterprise Analytics", page_icon="🏢", layout="wide")

# =====================================================================
# ★ 글씨 크기 +3px 업그레이드된 Samsung Enterprise 스타일 CSS 주입 ★
# =====================================================================
st.markdown("""
<style>
    /* 1. 전체 앱 배경 (세련된 쿨그레이) */
    .stApp {
        background-color: #F2F4F7 !important;
    }
    
    /* 2. Streamlit 기본 라디오 버튼 등 기본 텍스트 크기 증가 */
    .stRadio label {
        font-size: 17px !important;
    }
    
    /* 3. Streamlit 기본 컨테이너(border=True)를 삼성 스타일 카드로 강제 변환 */
    [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff;
        border: 1px solid #E5E8EB !important;
        border-radius: 16px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03) !important;
        padding: 8px;
        transition: all 0.2s ease-in-out;
    }
    [data-testid="stVerticalBlockBorderWrapper"]:hover {
        box-shadow: 0 6px 24px rgba(20, 40, 160, 0.06) !important;
        border-color: #D1D6DB !important;
    }

    /* 4. 커스텀 카드 (텍스트 요약 및 TOP RISK용) - ★라인 박스 및 흰 배경 복구★ */
    .sec-card {
        background-color: #ffffff !important;
        border: 1px solid #E5E8EB !important;
        border-radius: 16px;
        padding: 20px !important;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.03) !important;
        margin-bottom: 20px;
        transition: all 0.2s ease-in-out;
    }
    .sec-card:hover {
        box-shadow: 0 6px 24px rgba(20, 40, 160, 0.06) !important;
        border-color: #D1D6DB !important;
    }
    
    /* 5. 삼성 스타일 섹션 타이틀 (크기 16 -> 19px) */
    .sec-title {
        font-size: 19px; 
        font-weight: 700;
        color: #111111;
        display: flex;
        align-items: center;
        margin-bottom: 16px;
        letter-spacing: -0.5px;
    }
    .sec-title::before {
        content: '';
        display: inline-block;
        width: 4px;
        height: 19px; /* 타이틀 크기에 맞춰 16 -> 19px */
        background-color: #1428A0; 
        margin-right: 8px;
        border-radius: 2px;
    }

    /* 6. 텍스트 및 하이라이트 스타일 (전체 +3px) */
    .sec-text { font-size: 17px; color: #495057; line-height: 1.7; }
    .sec-highlight { color: #1428A0; font-weight: 800; font-size: 18px; }
    .sec-danger { color: #E6192B; font-weight: 800; }
    
    /* 7. 탑 배너 헤더 */
    .sec-header-banner {
        background: linear-gradient(135deg, #1428A0 0%, #0072CE 100%);
        border-radius: 16px;
        padding: 30px;
        color: white;
        margin-bottom: 24px;
        box-shadow: 0 8px 24px rgba(20, 40, 160, 0.15);
    }
    /* ★ 여기서 제목과 소제목 폰트 사이즈를 수정했습니다 ★ */
    .sec-header-banner h1 { color: white; margin: 0; font-size: 42px; font-weight: 800; letter-spacing: -1px; } 
    .sec-header-banner p { color: rgba(255,255,255,0.8); margin: 8px 0 0 0; font-size: 18px; } 
    
    /* 8. 데이터프레임 폰트 크기 증가 */
    [data-testid="stDataFrame"] {
        font-size: 15px;
    }
</style>
""", unsafe_allow_html=True)

TARGET_BATTS = ["B0005", "B0006", "B0007", "B0018"]

# =====================================================================
# 1. 데이터 생성 (기존 유지)
# =====================================================================
MAX_CYCLES = 150 

@st.cache_data
def generate_long_term_dummy_data():
    np.random.seed(42)
    data = []
    end_time = datetime.now()
    start_time = end_time - timedelta(days=MAX_CYCLES)
    
    for batt in TARGET_BATTS:
        if batt == "B0006":
            base_temp, base_rul, rul_deg_rate, soh_deg_rate, alert_prob = 37, 160, 1.0, 0.0022, 0.15
        elif batt == "B0018":
            base_temp, base_rul, rul_deg_rate, soh_deg_rate, alert_prob = 32, 172, 1.0, 0.0016, 0.12
        elif batt == "B0007":
            base_temp, base_rul, rul_deg_rate, soh_deg_rate, alert_prob = 28, 195, 0.8, 0.0010, 0.08
        else: # B0005
            base_temp, base_rul, rul_deg_rate, soh_deg_rate, alert_prob = 25, 192, 0.7, 0.0006, 0.05
            
        base_soh = 1.0
        
        for cycle in range(1, MAX_CYCLES + 1):
            timestamp = start_time + timedelta(days=cycle)
            current_rul = max(0, base_rul - (cycle * rul_deg_rate) + np.random.normal(0, 1.5))
            current_soh = max(0.5, base_soh - (cycle * soh_deg_rate) + np.random.normal(0, 0.001))
            temp = base_temp + (cycle * 0.02) + np.random.normal(0, 1.5)
            
            alert_type = "없음"
            severity = "정상"
            
            current_alert_prob = alert_prob
            if temp > 38 or current_rul < 70: current_alert_prob *= 1.5
            if np.random.rand() < current_alert_prob:
                alert_type = np.random.choice(["온도 이상 증후", "전압 강하 (Sag)", "내부저항 경고"])
                severity = np.random.choice(["주의(Warning)", "위험(Critical)"], p=[0.75, 0.25])
            
            data.append({
                "battery_id": batt, "cycle_id": cycle, "timestamp": timestamp,
                "SOH": current_soh * 100, "RUL_pred": current_rul, "H2_Max_Temp": temp,
                "alert_type": alert_type, "severity": severity
            })
    return pd.DataFrame(data)

try:
    df = generate_long_term_dummy_data()
except Exception as e:
    st.error(f"데이터 생성 실패: {e}")
    st.stop()

# --- 2. 상단 타이틀 (배너형 헤더 적용) ---
# ★ 여기서 제목과 소제목 텍스트를 변경했습니다 ★
st.markdown("""
<div class="sec-header-banner">
    <h1>배터리 수명 예측 및 운영 분석</h1>
    <p>장기 운영 데이터 기반 RUL 예측과 위험 셀 관리</p>
</div>
""", unsafe_allow_html=True)

# --- 3. 조회 기간 필터 ---
st.markdown("<div style='text-align: center; margin-bottom: 20px;'>", unsafe_allow_html=True)
scope_filter = st.radio(
    "조회 기간 선택", 
    ["최근 1분기 (3개월)", "최근 1년 (12개월)", "전체 기간"], 
    horizontal=True, label_visibility="collapsed"
)
st.markdown("</div>", unsafe_allow_html=True)

# 데이터 필터링
now = df['timestamp'].max()
if scope_filter == "최근 1분기 (3개월)": filtered_df = df[df['timestamp'] >= now - pd.Timedelta(days=90)]
elif scope_filter == "최근 1년 (12개월)": filtered_df = df[df['timestamp'] >= now - pd.Timedelta(days=365)]
else: filtered_df = df.copy()

latest_df = df[df['timestamp'] == now]
worst_batt_row = latest_df.loc[latest_df['RUL_pred'].idxmin()]
lowest_rul = worst_batt_row['RUL_pred']
worst_batt_id = worst_batt_row['battery_id']
short_scope = scope_filter.split(' ')[0] + " " + scope_filter.split(' ')[1]


# --- 4. 메인 레이아웃 (여백 조절) ---
left_col, right_col = st.columns([5, 5], gap="large")

# ==========================================
# (좌측) 장기 통계 분석 및 로그
# ==========================================
with left_col:
    # 제목 18 -> 21px
    st.markdown("<div style='font-size:21px; font-weight:800; color:#111; margin-bottom:15px; letter-spacing:-0.5px;'>장기 운영 통계 분석</div>", unsafe_allow_html=True)
    
    # 1. 월별 경보 발생 분포 
    with st.container(border=True):
        st.markdown("<div class='sec-title'>월별 경보 발생 분포</div>", unsafe_allow_html=True)
        monthly_alerts = filtered_df[filtered_df['alert_type'] != "없음"].copy()
        if not monthly_alerts.empty:
            monthly_alerts['Month'] = monthly_alerts['timestamp'].dt.to_period('M').astype(str)
            monthly_alerts = monthly_alerts.sort_values('timestamp')
            alert_summary = monthly_alerts.groupby(['Month', 'severity']).size().reset_index(name='count')
            
            # 위험(남색), 주의(회색) 적용 
            fig_alerts = px.bar(alert_summary, x='Month', y='count', color='severity', 
                                color_discrete_map={'주의(Warning)': '#D1D6DB', '위험(Critical)': '#1428A0'})
            
            # 차트 배경 흰색 적용 
            fig_alerts.update_layout(
                font=dict(size=15),
                height=230, margin=dict(t=10, b=0, l=0, r=0), 
                plot_bgcolor="white", paper_bgcolor="white",
                xaxis=dict(showgrid=False, title="", tickfont=dict(color="#888", size=15)), 
                yaxis=dict(showgrid=True, gridcolor="#F2F4F7", title="", tickfont=dict(color="#888", size=15)),
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="", font=dict(size=14))
            )
            st.plotly_chart(fig_alerts, use_container_width=True)
        else:
            st.info("해당 기간에 발생한 경보가 없습니다.")

    # 2. 운영 인사이트 요약
    alerts_df = filtered_df[filtered_df['alert_type'] != "없음"]
    crit_count = len(alerts_df[alerts_df['severity'] == '위험(Critical)'])
    most_freq_alert = alerts_df['alert_type'].mode()[0] if not alerts_df.empty else "없음"
    
    st.markdown(f"""
    <div class="sec-card">
        <div class="sec-title">운영 인사이트 요약 ({short_scope})</div>
        <div class="sec-text">
            ▪ <b>위험(Critical) 경보 누적:</b> <span class="sec-danger">총 {crit_count} 건 발생</span><br>
            ▪ <b>최다 발생 트렌드:</b> {most_freq_alert} 위주 모니터링 요망<br>
            ▪ <b>가장 시급한 조치 대상:</b> <span class="sec-highlight">{worst_batt_id}</span> 채널 점검 권장
        </div>
    </div>
    """, unsafe_allow_html=True)

    # 3. 이벤트 및 경보 발생 로그 
    with st.container(border=True):
        st.markdown("<div class='sec-title'>이벤트 및 경보 발생 로그</div>", unsafe_allow_html=True)
        logs = df[df['alert_type'] != "없음"].copy()
        log_display = logs.sort_values('timestamp', ascending=False).head(50)[['timestamp', 'battery_id', 'alert_type', 'severity']]
        log_display['timestamp'] = log_display['timestamp'].dt.strftime('%m-%d %H:%M')
        log_display.columns = ["발생 일시", "배터리 ID", "경보 유형", "심각도"]
        
        # 위험(남색 계열 투명도), 주의(회색 계열 투명도) 적용 
        def style_severity(val):
            if val == '위험(Critical)': return 'background-color: rgba(20, 40, 160, 0.15); color: #1428A0; font-weight: bold;'
            elif val == '주의(Warning)': return 'background-color: rgba(209, 214, 219, 0.4); color: #495057; font-weight: bold;'
            return ''
            
        st.dataframe(log_display.style.map(style_severity, subset=['심각도']), use_container_width=True, hide_index=True, height=170)


# ==========================================
# (우측) 예측 및 위험도 관리 패널
# ==========================================
with right_col:
    # 제목 18 -> 21px
    st.markdown("<div style='font-size:21px; font-weight:800; color:#111; margin-bottom:15px; letter-spacing:-0.5px;'>수명 예측 및 위험도 관리 패널</div>", unsafe_allow_html=True)
    
    rul_container = st.container()
    
    fp_c1, fp_c2 = st.columns([3.5, 6.5])
    with fp_c1:
        prob_val = max(5.0, min(99.0, 100.0 - lowest_rul))
        # Top Risk 패널 (padding 복구 및 박스화 적용)
        st.markdown(f"""
        <div class="sec-card" style="display:flex; flex-direction:column; justify-content:center; align-items:center; height: 185px;">
            <div style="font-size:15px; color:#686D76; font-weight:700; margin-bottom:8px;">TOP RISK CHANNEL</div>
            <div style="font-size:35px; color:#E6192B; font-weight:900; line-height:1.1;">{worst_batt_id}</div>
            <div style="font-size:16px; color:#111; margin-top:8px;">고장 확률: <b>{prob_val:.1f}%</b></div>
            <div style="font-size:14px; color:#888; margin-top:2px;">예상 잔존수명: {int(lowest_rul)} Cyc</div>
        </div>
        """, unsafe_allow_html=True)
            
    with fp_c2:
        with st.container(border=True):
            # 14->17px, 11->14px
            st.markdown("<div class='sec-title' style='margin-bottom:0px; font-size:17px;'>채널별 고장 확률 🖱️ <span style='font-size:14px; font-weight:normal; color:#888;'>(클릭)</span></div>", unsafe_allow_html=True)
            fail_probs = []
            for batt in TARGET_BATTS:
                b_rul = latest_df[latest_df['battery_id'] == batt]['RUL_pred'].values[0]
                prob = max(5.0, min(99.0, 100.0 - b_rul))
                fail_probs.append({"배터리": batt, "고장 확률(%)": prob})
                
            # 내림차순 정렬
            fail_probs_df = pd.DataFrame(fail_probs).sort_values(by='고장 확률(%)', ascending=False)
            
            # 제일 높은 건 남색, 나머지는 연한 남색 
            colors = ['#1428A0'] + ['#BDC7E1'] * (len(fail_probs_df) - 1)
            
            fig_prob = px.bar(
                fail_probs_df, x='배터리', y='고장 확률(%)', text_auto='.1f', 
                color='배터리', color_discrete_sequence=colors
            )
            
            # 차트 배경 흰색, 범례 숨김 
            fig_prob.update_layout(
                font=dict(size=15),
                height=135, margin=dict(t=5, b=0, l=0, r=0), 
                yaxis=dict(range=[0, 100], showgrid=True, gridcolor="#F2F4F7", title="", tickfont=dict(color="#888", size=15)), 
                xaxis=dict(title="", tickfont=dict(color="#555", weight="bold", size=15)),
                plot_bgcolor="white", paper_bgcolor="white", showlegend=False
            )
            
            try:
                event = st.plotly_chart(fig_prob, use_container_width=True, on_select="rerun", selection_mode="points")
                if event and event.get('selection', {}).get('points'):
                    target_batt = event['selection']['points'][0]['x']
                else:
                    target_batt = worst_batt_id 
            except TypeError:
                st.plotly_chart(fig_prob, use_container_width=True)
                target_batt = worst_batt_id

    with rul_container:
        with st.container(border=True):
            st.markdown(f"<div class='sec-title'>잔존수명(RUL) 예측 추이 <span style='color:#1428A0;'>({target_batt})</span></div>", unsafe_allow_html=True)
            forecast_df = df[df['battery_id'] == target_batt].tail(100)
            fig_forecast = go.Figure()
            
            fig_forecast.add_trace(go.Scatter(x=forecast_df['cycle_id'], y=forecast_df['RUL_pred'], mode='lines', name='실측 RUL', line=dict(width=3, color='#1428A0')))
            
            last_cycle = forecast_df['cycle_id'].max()
            last_rul = forecast_df['RUL_pred'].iloc[-1]
            future_cycles = np.arange(last_cycle, last_cycle + int(last_rul) + 30) 
            
            degradation_slope = 1.0 
            future_rul = last_rul - ((future_cycles - last_cycle) * degradation_slope)
            future_rul = np.clip(future_rul, -10, None) 
            expected_eol_cycle = future_cycles[np.argmin(np.abs(future_rul))]
            remaining_cycles = expected_eol_cycle - last_cycle
            
            fig_forecast.add_trace(go.Scatter(x=future_cycles, y=future_rul, mode='lines', line=dict(dash='dot', color='#8E9EAA'), name='예측 선'))
            fig_forecast.add_trace(go.Scatter(x=future_cycles, y=future_rul, fill='tozeroy', fillcolor='rgba(20, 40, 160, 0.06)', mode='none', name='오차 범위', showlegend=False))
            
            # ★ RUL 0 기준선 색상 변경: 빨간색 -> 진한 회색(#555555) ★
            fig_forecast.add_hline(y=0, line_dash="solid", line_color="#555555")
            
            # 주석 폰트 12 -> 15px
            fig_forecast.add_annotation(
                x=expected_eol_cycle, y=0,  
                text=f"<b style='color:#111;'>수명 종료 예상 (Cy {expected_eol_cycle})</b><br><span style='color:#555;'>남은 수명: {remaining_cycles} Cyc</span>",
                showarrow=False, xshift=10, yshift=45, xanchor="left", font=dict(size=15)
            )
            
            # 차트 배경 흰색 적용 
            fig_forecast.update_layout(
                font=dict(size=15),
                height=230, margin=dict(t=0, b=0, l=0, r=40), 
                plot_bgcolor="white", paper_bgcolor="white", 
                xaxis=dict(title=dict(text="진행 사이클 (Cycle)", font=dict(size=15, color="#888")), showgrid=False, tickfont=dict(size=15, color="#888")), 
                yaxis=dict(title=dict(text="RUL", font=dict(size=15, color="#888")), showgrid=True, gridcolor="#F2F4F7", tickfont=dict(size=15, color="#888")),
                showlegend=False
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

    # 3. 장기 SOH 열화 추세 
    with st.container(border=True):
        st.markdown("<div class='sec-title'>장기 SOH(건강상태) 열화 추세</div>", unsafe_allow_html=True)
        sampled_df = filtered_df.iloc[::3, :] 
        
        # B0006=남색 고정, 나머지=연한 파랑 및 회색 계열 적용 
        soh_color_map = {
            "B0006": "#1428A0", # 진한 남색 (주요 타겟)
            "B0005": "#8BA4D5", # 연한 파랑
            "B0007": "#A1ABB6", # 중간 회색
            "B0018": "#D1D6DB"  # 연한 회색
        }
        
        fig_soh = px.line(
            sampled_df, x='timestamp', y='SOH', color='battery_id',
            color_discrete_map=soh_color_map
        )
        
        # 차트 배경 흰색 적용 
        fig_soh.update_layout(
            font=dict(size=15),
            height=210, margin=dict(t=0, b=0, l=0, r=0), 
            plot_bgcolor="white", paper_bgcolor="white", 
            xaxis=dict(title="", showgrid=False, tickfont=dict(color="#888", size=15)), 
            yaxis=dict(title=dict(text="SOH (%)", font=dict(size=15, color="#888")), showgrid=True, gridcolor="#F2F4F7", tickfont=dict(color="#888", size=15)),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, title="", font=dict(size=15))
        )
        st.plotly_chart(fig_soh, use_container_width=True)