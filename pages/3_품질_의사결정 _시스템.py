import warnings
warnings.filterwarnings("ignore")

import os
import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go

try:
    import shap
    SHAP_OK = True
except Exception:
    SHAP_OK = False


# =========================================================
# 0) Page
# =========================================================
st.set_page_config(page_title="Quality Decision Console - Tab3", layout="wide")

# ★ 삼성 브랜드 아이덴티티 컬러로 교체
BLUE = "#1428A0"
SKY = "#0072CE"
RED = "#E6192B"
TEXT = "#111111"

BASELINE_MAE = 0.015300
MAE_STD = 0.010708


# =========================================================
# 1) Feature 설명
# =========================================================
FEATURE_DESC = {
    "discharge_t_to_V_below_3.5": "방전 과정에서 전압이 3.5V 이하로 떨어질 때까지 걸린 시간",
    "discharge_t_to_V_below_3.5_lag1": "이전 cycle에서 방전 중 전압이 3.5V 이하로 떨어질 때까지 걸린 시간",
    "discharge_t_to_V_below_3.5_diff1": "현재 cycle과 이전 cycle 사이에서 방전 중 전압이 3.5V 이하로 떨어질 때까지 걸린 시간의 변화량",
    "discharge_t_to_V_below_3.5_rm5": "최근 5개 cycle 동안 방전 중 전압이 3.5V 이하로 떨어질 때까지 걸린 시간의 평균",

    "discharge_V_mean": "방전 구간 동안 측정된 전압의 평균값",
    "discharge_V_mean_lag1": "이전 cycle 방전 구간에서 측정된 전압의 평균값",
    "discharge_V_mean_diff1": "현재 cycle과 이전 cycle 사이에서 방전 구간 평균 전압의 변화량",
    "discharge_V_mean_rm5": "최근 5개 cycle 동안 방전 구간 평균 전압의 평균값",

    "slope_Vmeas_50_1500": "방전 시작 후 약 50초부터 1500초 구간에서 측정된 전압 감소 속도(전압 변화 기울기)",
    "slope_Vmeas_50_1500_lag1": "이전 cycle에서 방전 시작 후 약 50초부터 1500초 구간에서 측정된 전압 감소 속도",
    "slope_Vmeas_50_1500_diff1": "현재 cycle과 이전 cycle 사이에서 방전 초기 구간 전압 감소 속도의 변화량",
    "slope_Vmeas_50_1500_rm5": "최근 5개 cycle 동안 방전 초기 구간 전압 감소 속도의 평균값",

    "HI2_Max_Temp": "방전 과정에서 측정된 배터리의 최대 온도",
    "HI2_Max_Temp_lag1": "이전 cycle 방전 과정에서 측정된 배터리 최대 온도",
    "HI2_Max_Temp_diff1": "현재 cycle과 이전 cycle 사이에서 방전 과정 중 최대 온도의 변화량",
    "HI2_Max_Temp_rm5": "최근 5개 cycle 동안 방전 과정에서 측정된 최대 온도의 평균값",

    "discharge_E_Wh_abs": "방전 과정에서 배터리가 방출한 총 에너지(Wh)",
    "discharge_E_Wh_abs_lag1": "이전 cycle 방전 과정에서 배터리가 방출한 총 에너지(Wh)",
    "discharge_E_Wh_abs_diff1": "현재 cycle과 이전 cycle 사이에서 방전 에너지(Wh)의 변화량",
    "discharge_E_Wh_abs_rm5": "최근 5개 cycle 동안 방전 과정에서 방출된 에너지(Wh)의 평균값",
}


# =========================================================
# 2) CSS (★ 가로 폭 제한 해제 및 탭 2와 배너 디자인 통일)
# =========================================================
st.markdown(
    """
    <style>
      /* 전체 앱 배경 (세련된 쿨그레이) */
      .stApp { background-color: #F2F4F7 !important; font-family: 'Arial', sans-serif; }
      html, body, [class*="css"], p, li, label { color:#111111; font-weight: 700 !important; font-size: 16px !important; }

      /* ★ 탭3에만 있던 가로 크기 제한(.block-container max-width) 삭제하여 탭2와 동일하게 화면 꽉 채우도록 수정 ★ */

      /* 각 구역별 하얀 배경 + 연한 회색 라인 박스 테두리 적용 */
      [data-testid="stVerticalBlockBorderWrapper"] {
        background-color: #ffffff !important;
        border: 1px solid #D1D6DB !important; 
        border-radius: 16px !important;
        padding: 24px !important;
        margin-bottom: 20px !important;
        box-shadow: 0 4px 16px rgba(0,0,0,0.02) !important;
      }

      /* 삼성 스타일 섹션 타이틀 */
      .sec-title {
        font-size: 21px !important; 
        font-weight: 800 !important;
        color: #111111;
        display: flex;
        align-items: center;
        margin: 0 0 15px 0;
        letter-spacing: -0.5px;
      }
      .sec-title::before {
        content: '';
        display: inline-block;
        width: 5px;
        height: 22px; 
        background-color: #1428A0; 
        margin-right: 10px;
        border-radius: 2px;
      }

      .title-row-fix { display:flex; align-items:center; }

      /* ★ 탑 배너 헤더 - 탭 2와 완벽히 동일한 디자인/여백/모서리 적용 ★ */
      .sec-header-banner {
          background: linear-gradient(135deg, #1428A0 0%, #0072CE 100%);
          border-radius: 16px;
          padding: 30px;
          color: white;
          margin-bottom: 24px;
          box-shadow: 0 8px 24px rgba(20, 40, 160, 0.15);
      }
      /* ★ h1 폰트 사이즈 42px, p 폰트 사이즈 18px 탭 2와 동일하게 적용 ★ */
      .sec-header-banner h1 { color: white; margin: 0; font-size: 42px !important; font-weight: 800; letter-spacing: -1px; } 
      .sec-header-banner p { color: rgba(255,255,255,0.8); margin: 8px 0 0 0; font-size: 18px !important; } 

      /* KPI 카드 스타일 */
      .kpi-wrapper { 
        background-color: #F8F9FA !important; 
        border: 1px solid #E5E8EB !important; 
        border-top: 4px solid #1428A0 !important;
        border-radius: 10px; 
        padding: 12px 16px !important; 
        height: 90px !important; 
        display: flex; flex-direction: column; justify-content: center; 
      }
      .kpi-title { font-size: 14px !important; color: #686D76 !important; font-weight: 800 !important; margin-bottom: 2px; }
      .kpi-value { font-size: 24px !important; color: #111 !important; font-weight: 900 !important; }

      /* 알람 Action Box */
      .action-good{ border-left: 6px solid #00A650; border-radius: 8px; padding: 18px; background: #ffffff; border: 1px solid #E5E8EB; font-size:16px !important; line-height: 1.6; margin-bottom: 8px;}
      .action-warn{ border-left: 6px solid #F08C00; border-radius: 8px; padding: 18px; background: #ffffff; border: 1px solid #E5E8EB; font-size:16px !important; line-height: 1.6; margin-bottom: 8px;}
      .action-bad{ border-left: 6px solid #E6192B; border-radius: 8px; padding: 18px; background: #ffffff; border: 1px solid #E5E8EB; font-size:16px !important; line-height: 1.6; margin-bottom: 8px;}

      /* Status Pills */
      .status-pill-good{ display:block; width:100%; text-align:center; padding: 16px; border-radius: 8px; background: #e6fcf5; color: #00A650; font-weight: 900; font-size: 18px !important; border: 1px solid #00A650; }
      .status-pill-warn{ display:block; width:100%; text-align:center; padding: 16px; border-radius: 8px; background: #fff9db; color: #F08C00; font-weight: 900; font-size: 18px !important; border: 1px solid #F08C00; }
      .status-pill-bad{ display:block; width:100%; text-align:center; padding: 16px; border-radius: 8px; background: #fff5f5; color: #E6192B; font-weight: 900; font-size: 18px !important; border: 1px solid #E6192B; }

      /* Feature Chips */
      .feature-chip{
        display:inline-block; margin: 4px 6px 4px 0; padding: 6px 14px;
        border-radius: 20px; background: #ffffff; border: 1px solid #ced4da;
        font-weight: 800; font-size: 14px; color: #1428A0; cursor: help;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
      }

      /* 툴팁 (i 아이콘) CSS */
      .info-tip {
        position: relative; display: inline-flex; align-items: center; justify-content: center;
        width: 18px; height: 18px; border-radius: 50%; background: #e8f0ff; color: #1428A0;
        font-size: 12px; font-weight: 900; border: 1px solid #b8caf1; cursor: help; flex-shrink: 0;
        margin-left: 8px;
      }
      .info-tip .tooltip-text {
        visibility: hidden; opacity: 0; 
        position: absolute; 
        top: 50%; left: 140%; 
        transform: translateY(-50%); 
        width: 380px; 
        background: #ffffff !important; 
        color: #111 !important; 
        border: 2px solid #1428A0; 
        text-align: left; padding: 16px;
        border-radius: 8px; 
        font-size: 14px !important; 
        font-weight: 700 !important; 
        line-height: 1.5;
        box-shadow: 0 8px 24px rgba(0,0,0,0.15); transition: opacity 0.2s ease; z-index: 9999; white-space: normal;
      }
      .info-tip .tooltip-text::after {
        content: ""; position: absolute; 
        top: 50%; right: 100%; left: auto; 
        margin-top: -6px;
        border-width: 6px; border-style: solid; 
        border-color: transparent #1428A0 transparent transparent; 
      }
      .info-tip:hover .tooltip-text { visibility: visible; opacity: 1; }

      /* Streamlit Element Styling */
      div[data-testid="stDataFrame"] * { font-size: 15px !important; }
      div[data-testid="stSelectbox"] label, div[data-testid="stSlider"] label, div[data-testid="stRadio"] label { font-size: 17px !important; font-weight: 900 !important; color:#111 !important; margin-bottom: 8px !important; }
      div[data-baseweb="select"] > div { background-color: #ffffff !important; border: 1px solid #adb5bd !important; }
    </style>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 3) Paths 
# =========================================================
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(CURRENT_DIR, "..") 

CSV_PATH = os.path.join(DATA_DIR, "with_soh_rul_.csv")
MODEL_PKL_PATH = os.path.join(DATA_DIR, "ridge_B0006_07_18.pkl")

GROUP_COL = "battery_id"
CYCLE_COL = "cycle_id"
TARGET_COL = "SOH"

BASE_FEATURES = [
    "discharge_t_to_V_below_3.5",
    "discharge_V_mean",
    "slope_Vmeas_50_1500",
    "HI2_Max_Temp",
    "discharge_E_Wh_abs"
]


# =========================================================
# 4) Utils
# =========================================================
def ewma(arr: np.ndarray, alpha: float = 0.3) -> np.ndarray:
    arr = np.asarray(arr, dtype=float)
    if len(arr) == 0:
        return arr
    out = np.zeros_like(arr, dtype=float)
    out[0] = arr[0]
    for i in range(1, len(arr)):
        out[i] = alpha * arr[i] + (1 - alpha) * out[i - 1]
    return out

def add_ts_features(df: pd.DataFrame, base_feats, group_col, cycle_col, lag=1, roll_w=5):
    df = df.sort_values([group_col, cycle_col]).copy()
    for c in base_feats:
        df[f"{c}_lag{lag}"] = df.groupby(group_col)[c].shift(lag)
        df[f"{c}_diff1"] = df.groupby(group_col)[c].diff(1)
        df[f"{c}_rm{roll_w}"] = (
            df.groupby(group_col)[c].rolling(roll_w).mean().reset_index(level=0, drop=True)
        )
    return df

def scenario_apply(df_in: pd.DataFrame, preset: str, n_events: int, severity: float, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = df_in.copy()
    cycles = df[CYCLE_COL].values
    if len(cycles) == 0 or n_events <= 0:
        return df

    pick = rng.choice(cycles, size=min(n_events, len(cycles)), replace=False)
    mask = df[CYCLE_COL].isin(pick)

    if preset.startswith("Preset 1"):
        df.loc[mask, "HI2_Max_Temp"] *= (1 + 0.06 * severity)
        df.loc[mask, "discharge_V_mean"] *= (1 - 0.015 * severity)
        df.loc[mask, "discharge_E_Wh_abs"] *= (1 - 0.035 * severity)
        df.loc[mask, "discharge_t_to_V_below_3.5"] *= (1 - 0.02 * severity)
    elif preset.startswith("Preset 2"):
        df.loc[mask, "discharge_t_to_V_below_3.5"] *= (1 - 0.08 * severity)
        df.loc[mask, "slope_Vmeas_50_1500"] *= (1 + 0.12 * severity)
        df.loc[mask, "discharge_V_mean"] *= (1 - 0.02 * severity)
        df.loc[mask, "discharge_E_Wh_abs"] *= (1 - 0.02 * severity)
    else:
        noise = rng.normal(0, 1, size=(mask.sum(), len(BASE_FEATURES)))
        scale = np.array([0.05, 0.02, 0.08, 0.06, 0.04]) * severity
        for j, c in enumerate(BASE_FEATURES):
            df.loc[mask, c] = df.loc[mask, c] * (1 + noise[:, j] * scale[j])

    return df

def get_cycle_adjusted_importance(model, data_slice: pd.DataFrame, features: list) -> pd.DataFrame:
    prep, ridge = None, None
    try:
        prep = model.named_steps.get("prep", None)
        ridge = model.named_steps.get("model", None)
    except Exception:
        prep, ridge = None, None

    if data_slice is None or len(data_slice) == 0:
        return pd.DataFrame({"feature": features, "importance": np.zeros(len(features))})

    if ridge is None or not hasattr(ridge, "coef_"):
        return pd.DataFrame({"feature": features, "importance": np.zeros(len(features))})

    coef = np.asarray(ridge.coef_).reshape(-1)

    try:
        if prep is not None:
            X_use = prep.transform(data_slice[features])
        else:
            X_use = data_slice[features].to_numpy(dtype=float)

        X_use = np.asarray(X_use, dtype=float)

        if X_use.ndim == 1:
            X_use = X_use.reshape(1, -1)

        if X_use.shape[1] == len(coef):
            imp_vals = np.mean(np.abs(X_use * coef.reshape(1, -1)), axis=0)
            out = pd.DataFrame({"feature": features, "importance": imp_vals})
        else:
            raw_x = data_slice[features].to_numpy(dtype=float)
            if raw_x.ndim == 1:
                raw_x = raw_x.reshape(1, -1)
            use_len = min(raw_x.shape[1], len(coef), len(features))
            imp_vals = np.mean(np.abs(raw_x[:, :use_len] * coef[:use_len].reshape(1, -1)), axis=0)
            out = pd.DataFrame({"feature": features[:use_len], "importance": imp_vals})

        out = out.sort_values("importance", ascending=False).reset_index(drop=True)
        return out

    except Exception:
        return pd.DataFrame({"feature": features, "importance": np.zeros(len(features))})

def make_short_label(feat: str) -> str:
    feat = str(feat)
    feat = feat.replace("discharge_", "")
    feat = feat.replace("slope_Vmeas_50_1500", "slope_V_50_1500")
    feat = feat.replace("HI2_Max_Temp", "Max_Temp")
    feat = feat.replace("E_Wh_abs", "E_Wh")
    feat = feat.replace("_lag1", " (lag1)")
    feat = feat.replace("_diff1", " (diff1)")
    feat = feat.replace("_rm5", " (rm5)")
    return feat

def feature_chip_html(feat: str) -> str:
    desc = FEATURE_DESC.get(feat, feat)
    desc = str(desc).replace('"', "&quot;")
    return f"""
    <span class="feature-chip" title="{desc}">
        {feat}
    </span>
    """

def draw_bar_chart(df_plot, value_col, title_text, color_positive=BLUE, color_negative=RED, signed=True):
    dff = df_plot.copy()

    if signed:
        colors = [color_positive if v >= 0 else color_negative for v in dff[value_col]]
        vals = dff[value_col].values
    else:
        colors = [BLUE for _ in range(len(dff))]
        vals = dff[value_col].values

    hover_text = []
    for _, r in dff.iterrows():
        feat = str(r["feature"])
        val = r[value_col]
        hover_text.append(
            f"<b>{feat}</b><br>"
            + (f"값: {val:+.5f}" if signed else f"중요도: {val:.5f}")
        )

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=vals[::-1],
        y=[make_short_label(v) for v in dff["feature"].iloc[::-1]],
        orientation="h",
        marker=dict(color=colors[::-1]),
        hovertext=hover_text[::-1],
        hovertemplate="%{hovertext}<extra></extra>"
    ))

    fig.update_layout(
        template="plotly_white",
        height=320,
        margin=dict(l=10, r=20, t=40, b=10),
        title=dict(text=title_text, font=dict(size=18, color=TEXT, family="Arial")),
        xaxis=dict(
            title="SHAP value" if signed else "Importance",
            tickfont=dict(size=14),
            title_font=dict(size=15)
        ),
        yaxis=dict(
            title="",
            tickfont=dict(size=13),
            automargin=True
        ),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)", # 차트 배경 투명 적용
        showlegend=False,
        hoverlabel=dict(font_size=14, font_family="Arial", bgcolor="white", align="left")
    )
    return fig


# =========================================================
# 5) Load
# =========================================================
if not os.path.exists(CSV_PATH):
    st.error(f"CSV 경로를 찾지 못했습니다:\n{CSV_PATH}\n\n데이터 파일을 확인해주세요.")
    st.stop()

if not os.path.exists(MODEL_PKL_PATH):
    st.error(f"PKL 경로를 찾지 못했습니다:\n{MODEL_PKL_PATH}\n\n모델 파일을 확인해주세요.")
    st.stop()

df_raw = pd.read_csv(CSV_PATH)

need = [GROUP_COL, CYCLE_COL, TARGET_COL] + BASE_FEATURES
miss = [c for c in need if c not in df_raw.columns]
if miss:
    st.error(f"CSV에 필요한 컬럼이 없습니다: {miss}")
    st.stop()

for c in [CYCLE_COL, TARGET_COL] + BASE_FEATURES:
    df_raw[c] = pd.to_numeric(df_raw[c], errors="coerce")

df_raw = (
    df_raw[df_raw[GROUP_COL].isin(["B0005", "B0006", "B0007", "B0018"])]
    .dropna(subset=need)
    .copy()
)

df_raw = (
    df_raw.groupby([GROUP_COL, CYCLE_COL], as_index=False)[BASE_FEATURES + [TARGET_COL]]
    .mean()
    .sort_values([GROUP_COL, CYCLE_COL])
    .reset_index(drop=True)
)

with open(MODEL_PKL_PATH, "rb") as f:
    payload = pickle.load(f)

model = payload["model"]
FEATURES = payload.get("features", BASE_FEATURES)

need_ts = any(("_lag" in c) or ("_diff" in c) or ("_rm" in c) for c in FEATURES)
df = df_raw.copy()

if need_ts:
    df = add_ts_features(df, BASE_FEATURES, GROUP_COL, CYCLE_COL, lag=1, roll_w=5)

missing_feat = [c for c in FEATURES if c not in df.columns]
if missing_feat:
    st.error("모델이 요구하는 feature가 데이터에 없습니다.\n" + "\n".join(missing_feat[:20]))
    st.stop()

df = df.dropna(subset=[GROUP_COL, CYCLE_COL, TARGET_COL] + FEATURES).copy()
df = df.sort_values([GROUP_COL, CYCLE_COL]).reset_index(drop=True)
BATTS = sorted(df[GROUP_COL].unique().tolist())

if len(BATTS) == 0:
    st.error("사용 가능한 배터리 데이터가 없습니다.")
    st.stop()


# =========================================================
# 6) Title
# =========================================================
st.markdown(
    """
    <div class="sec-header-banner">
        <h1>품질 의사결정 시스템</h1>
        <p>모델 성능 모니터링과 SHAP 기반 열화 영향 요인 분석</p>
    </div>
    """,
    unsafe_allow_html=True
)


# =========================================================
# 7) Main layout (★ 구역별 st.container(border=True) 묶음 적용)
# =========================================================
left, right = st.columns([1.12, 1.0], gap="large")

with left:
    # 묶음 1: 모델 모니터링 전체
    with st.container(border=True):
        
        st.markdown("<div class='sec-title'>모델 모니터링</div>", unsafe_allow_html=True)

        ctlL, ctlR = st.columns([0.55, 0.45], gap="medium")
        with ctlL:
            sel_batt = st.selectbox("Battery 선택", BATTS, index=0, key="batt_main")

        sub_all = df[df[GROUP_COL] == sel_batt].sort_values(CYCLE_COL).copy()
        cycles_all = sub_all[CYCLE_COL].astype(int).tolist()

        if len(cycles_all) == 0:
            st.warning("선택한 배터리에 표시할 데이터가 없습니다.")
            st.stop()

        min_c, max_c = int(min(cycles_all)), int(max(cycles_all))

        with ctlR:
            end_cycle = st.slider(
                "사이클 수 조절",
                min_value=min_c,
                max_value=max_c,
                value=min(max_c, min_c + 40),
                step=1,
                key="cycle_main"
            )

        sub = sub_all[sub_all[CYCLE_COL].astype(int) <= int(end_cycle)].copy().sort_values(CYCLE_COL)
        X = sub[FEATURES]
        y = sub[TARGET_COL].values
        pred = model.predict(X)
        err = pred - y
        abs_err = np.abs(err)

        last_n = 10
        tail_idx = max(0, len(sub) - last_n)
        sub10 = sub.iloc[tail_idx:].copy()
        pred10 = pred[tail_idx:]
        y10 = y[tail_idx:]
        err10 = err[tail_idx:]
        abs10 = np.abs(err10)

        mae10 = float(np.mean(abs10)) if len(abs10) else np.nan
        rmse10 = float(np.sqrt(np.mean(err10 ** 2))) if len(err10) else np.nan
        mape10 = float(np.mean(np.abs(err10 / np.clip(y10, 1e-6, None)))) * 100.0 if len(err10) else np.nan

        rolling_mae = pd.Series(abs_err).rolling(10, min_periods=3).mean().to_numpy()
        ew = ewma(abs_err, alpha=0.3)

        z = (mae10 - BASELINE_MAE) / (MAE_STD + 1e-12) if np.isfinite(mae10) else 0.0

        if abs(z) >= 3:
            status_html = f"<span class='status-pill-bad'>최근 오차 경고</span>"
        elif 1 <= abs(z) < 2:
            status_html = f"<span class='status-pill-warn'>최근 오차 주의</span>"
        else:
            status_html = f"<span class='status-pill-good'>최근 오차 안정</span>"

        # ★ KPI 박스 (세로 크기 대폭 줄인 디자인)
        k1, k2, k3 = st.columns(3, gap="small")
        with k1:
            st.markdown(f'''
            <div class="kpi-wrapper">
                <div class="kpi-title">MAE</div>
                <div class="kpi-value">{mae10:.5f}</div>
            </div>''', unsafe_allow_html=True)
        with k2:
            st.markdown(f'''
            <div class="kpi-wrapper">
                <div class="kpi-title">RMSE</div>
                <div class="kpi-value">{rmse10:.5f}</div>
            </div>''', unsafe_allow_html=True)
        with k3:
            st.markdown(f'''
            <div class="kpi-wrapper">
                <div class="kpi-title">MAPE</div>
                <div class="kpi-value">{mape10:.5f}%</div>
            </div>''', unsafe_allow_html=True)

        st.markdown("<div style='height: 5px;'></div>", unsafe_allow_html=True)

        L, R = st.columns([0.75, 0.25], gap="medium")

        with L:
            # ★ 오차 추이 테이블 옆에 i 툴팁 정보 버튼 추가 (오른쪽으로 펼쳐지게 수정됨)
            tooltip_html = """
            <span class="info-tip">i
                <span class="tooltip-text">
                    <b>MAE (Mean Absolute Error)</b><br>최근 10사이클의 예측 오차 |예측−실제| 의 평균입니다.<br><br>
                    <b>RMSE (Root Mean Squared Error)</b><br>최근 10사이클 오차를 제곱 평균 후 제곱근 처리한 값입니다. 큰 오차에 더 민감합니다.<br><br>
                    <b>MAPE (Mean Absolute Percentage Error)</b><br>실제값 대비 상대 오차(%)의 평균입니다.<br><br>
                    <b>Rolling MAE</b><br>최근 N개 오차의 단순 평균으로 최근 흐름을 부드럽게 보여줍니다.<br><br>
                    <b>EWMA</b><br>최근 오차에 더 큰 가중치를 줘서 최근 이상 징후를 더 빠르게 반영합니다.
                </span>
            </span>
            """
            st.markdown(f"<div style='font-size:18px; font-weight:800; color:#111; margin-bottom:10px; display:flex; align-items:center;'>오차 추이 테이블 (최근 10사이클) {tooltip_html}</div>", unsafe_allow_html=True)

            roll10 = rolling_mae[tail_idx:]
            ew10 = ew[tail_idx:]

            tbl = pd.DataFrame({
                "cycle": sub10[CYCLE_COL].astype(int).values,
                "pred_SOH": np.round(pred10, 5),
                "actual_SOH": np.round(y10, 5),
                "abs_err": np.round(abs10, 5),
                "Rolling_MAE": np.round(roll10, 5),
                "EWMA": np.round(ew10, 5),
            }).reset_index(drop=True)

            st.dataframe(tbl, use_container_width=True, height=250)

        with R:
            # ★ 모델 상태 마진 추가해서 위치 하향 조정
            st.markdown("<div style='font-size:18px; font-weight:800; color:#111; margin-bottom:10px; margin-top:40px;'>모델 상태</div>", unsafe_allow_html=True)
            st.markdown(status_html, unsafe_allow_html=True)

with left:
    # 묶음 2: 시나리오 구성 & 전후 비교
    with st.container(border=True):
        L2, R2 = st.columns([0.46, 0.54], gap="large")

        with L2:
            st.markdown("<div class='sec-title'>시나리오 구성</div>", unsafe_allow_html=True)

            preset = st.radio(
                "preset",
                [
                    "Preset 1 — Thermal Stress 반복",
                    "Preset 2 — Voltage sag 악화 반복",
                    "Preset 3 — Data Quality 저하",
                ],
                label_visibility="collapsed"
            )
            st.markdown("<div style='height:12px;'></div>", unsafe_allow_html=True)
            a, b = st.columns(2)
            with a:
                n_events = st.slider("이상 이벤트 횟수", 0, 20, 6, 1)
            with b:
                severity = st.slider("이상 강도", 0.0, 3.0, 1.5, 0.1)

        with R2:
            st.markdown("<div class='sec-title'>시나리오 전/후 SOH 예측 비교</div>", unsafe_allow_html=True)

            sdf = df[df[GROUP_COL] == sel_batt].sort_values(CYCLE_COL).copy()
            sdf = sdf[sdf[CYCLE_COL].astype(int) <= int(end_cycle)].copy()
            cycles = sdf[CYCLE_COL].astype(int).values

            base_line = model.predict(sdf[FEATURES])

            scen_df = scenario_apply(
                sdf.copy(),
                preset,
                n_events=int(n_events),
                severity=float(severity),
                seed=42
            )
            scen_line = model.predict(scen_df[FEATURES])

            delta = base_line - scen_line
            max_idx = int(np.argmax(delta)) if len(delta) else 0
            max_drop = float(delta[max_idx]) if len(delta) else 0.0
            max_cycle = int(cycles[max_idx]) if len(cycles) else 0
            y0 = float(base_line[max_idx]) if len(base_line) else 0.0
            y1 = float(scen_line[max_idx]) if len(scen_line) else 0.0

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=cycles, y=base_line, mode="lines", name="시나리오 전",
                line=dict(color=BLUE, width=4), # 남색 적용
                hoverlabel=dict(font_size=17)
            ))
            fig.add_trace(go.Scatter(
                x=cycles, y=scen_line, mode="lines", name="시나리오 후",
                line=dict(color=RED, width=4, dash="dash"),
                hoverlabel=dict(font_size=17)
            ))

            if len(cycles):
                fig.add_annotation(
                    x=max_cycle, y=y0, ax=max_cycle, ay=y1,
                    showarrow=True, arrowhead=3, arrowwidth=3, arrowcolor=RED
                )
                fig.add_annotation(
                    x=max_cycle,
                    y=(y0 + y1) / 2,
                    text=f"<b>cycle {max_cycle}</b><br>-{max_drop:.4f}",
                    showarrow=False,
                    font=dict(color=RED, size=14),
                    bgcolor="#ffffff", # 화이트 박스
                    bordercolor=RED,
                    borderwidth=1,
                    borderpad=6
                )
                fig.add_trace(go.Scatter(
                    x=[max_cycle],
                    y=[(y0 + y1) / 2],
                    mode="markers",
                    marker=dict(size=16, color="rgba(255,0,0,0)"),
                    showlegend=False,
                    hovertemplate=f"최대 감소 지점<br>cycle: {max_cycle}<br>감소량: {max_drop:.4f}<extra></extra>"
                ))

            # ★ 시나리오 전후 예측 차트 배경을 탭2처럼 완전한 흰색으로 변경 ★
            fig.update_layout(
                template="plotly_white",
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
                xaxis_title="cycle_id",
                yaxis_title="SOH",
                xaxis=dict(tickfont=dict(size=14), title_font=dict(size=15)),
                yaxis=dict(tickfont=dict(size=14), title_font=dict(size=15)),
                plot_bgcolor="white",  # 투명 -> 흰색 배경 적용
                paper_bgcolor="white", # 투명 -> 흰색 배경 적용
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1, font=dict(size=14)),
                hoverlabel=dict(
                    font_size=15,
                    font_family="Arial",
                    bgcolor="white"
                )
            )
            st.plotly_chart(fig, use_container_width=True)

with right:
    # 묶음 3: 알람과 조치
    with st.container(border=True):
        top_head_l, top_head_r = st.columns([0.65, 0.35], gap="small")
        with top_head_l:
            st.markdown("<div class='sec-title'>알람과 조치</div>", unsafe_allow_html=True)
        with top_head_r:
            decision = st.selectbox(
                "판정 모드 선택",
                ["AUTO", "REVIEW", "OFF"],
                index=0,
                key="decision_select"
            )

        st.markdown("<div style='height:5px;'></div>", unsafe_allow_html=True)

        if decision == "AUTO":
            st.markdown(
                """
                <div class="action-good">
                  <span style="font-size: 18px; color: #00A650;"><b>✅ AUTO ON — 자동 판정 허용</b></span><br/><br/>
                  • 현재 상태가 안정적이면 자동 판정을 유지합니다.<br/>
                  • 오차가 상승하면 REVIEW 전환 후 확인을 권고합니다.<br/>
                  • 권장: 알림 로그 주기 확인 및 기준선 재평가
                </div>
                """,
                unsafe_allow_html=True
            )
        elif decision == "REVIEW":
            st.markdown(
                """
                <div class="action-warn">
                  <span style="font-size: 18px; color: #F08C00;"><b>⚠️ REVIEW — 사람 확인 필요</b></span><br/><br/>
                  • 추가 검사: <b>임피던스 측정 권고</b><br/>
                  • 검증 사이클 추가 수행 권고 (예: <b>+10 cycles</b>)<br/>
                  • 담당자 알림 전송 대기 중
                </div>
                """,
                unsafe_allow_html=True
            )
            st.toast("사람 확인 필요가 선택되었습니다. 모델 담당자에게 자동 연락됩니다.", icon="📩")
        else:
            st.markdown(
                """
                <div class="action-bad">
                  <span style="font-size: 18px; color: #E6192B;"><b>⛔ OFF — 자동판정 금지</b></span><br/><br/>
                  • 시스템 제어에 개입하여 자동판정 비활성화<br/>
                  • 데이터 수집 시스템 점검 (센서 / 컷오프 이슈 파악)<br/>
                  • 필요 시 재측정/재현시험으로 원인 분리 권고
                </div>
                """,
                unsafe_allow_html=True
            )

with right:
    # 묶음 4: SHAP 및 특성 중요도 분석
    with st.container(border=True):
        st.markdown("<div class='sec-title'>SHAP 및 특성 중요도 분석</div>", unsafe_allow_html=True)
        
        tmp = df[df[GROUP_COL] == sel_batt].sort_values(CYCLE_COL).copy()
        tmp = tmp[tmp[CYCLE_COL].astype(int) <= int(end_cycle)].copy()

        if len(tmp) == 0:
            st.warning("분석용 데이터가 없습니다.")
            st.stop()

        pick_cycle = int(tmp[CYCLE_COL].astype(int).max())
        row = tmp[tmp[CYCLE_COL].astype(int) == pick_cycle].copy()

        A1, A2 = st.columns(2, gap="large")

        with A1:
            st.markdown("<div style='font-size:18px; font-weight:800; color:#111; margin-bottom:5px;'>SHAP 분석</div>", unsafe_allow_html=True)
            x_row = row[FEATURES]
            pred_row = float(model.predict(x_row)[0])

            shap_vals = None
            prep, ridge = None, None

            try:
                prep = model.named_steps.get("prep", None)
                ridge = model.named_steps.get("model", None)
            except Exception:
                prep, ridge = None, None

            if SHAP_OK and prep is not None and ridge is not None:
                try:
                    bg = tmp.head(200)
                    X_bg = prep.transform(bg[FEATURES])
                    X_one = prep.transform(x_row)
                    explainer = shap.LinearExplainer(ridge, X_bg, feature_perturbation="interventional")
                    shap_vals = explainer.shap_values(X_one)[0]
                except Exception:
                    shap_vals = None

            if shap_vals is None and prep is not None and ridge is not None:
                try:
                    X_one = prep.transform(x_row)
                    X_one = np.asarray(X_one).reshape(-1)
                    coef = np.asarray(ridge.coef_).reshape(-1)
                    shap_vals = coef * X_one
                except Exception:
                    shap_vals = np.zeros(len(FEATURES))

            if shap_vals is None:
                shap_vals = np.zeros(len(FEATURES))

            contrib = pd.DataFrame({"feature": FEATURES, "shap": shap_vals})
            contrib["abs"] = contrib["shap"].abs()
            contrib = contrib.sort_values("abs", ascending=False)
            top_shap = contrib.head(5).copy()

            fig_shap = draw_bar_chart(
                top_shap,
                value_col="shap",
                title_text=f"{sel_batt} | cycle={pick_cycle}",
                color_positive=BLUE,
                color_negative=RED,
                signed=True
            )
            st.plotly_chart(fig_shap, use_container_width=True)

            shap_top3 = contrib.head(3)["feature"].tolist()
            st.markdown("<div style='font-size:15px; font-weight:800; margin-bottom:5px;'>주요 원인 인자 Top 3</div>", unsafe_allow_html=True)
            if len(shap_top3) >= 3:
                chips_html = "".join([feature_chip_html(f) for f in shap_top3[:3]])
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.write("표시할 컬럼이 부족합니다.")

        with A2:
            st.markdown("<div style='font-size:18px; font-weight:800; color:#111; margin-bottom:5px;'>Feature Importance 분석</div>", unsafe_allow_html=True)

            imp = get_cycle_adjusted_importance(model, tmp, FEATURES)
            top_imp = imp.head(5).copy()

            fig_imp = draw_bar_chart(
                top_imp,
                value_col="importance",
                title_text="선택된 Cycle 구간 전체의 중요도",
                signed=False
            )
            st.plotly_chart(fig_imp, use_container_width=True)

            imp_top3 = imp.head(3)["feature"].tolist()
            st.markdown("<div style='font-size:15px; font-weight:800; margin-bottom:5px;'>주요 연구 대상 Top 3</div>", unsafe_allow_html=True)
            if len(imp_top3) >= 3:
                chips_html = "".join([feature_chip_html(f) for f in imp_top3[:3]])
                st.markdown(chips_html, unsafe_allow_html=True)
            else:
                st.write("표시할 컬럼이 부족합니다.")