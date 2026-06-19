import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Page configuration
st.set_page_config(page_title="OLED Gamma Tuning", layout="wide")

# ==========================================
# Login Authentication Logic
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False

if not st.session_state['logged_in']:
    st.title("Login")
    st.markdown("Please login to access the OLED Gamma Tuning App.")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        with st.form("login_form"):
            st.image("yitoa.png", width=150)
            id_input = st.text_input("ID")
            pw_input = st.text_input("Password", type="password")
            submit_button = st.form_submit_button("Login")

            if submit_button:
                is_valid_domain = id_input.endswith("@yitoa.co.jp") or id_input.endswith("@yitoa.com")
                is_valid_password = (pw_input == id_input) and (pw_input != "")

                if is_valid_domain and is_valid_password:
                    st.session_state['logged_in'] = True
                    st.rerun()
                else:
                    st.error("Invalid ID or password.")
    st.stop()  # Stop execution until logged in

# ==========================================
# Main Application Logic
# ==========================================
def process_gamma_tuning(measured_dac, measured_luminance, target_gamma=1.0):
    """OLED Gamma register calculation logic"""
    x_points = np.array([
        0, 4, 8, 16, 32, 48, 64, 96, 128, 160, 192, 256, 320,
        384, 448, 512, 576, 640, 704, 768, 832, 896, 960, 1008, 1023
    ])
    x_points_8bit = x_points / 1023.0 * 255.0
    
    y_unadj_points = np.round((x_points / 1023.0) * 4095).astype(int)

    max_luminance = np.max(measured_luminance)
    target_luminance_cp = ((x_points / 1023.0) ** target_gamma) * max_luminance
    
    sort_idx = np.argsort(measured_luminance)
    meas_lum_sorted = np.array(measured_luminance)[sort_idx]
    meas_dac_sorted = np.array(measured_dac)[sort_idx]
    
    y_points = np.interp(target_luminance_cp, meas_lum_sorted, meas_dac_sorted)
    y_points = np.round(y_points).clip(0, 4095).astype(int)
    
    x_continuous = np.linspace(0, 1023, 1024)
    x_continuous_8bit = x_continuous / 1023.0 * 255.0  
    
    target_luminance_continuous = ((x_continuous / 1023.0) ** target_gamma) * max_luminance
    y_continuous = np.interp(x_continuous, x_points, y_points)
    adjusted_luminance_continuous = np.interp(y_continuous, meas_dac_sorted, meas_lum_sorted)
    
    measured_dac_continuous = x_continuous * 4
    measured_luminance_continuous = np.interp(measured_dac_continuous, meas_dac_sorted, meas_lum_sorted)

    def calc_gamma_curve(lum_array, x_array_10bit):
        with np.errstate(divide='ignore', invalid='ignore'):
            g = np.log(lum_array / max_luminance) / np.log(x_array_10bit / 1023.0)
        return g

    gamma_meas = calc_gamma_curve(measured_luminance_continuous, x_continuous)
    gamma_adj = calc_gamma_curve(adjusted_luminance_continuous, x_continuous)
    gamma_target = np.full_like(x_continuous, target_gamma)
    
    lum_cp_adj = np.interp(y_points, meas_dac_sorted, meas_lum_sorted)
    gamma_cp = calc_gamma_curve(lum_cp_adj, x_points)
    
    reg_names = [f"CP{i:02d}" for i in range(len(x_points))]

    return {
        "x_reg_name": "X Reg (10-bit)", "y_reg_name": "Y Reg (12-bit)",
        "x_points": x_points, "x_points_8bit": x_points_8bit, "reg_names": reg_names,
        "y_points": y_points, "y_unadj_points": y_unadj_points,
        "max_lum": max_luminance, "target_lum_cp": target_luminance_cp, "gamma_cp": gamma_cp,
        "x_cont_8bit": x_continuous_8bit,
        "lum_meas": measured_luminance_continuous, "lum_adj": adjusted_luminance_continuous, "lum_tgt": target_luminance_continuous,
        "gam_meas": gamma_meas, "gam_adj": gamma_adj, "gam_tgt": gamma_target
    }

# ==========================================
# Streamlit UI Construction
# ==========================================
# サイドバーの設定
with st.sidebar:
    col1, col2, col3 = st.columns([1, 10, 1])
    with col2:
        # ワーニングを回避するため、固定幅で指定
        st.image("yitoa.png", width=200)
        st.markdown(
            """
            <div style='text-align: center; font-size: 0.85em; color: gray;'>
                Copyright(c) YITOA Technology.<br>
                All rights reserved.
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("---")

    if st.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()
        
    st.markdown("---")

    st.header("1. Basic Settings")
    target_gamma_input = st.number_input("Target Gamma", min_value=1.0, max_value=3.0, value=2.2, step=0.1)
    
    st.markdown("---")
    st.subheader("2. Upload CSV File(s)")
    st.markdown("* Required 4-column format: A: Gray, B: x, C: y, D: L (Luminance)")
    uploaded_files = st.file_uploader("Select Measurement Data (CSV)", type=["csv"], accept_multiple_files=True)
    
    st.markdown("---")
    st.header("3. Graph Display Settings")
    display_mode = st.radio("Luminance Mode", ("Absolute (nits)", "Normalized (0.0-1.0)"))
    is_normalized = display_mode.startswith("Norm")
    
    color_x_axis = st.radio("Color Tracking X-Axis", ("Grayscale", "Luminance"))
    
    # ---------------------------------------------------------
    # Pre-scan for dynamic Y-axis defaults (Luminance & L diff)
    # ---------------------------------------------------------
    pre_lum_min = 0.0
    pre_lum_max = 500.0
    pre_ldiff_min = -5.0
    pre_ldiff_max = 50.0

    if uploaded_files:
        tmp_lums = []
        tmp_ldiffs = []
        for f in uploaded_files:
            try:
                tmp_df = pd.read_csv(f, header=None)
                f.seek(0) # 本処理用にポインタをリセット
                tmp_df_numeric = tmp_df.apply(pd.to_numeric, errors='coerce')
                if len(tmp_df.columns) >= 4:
                    tmp_df_clean = tmp_df_numeric.dropna(subset=[0, 1, 2, 3])
                    if not tmp_df_clean.empty:
                        m_g = tmp_df_clean.iloc[:, 0].values
                        m_l = tmp_df_clean.iloc[:, 3].values
                        tmp_lums.append(m_l)
                        
                        s_idx = np.argsort(m_g)
                        l_asc = m_l[s_idx]
                        ld = np.diff(l_asc)
                        if len(ld) > 0:
                            tmp_ldiffs.append(ld)
            except Exception:
                f.seek(0)
                
        if tmp_lums:
            all_lums = np.concatenate(tmp_lums)
            pre_lum_min = float(np.nanmin(all_lums))
            pre_lum_max = float(np.nanmax(all_lums))
        if tmp_ldiffs:
            all_lds = np.concatenate(tmp_ldiffs)
            pre_ldiff_min = float(np.nanmin(all_lds))
            pre_ldiff_max = float(np.nanmax(all_lds))
    # ---------------------------------------------------------
    
    st.markdown("#### Y-Axis Scale Adjustment")
    with st.expander("Adjust Graph Y-Axis Range", expanded=False):
        st.caption("Grayscale vs Luminance")
        if is_normalized:
            lum_min_val = -0.05
            lum_max_val = 1.05
        else:
            margin = (pre_lum_max - pre_lum_min) * 0.05 if pre_lum_max > pre_lum_min else 10.0
            lum_min_val = float(pre_lum_min - margin)
            lum_max_val = float(pre_lum_max + margin)
            if lum_max_val <= lum_min_val:
                lum_max_val = lum_min_val + 10.0

        y_lum_min = st.number_input("Luminance Min", value=lum_min_val, step=0.1 if is_normalized else 10.0)
        y_lum_max = st.number_input("Luminance Max", value=lum_max_val, step=0.1 if is_normalized else 10.0)
        
        st.caption("Grayscale vs Gamma Value")
        y_gam_min = st.number_input("Gamma Min", value=2.0, step=0.1)
        y_gam_max = st.number_input("Gamma Max", value=2.4, step=0.1)
        
        st.caption("Grayscale vs Δu'v' (Δduv)")
        y_duv_min = st.number_input("Δduv Min", value=-0.005, step=0.001, format="%.4f")
        y_duv_max = st.number_input("Δduv Max", value=0.020, step=0.001, format="%.4f")
        
        st.caption("Grayscale vs ΔCCT")
        y_cct_min = st.number_input("ΔCCT Min", value=0, step=500)
        y_cct_max = st.number_input("ΔCCT Max", value=2000, step=500)
        
        st.caption("Grayscale vs Chromaticity (x, y)")
        y_xy_min = st.number_input("Chromaticity Min", value=0.0, step=0.05, format="%.4f")
        y_xy_max = st.number_input("Chromaticity Max", value=0.8, step=0.05, format="%.4f")
        
        st.caption("Grayscale vs L diff")
        ldiff_margin = (pre_ldiff_max - pre_ldiff_min) * 0.1 if pre_ldiff_max > pre_ldiff_min else 5.0
        def_ldiff_min = float(pre_ldiff_min - ldiff_margin)
        def_ldiff_max = float(pre_ldiff_max + ldiff_margin)
        if def_ldiff_max <= def_ldiff_min:
            def_ldiff_max = def_ldiff_min + 10.0
            
        y_ldiff_min = st.number_input("L diff Min", value=def_ldiff_min, step=1.0)
        y_ldiff_max = st.number_input("L diff Max", value=def_ldiff_max, step=1.0)

# メインコンテンツエリアの構築
st.title("OLED Gamma Tuning")
st.markdown("Upload multiple measured CSV data files (4-column format) to automatically calculate register values and compare characteristics.")

MAX_DAC = 4095.0
parsed_data = {}

# Data loading process (複数ファイルループ)
if uploaded_files:
    for f in uploaded_files:
        try:
            df = pd.read_csv(f, header=None)
            df_numeric = df.apply(pd.to_numeric, errors='coerce')
            if len(df.columns) >= 4:
                df_clean = df_numeric.dropna(subset=[0, 1, 2, 3])
                if not df_clean.empty:
                    m_gray = df_clean.iloc[:, 0].values
                    m_dac = (m_gray / 255.0) * MAX_DAC
                    m_x = df_clean.iloc[:, 1].values
                    m_y = df_clean.iloc[:, 2].values
                    m_lum = df_clean.iloc[:, 3].values
                    
                    denom = -2 * m_x + 12 * m_y + 3
                    valid_xy = denom != 0
                    u_p = np.zeros_like(m_x)
                    v_p = np.zeros_like(m_y)
                    u_p[valid_xy] = (4 * m_x[valid_xy]) / denom[valid_xy]
                    v_p[valid_xy] = (9 * m_y[valid_xy]) / denom[valid_xy]

                    n = (m_x - 0.3320) / (0.1858 - m_y)
                    cct = 449 * (n**3) + 3525 * (n**2) + 6823.3 * n + 5520.33

                    ref_idx = np.argmax(m_gray) 
                    u_ref = u_p[ref_idx]
                    v_ref = v_p[ref_idx]
                    cct_ref = cct[ref_idx]

                    d_cct = np.sqrt((cct - cct_ref)**2)
                    d_uv = np.sqrt((u_p - u_ref)**2 + (v_p - v_ref)**2)

                    l_max = np.max(m_lum)
                    g_norm = m_gray / 255.0
                    c_gamma = np.zeros_like(m_gray, dtype=float)
                    val_lum = (m_gray > 0) & (m_lum > 0)
                    
                    with np.errstate(divide='ignore', invalid='ignore'):
                        c_gamma[val_lum] = np.log(m_lum[val_lum] / l_max) / np.log(g_norm[val_lum])
                    c_gamma[~val_lum] = np.nan

                    res = process_gamma_tuning(m_dac, m_lum, target_gamma=target_gamma_input)

                    parsed_data[f.name] = {
                        "meas_gray": m_gray, "meas_dac": m_dac, "meas_x": m_x, "meas_y": m_y, "meas_lum": m_lum,
                        "u_prime": u_p, "v_prime": v_p, "cct": cct, "delta_cct": d_cct, "delta_uv": d_uv,
                        "calc_gamma": c_gamma, "res": res
                    }
                else:
                    st.error(f"No valid numeric data found in {f.name}.")
            else:
                st.error(f"Error: {f.name} has fewer than 4 columns. Please use [Gray, x, y, L] format.")
        except Exception as e:
            st.error(f"Failed to load CSV {f.name}: {e}")
else:
    st.info("👈 Please upload CSV file(s) from the sidebar. Currently displaying demo data.")
    m_gray = np.linspace(255, 0, 256)
    m_dac = (m_gray / 255.0) * MAX_DAC
    m_lum = (m_gray / 255.0) ** 1.0 * 500.0 
    m_x = np.full_like(m_gray, 0.3127) 
    m_y = np.full_like(m_gray, 0.3290) 

    res = process_gamma_tuning(m_dac, m_lum, target_gamma=target_gamma_input)
    
    denom = -2 * m_x + 12 * m_y + 3
    valid_xy = denom != 0
    u_p = np.zeros_like(m_x)
    v_p = np.zeros_like(m_y)
    u_p[valid_xy] = (4 * m_x[valid_xy]) / denom[valid_xy]
    v_p[valid_xy] = (9 * m_y[valid_xy]) / denom[valid_xy]
    n = (m_x - 0.3320) / (0.1858 - m_y)
    cct = 449 * (n**3) + 3525 * (n**2) + 6823.3 * n + 5520.33
    ref_idx = np.argmax(m_gray) 
    u_ref = u_p[ref_idx]
    v_ref = v_p[ref_idx]
    cct_ref = cct[ref_idx]
    d_cct = np.sqrt((cct - cct_ref)**2)
    d_uv = np.sqrt((u_p - u_ref)**2 + (v_p - v_ref)**2)

    l_max = np.max(m_lum)
    g_norm = m_gray / 255.0
    c_gamma = np.zeros_like(m_gray, dtype=float)
    val_lum = (m_gray > 0) & (m_lum > 0)
    with np.errstate(divide='ignore', invalid='ignore'):
        c_gamma[val_lum] = np.log(m_lum[val_lum] / l_max) / np.log(g_norm[val_lum])
    c_gamma[~val_lum] = np.nan

    parsed_data["Demo Data"] = {
        "meas_gray": m_gray, "meas_dac": m_dac, "meas_x": m_x, "meas_y": m_y, "meas_lum": m_lum,
        "u_prime": u_p, "v_prime": v_p, "cct": cct, "delta_cct": d_cct, "delta_uv": d_uv,
        "calc_gamma": c_gamma, "res": res
    }

# Graph plotting
if parsed_data:
    st.header("Tuning Curves & Color Tracking")
    
    # 複数ファイル入力時のグラフ表示選択
    display_files = list(parsed_data.keys())
    if len(parsed_data) > 1:
        display_files = st.multiselect(
            "Select files to display on graphs:", 
            options=list(parsed_data.keys()), 
            default=list(parsed_data.keys())
        )

    st.markdown("##### Toggle Graph Display")
    col_t1, col_t2, col_t3, col_t4 = st.columns(4)
    show_meas = col_t1.checkbox("Measured", value=True)
    show_target = col_t2.checkbox("Target", value=True)
    show_adj = col_t3.checkbox("Adjusted", value=True)
    show_cp = col_t4.checkbox("CP Points", value=True)
    
    color_x_label = "Luminance" if color_x_axis == "Luminance" else "Grayscale"
    
    # サブプロットを3行2列に変更
    fig = make_subplots(
        rows=3, cols=2, 
        subplot_titles=(
            "Grayscale vs Luminance", 
            f"{color_x_label} vs Gamma Value", 
            f"{color_x_label} vs Δu'v' (Δduv)", 
            f"{color_x_label} vs ΔCCT", 
            f"{color_x_label} vs Chromaticity (x, y)",
            f"{color_x_label} vs L diff"
        ),
        vertical_spacing=0.1 
    )
    
    y_title = 'Normalized Luminance' if is_normalized else 'Luminance (nits)'
    
    plotly_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

    for i, (fname, d) in enumerate(parsed_data.items()):
        # 選択されていないファイルはグラフ描画からスキップ
        if fname not in display_files:
            continue
            
        c = plotly_colors[(i * 2) % len(plotly_colors)]
        c_tgt = plotly_colors[(i * 2 + 1) % len(plotly_colors)]
        
        res = d["res"]
        scale_div = res['max_lum'] if is_normalized else 1.0
        num_points = len(res['x_points']) 

        if show_meas:
            fig.add_trace(go.Scatter(x=res['x_cont_8bit'], y=res['lum_meas']/scale_div, name=f"Meas ({fname})", line=dict(dash='dash', color=c), opacity=0.6), row=1, col=1)
            fig.add_trace(go.Scatter(x=d["meas_gray"], y=d["meas_lum"]/scale_div, mode='markers', name=f"Points ({fname})", marker=dict(color=c, size=5, symbol='circle'), showlegend=False), row=1, col=1)
        
        if show_target:
            fig.add_trace(go.Scatter(x=res['x_cont_8bit'], y=res['lum_tgt']/scale_div, name=f"Target ({fname})", line=dict(color=c_tgt, width=2)), row=1, col=1)
        
        if show_adj:
            fig.add_trace(go.Scatter(x=res['x_cont_8bit'], y=res['lum_adj']/scale_div, name=f"Adj ({fname})", line=dict(dash='dot', color=c, width=2)), row=1, col=1)
        
        if show_cp:
            hover_text_lum = [f"{fname} - {res['reg_names'][k]}<br>{res['x_reg_name']}: {res['x_points'][k]}<br>Before: {res['y_unadj_points'][k]}<br>After: {res['y_points'][k]}" for k in range(num_points)]
            fig.add_trace(go.Scatter(
                x=res['x_points_8bit'], y=res['target_lum_cp']/scale_div, mode='markers', name=f"CP ({fname})",
                marker=dict(color=c, size=8, symbol='diamond'), text=hover_text_lum, hovertemplate="%{text}<br>Data: %{x:.1f}<br>Lum: %{y:.4f}<extra></extra>"
            ), row=1, col=1)

        # Gamma Value グラフのX軸データ動的切り替え
        x_cont_gam = res['lum_meas']/scale_div if color_x_axis == "Luminance" else res['x_cont_8bit']
        x_pts_gam = d["meas_lum"]/scale_div if color_x_axis == "Luminance" else d["meas_gray"]
        x_tgt_gam = res['lum_tgt']/scale_div if color_x_axis == "Luminance" else res['x_cont_8bit']
        x_adj_gam = res['lum_adj']/scale_div if color_x_axis == "Luminance" else res['x_cont_8bit']
        x_cp_gam = res['target_lum_cp']/scale_div if color_x_axis == "Luminance" else res['x_points_8bit']

        if show_meas:
            fig.add_trace(go.Scatter(x=x_cont_gam, y=res['gam_meas'], name=f"Meas Gam ({fname})", line=dict(dash='dash', color=c), opacity=0.6, showlegend=False), row=1, col=2)
            
            # Gamma points hover (Luminance時にGray情報を付与)
            if color_x_axis == "Luminance":
                ht_gam_pts = fname + "<br>Lum: %{x:.4f}<br>Gray: %{customdata:.0f}<br>Gamma: %{y:.3f}<extra></extra>"
                cd_gam_pts = d["meas_gray"]
            else:
                ht_gam_pts = fname + "<br>Gray: %{x:.0f}<br>Gamma: %{y:.3f}<extra></extra>"
                cd_gam_pts = None
            fig.add_trace(go.Scatter(x=x_pts_gam, y=d["calc_gamma"], mode='markers', marker=dict(color=c, size=5, symbol='circle'), customdata=cd_gam_pts, hovertemplate=ht_gam_pts, showlegend=False), row=1, col=2)
            
        if show_target:
            fig.add_trace(go.Scatter(x=x_tgt_gam, y=res['gam_tgt'], name=f"Tgt Gam ({fname})", line=dict(color=c_tgt, width=2), showlegend=False), row=1, col=2)
        if show_adj:
            fig.add_trace(go.Scatter(x=x_adj_gam, y=res['gam_adj'], name=f"Adj Gam ({fname})", line=dict(dash='dot', color=c, width=2), showlegend=False), row=1, col=2)
        if show_cp:
            hover_text_gam = [f"{fname} - {res['reg_names'][k]}<br>{res['x_reg_name']}: {res['x_points'][k]}<br>Before: {res['y_unadj_points'][k]}<br>After: {res['y_points'][k]}" for k in range(num_points)]
            
            # CP points hover (Luminance時にGray情報を付与)
            if color_x_axis == "Luminance":
                ht_cp_gam = "%{text}<br>Lum: %{x:.4f}<br>Gray: %{customdata:.1f}<br>Gamma: %{y:.3f}<extra></extra>"
                cd_cp_gam = res['x_points_8bit']
            else:
                ht_cp_gam = "%{text}<br>Gray: %{x:.1f}<br>Gamma: %{y:.3f}<extra></extra>"
                cd_cp_gam = None
            fig.add_trace(go.Scatter(
                x=x_cp_gam, y=res['gamma_cp'], mode='markers', name=f"CP Gam ({fname})",
                marker=dict(color=c, size=8, symbol='diamond'), text=hover_text_gam, customdata=cd_cp_gam, hovertemplate=ht_cp_gam, showlegend=False
            ), row=1, col=2)

        color_x_data = d["meas_lum"] / scale_div if color_x_axis == "Luminance" else d["meas_gray"]
        
        # 2行目・3行目のグラフ用ホバーテンプレート (Luminance時にGray情報を付与)
        if color_x_axis == "Luminance":
            c_data = d["meas_gray"]
            ht_duv = fname + "<br>Lum: %{x:.4f}<br>Gray: %{customdata:.0f}<br>Δduv: %{y:.4f}<extra></extra>"
            ht_cct = fname + "<br>Lum: %{x:.4f}<br>Gray: %{customdata:.0f}<br>ΔCCT: %{y:.0f} K<extra></extra>"
            ht_x   = fname + "<br>Lum: %{x:.4f}<br>Gray: %{customdata:.0f}<br>x: %{y:.4f}<extra></extra>"
            ht_y   = fname + "<br>Lum: %{x:.4f}<br>Gray: %{customdata:.0f}<br>y: %{y:.4f}<extra></extra>"
            ht_ld  = fname + "<br>Lum: %{x:.4f}<br>Gray: %{customdata:.0f}<br>L diff: %{y:.3f}<extra></extra>"
        else:
            c_data = None
            ht_duv = fname + "<br>Gray: %{x:.0f}<br>Δduv: %{y:.4f}<extra></extra>"
            ht_cct = fname + "<br>Gray: %{x:.0f}<br>ΔCCT: %{y:.0f} K<extra></extra>"
            ht_x   = fname + "<br>Gray: %{x:.0f}<br>x: %{y:.4f}<extra></extra>"
            ht_y   = fname + "<br>Gray: %{x:.0f}<br>y: %{y:.4f}<extra></extra>"
            ht_ld  = fname + "<br>Gray: %{x:.0f}<br>L diff: %{y:.3f}<extra></extra>"

        if show_meas:
            fig.add_trace(go.Scatter(x=color_x_data, y=d["delta_uv"], mode='lines+markers', name=f"Δduv ({fname})", line=dict(color=c, width=2), marker=dict(size=4), customdata=c_data, hovertemplate=ht_duv, showlegend=False), row=2, col=1)

        if show_meas:
            fig.add_trace(go.Scatter(x=color_x_data, y=d["delta_cct"], mode='lines+markers', name=f"ΔCCT ({fname})", line=dict(color=c, width=2), marker=dict(size=4), customdata=c_data, hovertemplate=ht_cct, showlegend=False), row=2, col=2)

        # Chromaticity (x, y) - X軸変更対応
        if show_meas:
            fig.add_trace(go.Scatter(x=color_x_data, y=d["meas_x"], mode='lines+markers', name=f"x ({fname})", line=dict(color=c, width=2, dash='solid'), marker=dict(size=4), customdata=c_data, hovertemplate=ht_x, showlegend=False), row=3, col=1)
            fig.add_trace(go.Scatter(x=color_x_data, y=d["meas_y"], mode='lines+markers', name=f"y ({fname})", line=dict(color=c, width=2, dash='dash'), marker=dict(size=4), customdata=c_data, hovertemplate=ht_y, showlegend=False), row=3, col=1)
            
            # L diff - X軸変更対応
            temp_sgray = d["meas_gray"]
            temp_slum = d["meas_lum"]
            temp_sort_asc = np.argsort(temp_sgray)
            temp_lum_asc = temp_slum[temp_sort_asc]
            temp_l_diff_asc = np.zeros_like(temp_lum_asc)
            temp_l_diff_asc[1:] = np.diff(temp_lum_asc)
            temp_l_diff_asc[0] = np.nan
            temp_l_diff = np.zeros_like(temp_slum)
            temp_l_diff[temp_sort_asc] = temp_l_diff_asc

            fig.add_trace(go.Scatter(x=color_x_data, y=temp_l_diff, mode='lines+markers', name=f"L diff ({fname})", line=dict(color=c, width=2), marker=dict(size=4), customdata=c_data, hovertemplate=ht_ld, showlegend=False), row=3, col=2)

    # データ量（Grayscaleの最大値）に合わせたX軸の動的範囲計算
    valid_max_vals = [float(np.max(parsed_data[fname]["meas_gray"])) for fname in display_files if fname in parsed_data]
    max_gray_val = max(valid_max_vals) if valid_max_vals else 255.0
                
    gray_x_range = [-5, max_gray_val + 5]
    gray_x_title = f"Input Grayscale (0-{int(max_gray_val)})"

    fig.update_xaxes(title_text=gray_x_title, range=gray_x_range, row=1, col=1)
    
    # 1行目右、2行目、3行目のX軸設定を統合
    if color_x_axis == "Luminance":
        fig.update_xaxes(title_text=y_title, row=1, col=2)
        fig.update_xaxes(title_text=y_title, row=2, col=1)
        fig.update_xaxes(title_text=y_title, row=2, col=2)
        fig.update_xaxes(title_text=y_title, row=3, col=1)
        fig.update_xaxes(title_text=y_title, row=3, col=2)
    else:
        fig.update_xaxes(title_text=gray_x_title, range=gray_x_range, row=1, col=2)
        fig.update_xaxes(title_text=gray_x_title, range=gray_x_range, row=2, col=1)
        fig.update_xaxes(title_text=gray_x_title, range=gray_x_range, row=2, col=2)
        fig.update_xaxes(title_text=gray_x_title, range=gray_x_range, row=3, col=1)
        fig.update_xaxes(title_text=gray_x_title, range=gray_x_range, row=3, col=2)

    fig.update_yaxes(title_text=y_title, row=1, col=1, range=[y_lum_min, y_lum_max])
    fig.update_yaxes(title_text="Gamma Value", row=1, col=2, range=[y_gam_min, y_gam_max])
    fig.update_yaxes(title_text="Δu'v' (Δduv)", row=2, col=1, range=[y_duv_min, y_duv_max])
    fig.update_yaxes(title_text="ΔCCT (K)", row=2, col=2, range=[y_cct_min, y_cct_max])
    fig.update_yaxes(title_text="Color Coordinate (x, y)", row=3, col=1, range=[y_xy_min, y_xy_max])
    fig.update_yaxes(title_text="Luminance Diff", row=3, col=2, range=[y_ldiff_min, y_ldiff_max])
    
    # 高さを1100に維持
    fig.update_layout(height=1100, hovermode="closest", legend=dict(orientation="h", yanchor="bottom", y=1.05, xanchor="center", x=0.5, bgcolor="rgba(255, 255, 255, 0.8)", bordercolor="lightgray", borderwidth=1), margin=dict(t=80))
    
    # ワーニングを回避するため、width="stretch" を使用
    st.plotly_chart(fig, width="stretch")
    st.markdown("---")

    # Tables display
    st.header("Data Tables")
    if len(parsed_data) > 1:
        selected_file = st.selectbox("Select file to view detailed tables:", list(parsed_data.keys()))
    else:
        selected_file = list(parsed_data.keys())[0]
        st.write(f"**Showing data for:** {selected_file}")

    d_selected = parsed_data[selected_file]
    res_sel = d_selected["res"]
    num_pts = len(res_sel['x_points'])

    st.subheader("Gamma Register Settings")
    table_data = []
    for i in range(num_pts):
        x_str = f"{int(res_sel['x_points'][i])} (Fix)" if i in (0, num_pts - 1) else str(int(res_sel['x_points'][i]))
        table_data.append({
            "Control Node": res_sel['reg_names'][i],
            res_sel['x_reg_name']: x_str,
            f"Before: {res_sel['y_reg_name']}": res_sel['y_unadj_points'][i],
            f"After: {res_sel['y_reg_name']}": res_sel['y_points'][i]
        })
    st.dataframe(pd.DataFrame(table_data), width=800, height=400, hide_index=True)
    
    st.subheader("Detailed Measurement Data")
    
    sgray = d_selected["meas_gray"]
    slum = d_selected["meas_lum"]
    sort_asc = np.argsort(sgray)
    lum_asc = slum[sort_asc]
    l_diff_asc = np.zeros_like(lum_asc)
    l_diff_asc[1:] = np.diff(lum_asc)
    l_diff_asc[0] = np.nan
    l_diff = np.zeros_like(slum)
    l_diff[sort_asc] = l_diff_asc

    detailed_df = pd.DataFrame({
        "Gray": sgray, "x": d_selected["meas_x"], "y": d_selected["meas_y"], "L (nits)": slum,
        "u'": d_selected["u_prime"], "v'": d_selected["v_prime"], "Gamma": d_selected["calc_gamma"], "L diff": l_diff,
        "CCT (K)": d_selected["cct"], "ΔCCT": d_selected["delta_cct"], "Δu'v' (Δduv)": d_selected["delta_uv"]
    })

    def highlight_inversion(val):
        if pd.isna(val): return ''
        return 'color: red; font-weight: bold;' if val < 0 else ''

    styled_df = detailed_df.style.format({
        "Gray": "{:.0f}", "x": "{:.4f}", "y": "{:.4f}", "L (nits)": "{:.2f}",
        "u'": "{:.4f}", "v'": "{:.4f}", "Gamma": "{:.3f}", "L diff": "{:.3f}",
        "CCT (K)": "{:.0f}", "ΔCCT": "{:.0f}", "Δu'v' (Δduv)": "{:.4f}"
    }, na_rep="-").map(highlight_inversion, subset=["L diff"])

    # ワーニングを回避するため、width="stretch" を使用
    st.dataframe(styled_df, width="stretch", height=500)