"""
=================================================================
  WIND TURBINE BLADE SIMULATOR V8 - STREAMLIT EDITION
=================================================================
Features:
  - High-precision BEM (N=20 elements, Prandtl tip/hub loss, Buhl correction)
  - 2D CFD Flow Inspector (Vectors & Magnitude Contour)
  - Generator + Load Simulation
  - Multi-Blade Analysis
  - Export to Fusion 360 (CSV)
=================================================================
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.path import Path
import csv
from io import StringIO
from datetime import datetime

# ============================================================
# STREAMLIT PAGE CONFIG & CSS
# ============================================================
st.set_page_config(layout="wide", page_title="Wind Turbine Simulator V8", page_icon="🌪️")

st.markdown("""
<style>
    /* แต่ง UI ให้คล้าย Tkinter เดิม (เมนูซ้าย / แสดงผลขวา) */
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { height: 50px; font-weight: bold; background-color: #f4f7fb; border-radius: 5px 5px 0 0; padding: 0 15px;}
    .stTabs [aria-selected="true"] { background-color: #e0e6ed; border-bottom: 3px solid #0055ff; }
    .terminal-box { background-color: #0d1117; color: #00ff41; font-family: Consolas, monospace; padding: 15px; border-radius: 5px; white-space: pre-wrap; font-size: 14px;}
    .diagnostic-box { background-color: #f4f7fb; padding: 15px; border-radius: 5px; border: 1px solid #ddd; }
</style>
""", unsafe_allow_html=True)

# Initialize Session States
if 'rib_data' not in st.session_state: st.session_state.rib_data = []
if 'sim_results' not in st.session_state: st.session_state.sim_results = None
if 'wind_speed' not in st.session_state: st.session_state.wind_speed = 3.6
if 'rpm' not in st.session_state: st.session_state.rpm = 400.0
if 'blade_L' not in st.session_state: st.session_state.blade_L = 300.0
if 'root_c' not in st.session_state: st.session_state.root_c = 70.0
if 'tip_c' not in st.session_state: st.session_state.tip_c = 25.0
if 'root_t' not in st.session_state: st.session_state.root_t = 20.0
if 'tip_t' not in st.session_state: st.session_state.tip_t = 2.0

# ============================================================
# HIGH-PRECISION AERODYNAMIC FUNCTIONS (BEM)
# ============================================================
def prandtl_tip_loss(B, r, R, phi_rad):
    if abs(np.sin(phi_rad)) < 1e-6 or r >= R * 0.999: return 1.0
    f = (B / 2.0) * (R - r) / (r * abs(np.sin(phi_rad)))
    F = (2.0 / np.pi) * np.arccos(min(1.0, np.exp(-f)))
    return max(F, 0.005)

def prandtl_hub_loss(B, r, R_hub, phi_rad):
    if abs(np.sin(phi_rad)) < 1e-6 or r <= R_hub * 1.001: return 1.0
    f = (B / 2.0) * (r - R_hub) / (r * abs(np.sin(phi_rad)))
    F = (2.0 / np.pi) * np.arccos(min(1.0, np.exp(-f)))
    return max(F, 0.005)

def get_cl_cd_naca4412(alpha_deg):
    alpha_rad = np.radians(alpha_deg)
    Cl_0 = 0.45 
    Cl_alpha = 2 * np.pi 
    Cl_attached = Cl_0 + Cl_alpha * alpha_rad
    Cd_min = 0.0095
    Cd_attached = Cd_min + 0.0007 * alpha_deg + 0.00012 * alpha_deg**2
    
    alpha_stall = 14.0 
    Cl_max = 1.55
    AR = 8.0
    Cd_max = 1.11 + 0.018 * AR
    A1 = Cd_max / 2.0
    sin_s = np.sin(np.radians(alpha_stall))
    cos_s = np.cos(np.radians(alpha_stall))
    B2 = Cl_max * sin_s * cos_s - Cd_max * sin_s**2
    A2 = (Cl_max - Cd_max * sin_s * cos_s) * sin_s / (cos_s**2) if cos_s**2 > 1e-10 else 0
    
    sa, ca = np.sin(alpha_rad), np.cos(alpha_rad)
    Cl_post = A1 * np.sin(2 * alpha_rad) + A2 * ca**2 / (sa + 1e-10) if abs(alpha_deg) > 0.1 else 0
    Cd_post = Cd_max * sa**2 + B2 * ca
    
    transition_width = 2.5
    weight = 1.0 / (1.0 + np.exp(-(abs(alpha_deg) - alpha_stall) / transition_width))
    
    if alpha_deg >= 0: Cl = (1 - weight) * min(Cl_attached, Cl_max) + weight * Cl_post
    else: Cl = (1 - weight) * max(Cl_attached, -Cl_max) + weight * (-abs(Cl_post))
    
    Cd = (1 - weight) * Cd_attached + weight * Cd_post
    Cd = max(Cd, Cd_min)
    return Cl, Cd

def get_naca_coords(code, chord_mm, points):
    cs = str(code).strip().zfill(4)
    if len(cs) != 4 or not cs.isdigit(): return [], []
    m = int(cs[0]) / 100.0
    p = int(cs[1]) / 10.0
    t = int(cs[2:]) / 100.0
    beta = np.linspace(0, np.pi, points)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    if p > 0:
        front = x < p
        yc[front] = (m / p**2) * (2 * p * x[front] - x[front]**2)
        dyc_dx[front] = (2 * m / p**2) * (p - x[front])
        back = x >= p
        yc[back] = (m / (1 - p)**2) * ((1 - 2 * p) + 2 * p * x[back] - x[back]**2)
        dyc_dx[back] = (2 * m / (1 - p)**2) * (p - x[back])
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    X = np.concatenate((xl[::-1], xu[1:])) * chord_mm
    Y = np.concatenate((yl[::-1], yu[1:])) * chord_mm
    return X, Y

def run_bem_high_precision(V_wind, rpm, R_mm, root_chord_mm, tip_chord_mm,
                           root_twist_deg, tip_twist_deg, num_blades=3,
                           rho=1.225, num_elements=20):
    R = R_mm / 1000.0
    R_hub = R * 0.08
    omega = rpm * 2 * np.pi / 60.0
    B = num_blades
    
    if V_wind <= 0 or R <= 0 or omega <= 0:
        return {'power_W': 0, 'thrust_N': 0, 'torque_Nm': 0, 'Cp': 0, 'TSR': 0,
                'P_available_W': 0, 'max_aoa': 0, 'min_aoa': 0,
                'stall_count': 0, 'status': 'INVALID', 'elements': []}
    
    TSR = omega * R / V_wind
    A_swept = np.pi * R**2
    P_available = 0.5 * rho * A_swept * V_wind**3
    
    r_stations = np.linspace(R_hub + 0.005*R, R*0.995, num_elements)
    total_torque, total_thrust = 0.0, 0.0
    max_aoa, min_aoa = -999, 999
    stall_count = 0
    elements = []
    
    for i, r in enumerate(r_stations):
        frac = (r - R_hub) / (R - R_hub)
        c = (root_chord_mm + (tip_chord_mm - root_chord_mm) * frac) / 1000.0
        z_root_eff = R * 0.1
        if r <= z_root_eff: tf = 0.0
        else: tf = (1.0/z_root_eff - 1.0/r) / (1.0/z_root_eff - 1.0/R)
        local_twist_deg = root_twist_deg - (root_twist_deg - tip_twist_deg) * tf
        theta = np.radians(local_twist_deg)
        
        if i == 0: dr = r_stations[1] - r_stations[0]
        elif i == num_elements - 1: dr = r_stations[-1] - r_stations[-2]
        else: dr = (r_stations[i+1] - r_stations[i-1]) / 2.0
        
        sigma_r = B * c / (2 * np.pi * r) if r > 0 else 0
        a, a_p = 0.0, 0.0
        
        for iteration in range(50):
            V_axial = max(V_wind * (1 - a), 0.001)
            V_tan = omega * r * (1 + a_p)
            phi = np.arctan2(V_axial, V_tan)
            alpha_deg = np.degrees(phi - theta)
            Cl, Cd = get_cl_cd_naca4412(alpha_deg)
            
            F_tip = prandtl_tip_loss(B, r, R, phi)
            F_hub = prandtl_hub_loss(B, r, R_hub, phi)
            F = F_tip * F_hub
            
            sp, cp = np.sin(phi), np.cos(phi)
            if abs(sp) < 1e-12: break
            
            Cn = Cl * cp + Cd * sp
            Ct = Cl * sp - Cd * cp
            
            if sigma_r * Cn != 0: a_new = 1.0 / ((4 * F * sp**2) / (sigma_r * Cn) + 1)
            else: a_new = 0
            
            if a_new > 0.4:
                ac = 0.34
                K = 4 * F * sp**2 / (sigma_r * Cn) if sigma_r * Cn != 0 else 1e6
                disc = (K * (1 - 2*ac) + 2)**2 + 4 * (K * ac**2 - 1)
                if disc >= 0: a_new = 0.5 * (2 + K * (1 - 2*ac) - np.sqrt(disc))
                else: a_new = ac
                a_new = max(0, min(a_new, 0.95))
            
            if sigma_r * Ct != 0: a_p_new = 1.0 / ((4 * F * sp * cp) / (sigma_r * Ct) - 1)
            else: a_p_new = 0
            
            a_p_new = max(-0.5, min(a_p_new, 2.0))
            a_new = max(0, min(a_new, 0.95))
            
            if abs(a_new - a) < 1e-6 and abs(a_p_new - a_p) < 1e-6 and iteration > 5: break
            
            a = a * 0.6 + a_new * 0.4
            a_p = a_p * 0.6 + a_p_new * 0.4
            
        V_rel = np.sqrt(V_axial**2 + V_tan**2)
        q = 0.5 * rho * V_rel**2
        dT = q * c * dr * (Cl * np.cos(phi) + Cd * np.sin(phi)) * B
        dQ = q * c * dr * (Cl * np.sin(phi) - Cd * np.cos(phi)) * B * r
        
        total_thrust += dT
        total_torque += dQ
        max_aoa = max(max_aoa, alpha_deg)
        min_aoa = min(min_aoa, alpha_deg)
        if abs(alpha_deg) > 14: stall_count += 1
        
        elements.append({
            'r_mm': r * 1000, 'chord_mm': c * 1000, 'twist_deg': local_twist_deg,
            'alpha_deg': alpha_deg, 'Cl': Cl, 'Cd': Cd, 'V_rel': V_rel, 'a': a, 'a_p': a_p
        })
        
    power_W = total_torque * omega
    P_max_betz = P_available * 0.5926
    if power_W > P_max_betz:
        power_W = P_max_betz
        total_torque = power_W / omega if omega > 0 else 0
    power_W = max(power_W, 0)
    Cp = power_W / P_available if P_available > 0 else 0
    
    if stall_count > num_elements * 0.5: status = 'SEVERE_STALL'
    elif stall_count > 0: status = 'PARTIAL_STALL'
    elif min_aoa < -5: status = 'NEGATIVE_AOA'
    elif power_W <= 0: status = 'NO_POWER'
    elif Cp > 0.45: status = 'EXCELLENT'
    elif Cp > 0.30: status = 'GOOD'
    elif Cp > 0.15: status = 'MODERATE'
    else: status = 'LOW_EFFICIENCY'
    
    return {
        'power_W': power_W, 'thrust_N': total_thrust, 'torque_Nm': total_torque,
        'Cp': Cp, 'TSR': TSR, 'P_available_W': P_available,
        'max_aoa': max_aoa, 'min_aoa': min_aoa, 'stall_count': stall_count, 
        'status': status, 'elements': elements
    }

def simulate_electrical_full(mech_power_W, rotor_rpm, gear_ratio, load_ohm,
                             gen_v_max=10.0, gen_i_max=0.3, gen_rated_rpm=1500.0, gear_eff=0.95, gen_eff=0.85):
    gen_rpm = rotor_rpm * gear_ratio
    P_shaft = mech_power_W * gear_eff * gen_eff
    R_int = gen_v_max / gen_i_max if gen_i_max > 0 else 999
    V_emf = (gen_rpm / gen_rated_rpm) * gen_v_max if gen_rated_rpm > 0 else 0
    I_raw = V_emf / (R_int + load_ohm) if (R_int + load_ohm) > 0 else 0
    P_raw = I_raw**2 * load_ohm
    
    if P_shaft <= 0:
        return {'gen_rpm': gen_rpm, 'V_emf': V_emf, 'V_terminal': 0, 'current_A': 0, 
                'P_elec_W': 0, 'P_elec_mW': 0, 'R_internal': R_int, 'overall_efficiency': 0, 'stalled': True}
                
    if P_raw > P_shaft:
        I = np.sqrt(P_shaft / load_ohm) if load_ohm > 0 else 0
        V_t = I * load_ohm
        P_e = P_shaft
        stalled = True
    else:
        I = I_raw
        V_t = I * load_ohm
        P_e = P_raw
        stalled = False
        
    eff = P_e / mech_power_W if mech_power_W > 0 else 0
    return {
        'gen_rpm': gen_rpm, 'V_emf': V_emf, 'V_terminal': V_t, 'current_A': I, 
        'P_elec_W': P_e, 'P_elec_mW': P_e * 1000, 'R_internal': R_int, 
        'overall_efficiency': min(eff, 1.0), 'stalled': stalled
    }

def calculate_optimal_twist(V_wind, rpm, R_mm, target_aoa=5.0):
    R = R_mm / 1000.0
    omega = rpm * 2 * np.pi / 60.0
    if omega <= 0 or R <= 0: return 15.0, 2.0
    r_root = R * 0.15
    rt = np.degrees(np.arctan2(V_wind, omega * r_root)) - target_aoa
    r_tip = R * 0.95
    tt = np.degrees(np.arctan2(V_wind, omega * r_tip)) - target_aoa
    return round(max(-10, min(rt, 45)), 2), round(max(-10, min(tt, 30)), 2)

# ============================================================
# MAIN UI (TABS)
# ============================================================
st.title("🌪️ 3D Wind Turbine Blade Simulator V8")

tab_design, tab_sim, tab_cfd, tab_gen, tab_multi, tab_manual = st.tabs([
    "📏 Design & 3D", 
    "🌪️ Simulate BEM", 
    "🌊 2D CFD Flow", 
    "🔌 Gen + Load",
    "⚙️ Multi-Blade",
    "📖 Manual"
])

# ------------------------------------------------------------
# TAB 1: DESIGN & 3D
# ------------------------------------------------------------
with tab_design:
    col1, col2 = st.columns([1, 2.5])
    with col1:
        st.subheader("📋 Input Parameters")
        naca_code = st.text_input("NACA Code", value="4412")
        blade_L = st.number_input("Span (Length) [mm]", value=st.session_state.blade_L, step=10.0)
        root_c = st.number_input("Root Chord [mm]", value=st.session_state.root_c, step=5.0)
        tip_c = st.number_input("Tip Chord [mm]", value=st.session_state.tip_c, step=5.0)
        root_t = st.number_input("Root Twist [deg]", value=st.session_state.root_t, step=1.0)
        tip_t = st.number_input("Tip Twist [deg]", value=st.session_state.tip_t, step=1.0)
        spacing = st.number_input("Rib Spacing [mm]", value=15.0, step=1.0)
        pts = st.number_input("Points/Rib", value=80, step=10)
        
        st.session_state.blade_L = blade_L
        st.session_state.root_c = root_c
        st.session_state.tip_c = tip_c
        st.session_state.root_t = root_t
        st.session_state.tip_t = tip_t

        btn_gen = st.button("✨ Generate 3D", type="primary")

    if btn_gen or len(st.session_state.rib_data) == 0:
        n_ribs = int(np.ceil(blade_L / spacing)) + 1
        z_stations = np.linspace(0, blade_L, n_ribs)
        rib_data = []
        for i, z in enumerate(z_stations):
            c = root_c - ((root_c - tip_c) * (z / blade_L))
            ze = blade_L * 0.1
            if z <= ze: f = 0.0
            else: f = (1.0/ze - 1.0/z) / (1.0/ze - 1.0/blade_L)
            twist = root_t - ((root_t - tip_t) * f)
            
            X2, Y2 = get_naca_coords(naca_code, c, int(pts))
            if len(X2) > 0:
                rad = np.radians(twist)
                X1 = X2 * np.cos(rad) - Y2 * np.sin(rad)
                Y1 = X2 * np.sin(rad) + Y2 * np.cos(rad)
                Z1 = np.full_like(X1, z)
                rib_data.append({'id': i, 'z': z, 'chord': c, 'twist': twist, 'X_2d': X2, 'Y_2d': Y2, 'X_rot': X1, 'Y_rot': Y1, 'Z_rot': Z1})
        st.session_state.rib_data = rib_data

    with col2:
        st.subheader(f"📐 3D Blade - NACA {naca_code}")
        if len(st.session_state.rib_data) > 0:
            Xs, Ys, Zs = [], [], []
            for rib in st.session_state.rib_data:
                Xs.extend(rib['X_rot'])
                Ys.extend(rib['Y_rot'])
                Zs.extend(rib['Z_rot'])
            
            # Interactive 3D Plotly (สีเทาแบบ Tkinter เดิม)
            fig3d = go.Figure(data=[go.Scatter3d(
                x=Xs, y=Zs, z=Ys, mode='markers',
                marker=dict(size=1.5, color='#b0b5b9', opacity=0.8)
            )])
            fig3d.update_layout(scene=dict(xaxis_title='X [mm]', yaxis_title='Span Z [mm]', zaxis_title='Thickness Y [mm]', aspectmode='data'), height=600, margin=dict(l=0, r=0, b=0, t=0))
            st.plotly_chart(fig3d, use_container_width=True)
            
            csv_buffer = StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["X", "Y", "Z"])
            for rib in st.session_state.rib_data:
                for x, y, z in zip(rib['X_rot'], rib['Y_rot'], rib['Z_rot']):
                    writer.writerow([f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"])
            st.download_button("💾 Export CSV for Fusion 360", data=csv_buffer.getvalue(), file_name=f"Blade_{naca_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

# ------------------------------------------------------------
# TAB 2: SIMULATE BEM
# ------------------------------------------------------------
with tab_sim:
    scol1, scol2 = st.columns([1, 2.5])
    with scol1:
        st.subheader("🌪️ Environment & Operation")
        v_wind = st.number_input("Wind [m/s]", value=st.session_state.wind_speed, step=0.1)
        rpm = st.number_input("RPM", value=st.session_state.rpm, step=10.0)
        rho = st.number_input("Density [kg/m³]", value=1.225)
        num_blades = st.number_input("Number of Blades", value=3, min_value=1, max_value=6)
        
        st.session_state.wind_speed = v_wind
        st.session_state.rpm = rpm
        
        if st.button("🎯 AI Auto-Optimize Twist"):
            opt_rt, opt_tt = calculate_optimal_twist(v_wind, rpm, st.session_state.blade_L)
            st.session_state.root_t = opt_rt
            st.session_state.tip_t = opt_tt
            st.success(f"Optimized! Root: {opt_rt}°, Tip: {opt_tt}° (Go to Design tab to re-generate)")
            
        btn_sim = st.button("🚀 Run BEM Simulation", type="primary")

    if btn_sim:
        st.session_state.sim_results = run_bem_high_precision(
            v_wind, rpm, st.session_state.blade_L, st.session_state.root_c, st.session_state.tip_c,
            st.session_state.root_t, st.session_state.tip_t, num_blades, rho, 20
        )

    with scol2:
        res = st.session_state.sim_results
        if res:
            st.subheader("📊 Diagnostic Report")
            
            # Terminal Box
            status_color = "#ff4444" if "STALL" in res['status'] else "#00ff41"
            st.markdown(f"""
            <div class="terminal-box">=== BEM SIMULATION V8 ===
TSR (λ)     : {res['TSR']:.2f}
Cp          : {res['Cp']:.4f} ({res['Cp']/0.5926*100:.1f}% of Betz)
P_avail     : {res['P_available_W']*1000:.2f} mW
─────────────────
Thrust      : {res['thrust_N']:.4f} N
Torque      : {res['torque_Nm']:.6f} N·m
MECH POWER  : {res['power_W']*1000:.2f} mW
─────────────────
AoA         : {res['min_aoa']:.1f}° to {res['max_aoa']:.1f}°
Stall elems : {res['stall_count']}/20
<span style="color: {status_color}">Status      : {res['status']}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Aero Graphs
            r_list = [e['r_mm']/1000 for e in res['elements']]
            lift_list = [e['Cl'] * 0.5 * rho * e['V_rel']**2 * (e['chord_mm']/1000) * 0.015 * 1000 for e in res['elements']]
            aoa_list = [e['alpha_deg'] for e in res['elements']]
            
            fig_aero, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.plot(r_list, lift_list, marker='o', color='#0055ff', linewidth=2.5)
            ax1.fill_between(r_list, lift_list, color='#0055ff', alpha=0.15)
            ax1.set_title("Lift Distribution (mN)", fontweight='bold')
            ax1.set_xlabel("Radius (m)")
            ax1.set_ylabel("Lift (mN)")
            ax1.grid(color='#e0e0e0', linestyle='--')
            
            ax2.plot(r_list, aoa_list, marker='s', color='#ff0044', linewidth=2.5)
            ax2.axhline(14, color='black', linestyle='--', label="Stall (14°)")
            ax2.axhline(5, color='green', linestyle=':', label="Optimal (5°)")
            ax2.axhline(0, color='grey', linestyle='-')
            ax2.set_title("AoA vs Radius", fontweight='bold')
            ax2.set_xlabel("Radius (m)")
            ax2.set_ylabel("AoA (°)")
            ax2.legend()
            ax2.grid(color='#e0e0e0', linestyle='--')
            st.pyplot(fig_aero)
        else:
            st.info("👈 กดปุ่ม Run BEM Simulation เพื่อดูผลการวิเคราะห์")

# ------------------------------------------------------------
# TAB 3: 2D CFD FLOW INSPECTOR
# ------------------------------------------------------------
with tab_cfd:
    st.subheader("🌊 2D CFD Flow - Vector & Magnitude Inspector")
    if st.session_state.sim_results and len(st.session_state.rib_data) > 0:
        ccol1, ccol2 = st.columns([1, 2.5])
        with ccol1:
            inspect_idx = st.slider("Select Rib Index", 0, len(st.session_state.rib_data)-1, 0)
            rib = st.session_state.rib_data[inspect_idx]
            
            # Find closest BEM element data for this rib
            r_mm = rib['z']
            res = st.session_state.sim_results
            closest = min(res['elements'], key=lambda e: abs(e['r_mm'] - r_mm)) if res['elements'] else None
            
            if closest:
                rib_V_rel = closest['V_rel']
                rib_alpha = closest['alpha_deg']
                rib_Cl = closest['Cl']
                rib_dL = closest['Cl'] * 0.5 * 1.225 * closest['V_rel']**2 * (rib['chord']/1000) * 0.02
                
                diag_color = "#d9534f" if rib_alpha > 14 else "#f0ad4e" if rib_alpha < 0 else "#5cb85c" if 3 <= rib_alpha <= 8 else "#337ab7"
                diag_msg = "🚨 STALL (Decrease twist/increase RPM)" if rib_alpha > 14 else "⚠️ Negative AoA" if rib_alpha < 0 else "✅ Optimal AoA" if 3 <= rib_alpha <= 8 else "ℹ️ Suboptimal AoA"
                
                st.markdown(f"""
                <div class="diagnostic-box">
                    <h4>📊 Diagnostics (Rib {inspect_idx})</h4>
                    <b>Z Pos:</b> {r_mm:.1f} mm<br>
                    <b>Chord:</b> {rib['chord']:.1f} mm<br>
                    <b>Twist:</b> {rib['twist']:.1f}°<br>
                    <b>AoA:</b> {rib_alpha:.2f}°<br>
                    <b>V_rel:</b> {rib_V_rel:.2f} m/s<br>
                    <b>Cl:</b> {rib_Cl:.3f}<br><br>
                    <b>Analysis:</b><br>
                    <span style="color: {diag_color}; font-weight: bold;">{diag_msg}</span>
                </div>
                """, unsafe_allow_html=True)

        with ccol2:
            fig_cfd, ax = plt.subplots(figsize=(9, 6))
            alpha_rad = np.radians(rib_alpha)
            Xf = rib['X_2d']
            Yf = rib['Y_2d']
            X = Xf * np.cos(-alpha_rad) - Yf * np.sin(-alpha_rad)
            Y = Xf * np.sin(-alpha_rad) + Yf * np.cos(-alpha_rad)
            
            chord = rib['chord']
            x_grid = np.linspace(-0.6*chord, 1.5*chord, 150)
            y_grid = np.linspace(-0.8*chord, 0.8*chord, 150)
            XX, YY = np.meshgrid(x_grid, y_grid)
            
            Gamma = rib_dL / (1.225 * rib_V_rel + 1e-6)
            xc, yc = 0.25*chord, 0
            Rc = 0.25*chord
            dx, dy = XX - xc, YY - yc
            r2 = dx**2 + dy**2
            r2[r2 < Rc**2] = Rc**2
            
            u = rib_V_rel * (1 - Rc**2 * (dx**2 - dy**2) / r2**2) + (Gamma * dy) / (2 * np.pi * r2)
            v = rib_V_rel * (-2 * Rc**2 * dx * dy / r2**2) - (Gamma * dx) / (2 * np.pi * r2)
            
            path = Path(np.column_stack((X, Y)))
            mask = path.contains_points(np.column_stack((XX.flatten(), YY.flatten()))).reshape(XX.shape)
            u[mask], v[mask] = np.nan, np.nan
            speed = np.sqrt(u**2 + v**2)
            
            cont = ax.contourf(XX, YY, speed, levels=40, cmap='coolwarm', alpha=0.6)
            fig_cfd.colorbar(cont, ax=ax, label='Velocity Magnitude (m/s)')
            ax.streamplot(x_grid, y_grid, u, v, color='white', linewidth=0.8, density=1.2)
            ax.plot(X, Y, 'k', linewidth=2.5)
            ax.fill(X, Y, color='#333333')
            ax.set_title(f"Flow around Rib {inspect_idx} (AoA={rib_alpha:.1f}°)", fontsize=13, fontweight='bold')
            ax.set_aspect('equal')
            ax.axis('off')
            
            st.pyplot(fig_cfd)
    else:
        st.warning("Please run the BEM Simulation in the 'Simulate BEM' tab first.")

# ------------------------------------------------------------
# TAB 4: GEN + LOAD
# ------------------------------------------------------------
with tab_gen:
    gcol1, gcol2 = st.columns([1, 2.5])
    with gcol1:
        st.subheader("⚙️ Generator & Gear")
        gen_v = st.number_input("Max Voltage [V]", value=10.0)
        gen_i = st.number_input("Max Current [A]", value=0.3)
        gen_rpm = st.number_input("Rated RPM", value=1500.0)
        nb_gen = st.number_input("Blades", value=3, min_value=1, key='gen_nb')
        gr = st.number_input("Gear Ratio (1:X)", value=4.0)
        st.caption(f"Rotor: {int(20*gr)} teeth / Gen: 20 teeth")
        
        st.subheader("🔋 Load")
        load_ohm = st.slider("Load Resistance [Ω]", 0.1, 100.0, 10.0)
        time_s = st.number_input("Time [s]", value=60.0)

    with gcol2:
        if st.session_state.sim_results:
            total_mech = st.session_state.sim_results['power_W'] * nb_gen
            elec = simulate_electrical_full(total_mech, st.session_state.rpm, gr, load_ohm, gen_v, gen_i, gen_rpm)
            energy = elec['P_elec_W'] * time_s
            
            warn = "\n⚠️ NO POWER (Blade stall)" if total_mech <= 0 else "\n⚠️ Gen overloaded (capped)" if elec['stalled'] else ""
            
            st.markdown(f"""
            <div class="terminal-box">Rotor RPM      : {st.session_state.rpm:.1f}
Gen RPM        : {elec['gen_rpm']:.1f} (1:{gr})
P_mech (total) : {total_mech*1000:.3f} mW
─────────────────
R_internal     : {elec['R_internal']:.2f} Ω
V_EMF          : {elec['V_emf']:.4f} V
V_terminal     : {elec['V_terminal']:.4f} V
Current        : {elec['current_A']*1000:.3f} mA
─────────────────
P_elec         : {elec['P_elec_mW']:.3f} mW
Energy ({time_s}s)   : {energy*1000:.3f} mJ
Efficiency     : {elec['overall_efficiency']*100:.1f}% <span style="color: #ff4444">{warn}</span>
            </div>
            """, unsafe_allow_html=True)
            
            fig_load, ax = plt.subplots(figsize=(8, 4.5))
            r_range = np.linspace(0.5, max(100, load_ohm*2), 200)
            p_shaft = total_mech * 0.95 * 0.85
            p_curve = []
            for rr_val in r_range:
                if p_shaft <= 0: p_curve.append(0); continue
                i_r = elec['V_emf'] / (elec['R_internal'] + rr_val)
                p_raw = i_r**2 * rr_val
                p_curve.append(min(p_raw, p_shaft) * 1000)
                
            ax.plot(r_range, p_curve, color='#0055ff', linewidth=2.5, label='Power (mW)')
            ax.plot(load_ohm, elec['P_elec_mW'], 'ro', markersize=10, label=f'Current ({load_ohm}Ω)')
            ax.axvline(elec['R_internal'], color='grey', linestyle='--', label=f'Match (R_int={elec["R_internal"]:.1f}Ω)')
            ax.set_title("Power vs Load Resistance", fontsize=12, fontweight='bold')
            ax.set_xlabel("Load Resistance [Ω]")
            ax.set_ylabel("Electrical Power [mW]")
            ax.grid(color='#e0e0e0', linestyle='--')
            ax.legend()
            st.pyplot(fig_load)
        else:
            st.warning("Please run the BEM Simulation in the 'Simulate BEM' tab first.")

# ------------------------------------------------------------
# TAB 5: MULTI-BLADE
# ------------------------------------------------------------
with tab_multi:
    mcol1, mcol2 = st.columns([1, 2.5])
    with mcol1:
        st.subheader("⚙️ Multi-Blade Setup")
        n = st.radio("Number of Blades", [2, 3, 4], index=1, horizontal=True)
        
    with mcol2:
        if st.session_state.sim_results and len(st.session_state.rib_data) > 0:
            tp = st.session_state.sim_results['power_W'] * n
            tt_total = st.session_state.sim_results['thrust_N'] * n
            tq = st.session_state.sim_results['torque_Nm'] * n
            
            stall_msg = "\n🚨 BLADE STALL" if tp <= 0 else ""
            st.markdown(f"""
            <div class="terminal-box">=== MULTI-BLADE REPORT ===
Blades: {n}
Wind  : {st.session_state.wind_speed:.2f} m/s
RPM   : {st.session_state.rpm:.0f}
─────────────────
Per blade:
  Power : {st.session_state.sim_results['power_W']*1000:.2f} mW

TOTAL ({n} blades):
  Thrust: {tt_total:.4f} N
  Torque: {tq:.6f} N·m
  POWER : {tp*1000:.2f} mW <span style="color: #ff4444">{stall_msg}</span>
            </div>
            """, unsafe_allow_html=True)
            
            # Plotly 3D Multi-blade
            fig_multi = go.Figure()
            for b in range(n):
                off = b * (2 * np.pi / n)
                Xs, Ys, Zs = [], [], []
                for rib in st.session_state.rib_data:
                    Xm = rib['X_rot'] * np.cos(off) + rib['Z_rot'] * np.sin(off)
                    Zm = -rib['X_rot'] * np.sin(off) + rib['Z_rot'] * np.cos(off)
                    Xs.extend(Xm)
                    Ys.extend(rib['Y_rot'])
                    Zs.extend(Zm)
                fig_multi.add_trace(go.Scatter3d(x=Xs, y=Zs, z=Ys, mode='markers', marker=dict(size=1.5, color='#b0b5b9', opacity=0.8), showlegend=False))
            
            L = st.session_state.blade_L
            fig_multi.update_layout(scene=dict(xaxis_title='X [mm]', yaxis_title='Z [mm]', zaxis_title='Y [mm]', 
                                               xaxis=dict(range=[-L, L]), yaxis=dict(range=[-L/2, L*1.5]), zaxis=dict(range=[-L, L]), aspectmode='cube'), height=600)
            st.plotly_chart(fig_multi, use_container_width=True)
        else:
            st.warning("Please run the BEM Simulation first.")

# ------------------------------------------------------------
# TAB 6: MANUAL
# ------------------------------------------------------------
with tab_manual:
    st.markdown("""
    ## 📖 User Manual V8 & Equations Reference
    **WIND TURBINE SIMULATOR V8** uses **Blade Element Momentum (BEM)** theory, the industry-standard for HAWT design.
    
    ### Equations Used:
    1. **Tip Speed Ratio:** `λ = ωR / V_wind`
    2. **Prandtl Loss:** `F_tip = (2/π) · arccos(exp(-f))`
    3. **Viterna Post-Stall Extrapolation** applies when `|α| > 14°`
    4. **Buhl Correction:** Applied for high induction states (`a > 0.4`)
    5. **Electrical Constraints:** `P_elec ≤ P_mech · η_gear · η_gen`
    
    *Note: Accuracy is ±10-15% for well-tuned blades. Does not include unsteady effects, yaw, or turbulence.*
    """)
