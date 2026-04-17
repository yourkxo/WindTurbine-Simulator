"""
=================================================================
  WIND TURBINE BLADE SIMULATOR V8 - STREAMLIT WEB EDITION
=================================================================
"""
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from matplotlib.path import Path
import csv
from io import StringIO
from datetime import datetime

# ============================================================
# STREAMLIT PAGE CONFIG & SESSION STATE
# ============================================================
st.set_page_config(layout="wide", page_title="Wind Turbine Simulator V8", page_icon="🌪️")

st.markdown("""
<style>
    /* ปรับแต่ง UI ให้คล้ายโปรแกรม Desktop */
    .stTabs [data-baseweb="tab-list"] { gap: 20px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; font-weight: bold; }
    div[data-testid="stSidebar"] { padding-top: 1rem; }
    .stButton>button { width: 100%; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Initialize Session States (แทนการใช้ self.variable ใน Tkinter)
if 'rib_data' not in st.session_state: st.session_state.rib_data = []
if 'sim_results' not in st.session_state: st.session_state.sim_results = None
if 'smart_best' not in st.session_state: st.session_state.smart_best = None
if 'wind_speed' not in st.session_state: st.session_state.wind_speed = 3.0
if 'rpm' not in st.session_state: st.session_state.rpm = 400.0
if 'blade_L' not in st.session_state: st.session_state.blade_L = 300.0
if 'root_c' not in st.session_state: st.session_state.root_c = 70.0
if 'tip_c' not in st.session_state: st.session_state.tip_c = 25.0
if 'root_t' not in st.session_state: st.session_state.root_t = 20.0
if 'tip_t' not in st.session_state: st.session_state.tip_t = 2.0

# ============================================================
# HIGH-PRECISION AERODYNAMIC FUNCTIONS (BEM) - FULL V8 MATH
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
# MAIN UI: TABS (ถอดแบบ Tkinter)
# ============================================================
st.title("🌪️ 3D Wind Turbine Blade Simulator V8 (Ultimate Precision)")

tab_design, tab_sim, tab_smart, tab_manual = st.tabs([
    "📏 Design & 3D Generate", 
    "🌪️ Simulate & Inspect", 
    "🎯 Smart Reverse Design", 
    "📖 Manual & Equations"
])

# ------------------------------------------------------------
# TAB 1: DESIGN (แบบแบ่งซ้าย ขวา)
# ------------------------------------------------------------
with tab_design:
    col1, col2 = st.columns([1, 2.5])
    
    with col1:
        st.subheader("📋 Input Parameters")
        
        with st.expander("Airfoil", expanded=True):
            naca_code = st.text_input("NACA Code", value="4412")
            
        with st.expander("Dimensions [mm]", expanded=True):
            blade_L = st.number_input("Span (Length)", value=st.session_state.blade_L)
            root_c = st.number_input("Root Chord", value=st.session_state.root_c)
            tip_c = st.number_input("Tip Chord", value=st.session_state.tip_c)
            
        with st.expander("Twist [deg]", expanded=True):
            root_t = st.number_input("Root Twist", value=st.session_state.root_t)
            tip_t = st.number_input("Tip Twist", value=st.session_state.tip_t)
            
        with st.expander("Mesh Properties", expanded=True):
            spacing = st.number_input("Rib Spacing [mm]", value=15.0)
            pts = st.number_input("Points/Rib", value=80)
            
        btn_gen = st.button("✨ Generate 3D", type="primary")

        # บันทึกสถานะล่าสุด
        st.session_state.blade_L = blade_L
        st.session_state.root_c = root_c
        st.session_state.tip_c = tip_c
        st.session_state.root_t = root_t
        st.session_state.tip_t = tip_t

    # --- กระบวนการสร้าง 3D ---
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
        st.subheader("📐 3D Blade Model (Interactive)")
        if len(st.session_state.rib_data) > 0:
            Xs, Ys, Zs = [], [], []
            for rib in st.session_state.rib_data:
                Xs.extend(rib['X_rot'])
                Ys.extend(rib['Y_rot'])
                Zs.extend(rib['Z_rot'])
                
            fig3d = go.Figure(data=[go.Scatter3d(
                x=Xs, y=Zs, z=Ys, mode='markers',
                marker=dict(size=2, color=Zs, colorscale='Viridis', opacity=0.8)
            )])
            fig3d.update_layout(scene=dict(xaxis_title='X [mm]', yaxis_title='Span Z [mm]', zaxis_title='Thickness Y [mm]', aspectmode='data'), height=600)
            st.plotly_chart(fig3d, use_container_width=True)
            
            # Export CSV สำหรับ Fusion360 แบบเดิม
            csv_buffer = StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["X", "Y", "Z"])
            for rib in st.session_state.rib_data:
                for x, y, z in zip(rib['X_rot'], rib['Y_rot'], rib['Z_rot']):
                    writer.writerow([f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"])
            st.download_button("💾 Export CSV for Fusion 360", data=csv_buffer.getvalue(), file_name=f"Blade_{naca_code}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", mime="text/csv")

# ------------------------------------------------------------
# TAB 2: SIMULATE & INSPECT
# ------------------------------------------------------------
with tab_sim:
    scol1, scol2 = st.columns([1, 2.5])
    
    with scol1:
        st.subheader("🌪️ Environment")
        v_wind = st.number_input("Wind Speed [m/s]", value=st.session_state.wind_speed, key='sim_wind')
        rpm = st.number_input("RPM", value=st.session_state.rpm, key='sim_rpm')
        rho = st.number_input("Air Density [kg/m³]", value=1.225)
        num_blades = st.number_input("Number of Blades", value=3, min_value=1, max_value=6)
        
        if st.button("🎯 AI Auto-Optimize Twist"):
            opt_rt, opt_tt = calculate_optimal_twist(v_wind, rpm, st.session_state.blade_L)
            st.session_state.root_t = opt_rt
            st.session_state.tip_t = opt_tt
            st.success(f"Optimized! Root: {opt_rt}°, Tip: {opt_tt}° (Please Re-Generate 3D in Design Tab)")
            
        btn_sim = st.button("🚀 Run BEM Simulation", type="primary")
        
        st.markdown("---")
        st.subheader("2D Flow Inspector")
        if len(st.session_state.rib_data) > 0:
            inspect_idx = st.slider("Select Rib to Inspect", 0, len(st.session_state.rib_data)-1, 0)
            rib = st.session_state.rib_data[inspect_idx]
            st.info(f"Rib #{inspect_idx} | Z = {rib['z']:.1f} mm | Chord = {rib['chord']:.1f} mm | Twist = {rib['twist']:.1f}°")

    if btn_sim:
        st.session_state.wind_speed = v_wind
        st.session_state.rpm = rpm
        st.session_state.sim_results = run_bem_high_precision(
            v_wind, rpm, st.session_state.blade_L, st.session_state.root_c, st.session_state.tip_c,
            st.session_state.root_t, st.session_state.tip_t, num_blades, rho, 20
        )

    with scol2:
        res = st.session_state.sim_results
        if res:
            st.subheader("📊 Simulation Results")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status", res['status'])
            c2.metric("TSR (λ)", f"{res['TSR']:.2f}")
            c3.metric("Cp (Efficiency)", f"{res['Cp']:.4f}")
            c4.metric("Mech Power", f"{res['power_W']*1000 * (num_blades/3):.2f} mW")
            
            # กราฟ Aerodynamics ถอดแบบ Tkinter เดิม
            r_list = [e['r_mm']/1000 for e in res['elements']]
            lift_list = [e['Cl'] * 0.5 * rho * e['V_rel']**2 * (e['chord_mm']/1000) * 0.015 * 1000 for e in res['elements']]
            aoa_list = [e['alpha_deg'] for e in res['elements']]
            
            fig_aero, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            ax1.plot(r_list, lift_list, marker='o', color='blue')
            ax1.fill_between(r_list, lift_list, color='blue', alpha=0.1)
            ax1.set_title("Lift Distribution (mN)")
            ax1.set_xlabel("Radius (m)")
            ax1.grid(True)
            
            ax2.plot(r_list, aoa_list, marker='s', color='red')
            ax2.axhline(14, color='black', linestyle='--', label="Stall (14°)")
            ax2.axhline(5, color='green', linestyle=':', label="Optimal (5°)")
            ax2.set_title("Angle of Attack vs Radius")
            ax2.set_xlabel("Radius (m)")
            ax2.set_ylabel("AoA (°)")
            ax2.legend()
            ax2.grid(True)
            st.pyplot(fig_aero)
            
            # Text Report แบบ Console
            st.code(f"""
=== BEM SIMULATION V8 REPORT ===
Wind: {v_wind} m/s | RPM: {rpm} | Blades: {num_blades}
────────────────────────────────
TSR (λ)     : {res['TSR']:.2f}
Cp          : {res['Cp']:.4f} ({res['Cp']/0.5926*100:.1f}% of Betz)
Thrust      : {res['thrust_N'] * (num_blades/3):.4f} N
Torque      : {res['torque_Nm'] * (num_blades/3):.6f} N·m
POWER (MECH): {res['power_W']*1000 * (num_blades/3):.2f} mW
────────────────────────────────
AoA Range   : {res['min_aoa']:.1f}° to {res['max_aoa']:.1f}°
Stall elems : {res['stall_count']}/20
Status      : {res['status']}
            """, language="text")
        else:
            st.info("👈 กดปุ่ม Run BEM Simulation เพื่อดูผลลัพธ์")

# ------------------------------------------------------------
# TAB 3: SMART REVERSE DESIGN
# ------------------------------------------------------------
with tab_smart:
    rcol1, rcol2 = st.columns([1, 2])
    with rcol1:
        st.subheader("📋 Reverse Design Inputs")
        r_wind = st.number_input("Target Wind [m/s] (≤3.6)", value=3.0)
        r_load = st.number_input("Load Resistance [Ω]", value=10.0)
        r_tgt_p = st.number_input("Target Power [mW] (0=Max)", value=0.0)
        r_max_l = st.number_input("Max Blade Length [mm]", value=400.0)
        
        with st.expander("Generator Specs", expanded=False):
            gv = st.number_input("Max Voltage [V]", value=10.0)
            gi = st.number_input("Max Current [A]", value=0.3)
            gr = st.number_input("Rated RPM", value=1500.0)
            
        btn_search = st.button("🔍 Run Live BEM Search", type="primary")

    with rcol2:
        if btn_search:
            with st.spinner("🧮 Running Live BEM Permutations... (Please wait)"):
                blade_lens = [l for l in [100, 150, 200, 250, 300, 350, 400] if l <= r_max_l]
                root_chords = [30, 50, 70, 90, 110]
                tip_chords = [15, 25, 35, 50, 65]
                rpms = [80, 150, 250, 400, 550]
                gears = [1, 2, 4, 6]
                
                live_results = []
                for L in blade_lens:
                    for rc in root_chords:
                        for tc in tip_chords:
                            if tc >= rc: continue
                            for rpm in rpms:
                                rt, tt = calculate_optimal_twist(r_wind, rpm, L)
                                aero = run_bem_high_precision(r_wind, rpm, L, rc, tc, rt, tt, 3, 1.225, 15)
                                if aero['power_W'] <= 0 or aero['status'] in ('SEVERE_STALL', 'NO_POWER', 'INVALID'): continue
                                
                                for g in gears:
                                    elec = simulate_electrical_full(aero['power_W'], rpm, g, r_load, gv, gi, gr)
                                    if elec['stalled'] or elec['P_elec_mW'] <= 0: continue
                                    
                                    live_results.append({
                                        'L': L, 'rc': rc, 'tc': tc, 'rt': rt, 'tt': tt, 'rpm': rpm, 'gear': g,
                                        'tsr': aero['TSR'], 'cp': aero['Cp'], 'pmech': aero['power_W']*1000,
                                        'pelec': elec['P_elec_mW'], 'volt': elec['V_terminal'],
                                        'status': aero['status']
                                    })
                
                if live_results:
                    for d in live_results:
                        if r_tgt_p > 0: d['_score'] = -abs(d['pelec'] - r_tgt_p)
                        else: d['_score'] = d['pelec']
                    live_results.sort(key=lambda x: x['_score'], reverse=True)
                    st.session_state.smart_best = live_results[0]
                    
                    st.success(f"✅ Found Valid Designs! Showing Top 1:")
                    best = live_results[0]
                    st.code(f"""
╔══════ RANK #1 [LIVE BEM] ══════
║ ━━━ BLADE GEOMETRY ━━━
║   Blade Length     : {best['L']:.0f} mm
║   Root Chord       : {best['rc']:.0f} mm
║   Tip Chord        : {best['tc']:.0f} mm
║   Root Twist       : {best['rt']:.2f}°
║   Tip Twist        : {best['tt']:.2f}°
║ ━━━ OPERATING POINT ━━━
║   Rotor RPM        : {best['rpm']:.0f}
║   TSR (λ)          : {best['tsr']:.2f}
║   Cp (efficiency)  : {best['cp']:.4f}
║ ━━━ GEAR & ELEC ━━━
║   Gear Ratio       : 1 : {best['gear']}
║   Mechanical Power : {best['pmech']:.2f} mW
║   ELECTRICAL POWER : {best['pelec']:.2f} mW  ⚡
║   Terminal Voltage : {best['volt']:.3f} V
╚════════════════════════
                    """, language="text")
                else:
                    st.error("❌ NO VALID DESIGN FOUND. Try increasing max length or wind speed.")
        elif st.session_state.smart_best:
            st.info("Best design is stored in session. Go to Design tab and manually input these values to generate.")

# ------------------------------------------------------------
# TAB 4: MANUAL
# ------------------------------------------------------------
with tab_manual:
    st.markdown("""
    ### 📖 User Manual V8 & Equations Reference
    **WIND TURBINE SIMULATOR V8** uses **Blade Element Momentum (BEM)** theory, the industry-standard for HAWT design.
    
    #### Equations Used:
    1. **Tip Speed Ratio:** `λ = ωR / V_wind`
    2. **Prandtl Loss:** `F_tip = (2/π) · arccos(exp(-f))`
    3. **Viterna Post-Stall Extrapolation** applies when `|α| > 14°`
    4. **Buhl Correction:** Applied for high induction states (`a > 0.4`)
    5. **Electrical Constraints:** `P_elec ≤ P_mech · η_gear · η_gen`
    """)
    st.info("Accuracy: ±10-15% for well-tuned blades. Does not include unsteady effects, yaw, or turbulence.")
