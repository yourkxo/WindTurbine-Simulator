"""
=================================================================
  WIND TURBINE BLADE SIMULATOR V8 - STREAMLIT WEB EDITION
=================================================================
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import csv
from io import StringIO
from datetime import datetime
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(layout="wide", page_title="Wind Turbine Simulator V8", page_icon="🌪️")
st.markdown("""
<style>
    .css-18e3th9 { padding-top: 1rem; }
    .stButton>button { width: 100%; border-radius: 5px; }
</style>
""", unsafe_allow_html=True)

# Initialize Session State for cross-tab communication
if 'v_wind' not in st.session_state: st.session_state.v_wind = 3.6
if 'rpm' not in st.session_state: st.session_state.rpm = 400.0
if 'blade_L' not in st.session_state: st.session_state.blade_L = 300.0
if 'root_c' not in st.session_state: st.session_state.root_c = 70.0
if 'tip_c' not in st.session_state: st.session_state.tip_c = 25.0
if 'root_t' not in st.session_state: st.session_state.root_t = 20.0
if 'tip_t' not in st.session_state: st.session_state.tip_t = 2.0
if 'rib_data' not in st.session_state: st.session_state.rib_data = []
if 'bem_results' not in st.session_state: st.session_state.bem_results = None

# ============================================================
# HIGH-PRECISION AERODYNAMIC FUNCTIONS (BEM)
# ============================================================
def prandtl_tip_loss(B, r, R, phi_rad):
    if abs(np.sin(phi_rad)) < 1e-6 or r >= R * 0.999:
        return 1.0
    f = (B / 2.0) * (R - r) / (r * abs(np.sin(phi_rad)))
    F = (2.0 / np.pi) * np.arccos(min(1.0, np.exp(-f)))
    return max(F, 0.005)

def prandtl_hub_loss(B, r, R_hub, phi_rad):
    if abs(np.sin(phi_rad)) < 1e-6 or r <= R_hub * 1.001:
        return 1.0
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
    if alpha_deg >= 0:
        Cl = (1 - weight) * min(Cl_attached, Cl_max) + weight * Cl_post
    else:
        Cl = (1 - weight) * max(Cl_attached, -Cl_max) + weight * (-abs(Cl_post))
    Cd = (1 - weight) * Cd_attached + weight * Cd_post
    Cd = max(Cd, Cd_min)
    return Cl, Cd

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
    total_torque = 0.0
    total_thrust = 0.0
    max_aoa, min_aoa = -999, 999
    stall_count = 0
    elements = []
    
    for i, r in enumerate(r_stations):
        frac = (r - R_hub) / (R - R_hub)
        c = (root_chord_mm + (tip_chord_mm - root_chord_mm) * frac) / 1000.0
        z_root_eff = R * 0.1
        if r <= z_root_eff:
            tf = 0.0
        else:
            tf = (1.0/z_root_eff - 1.0/r) / (1.0/z_root_eff - 1.0/R)
        local_twist_deg = root_twist_deg - (root_twist_deg - tip_twist_deg) * tf
        theta = np.radians(local_twist_deg)
        
        if i == 0:
            dr = r_stations[1] - r_stations[0]
        elif i == num_elements - 1:
            dr = r_stations[-1] - r_stations[-2]
        else:
            dr = (r_stations[i+1] - r_stations[i-1]) / 2.0
            
        sigma_r = B * c / (2 * np.pi * r) if r > 0 else 0
        a, a_p = 0.0, 0.0
        converged = False
        
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
            
            if sigma_r * Cn != 0:
                a_new = 1.0 / ((4 * F * sp**2) / (sigma_r * Cn) + 1)
            else:
                a_new = 0
            
            if a_new > 0.4:
                ac = 0.34
                K = 4 * F * sp**2 / (sigma_r * Cn) if sigma_r * Cn != 0 else 1e6
                disc = (K * (1 - 2*ac) + 2)**2 + 4 * (K * ac**2 - 1)
                if disc >= 0:
                    a_new = 0.5 * (2 + K * (1 - 2*ac) - np.sqrt(disc))
                else:
                    a_new = ac
                a_new = max(0, min(a_new, 0.95))
            
            if sigma_r * Ct != 0:
                a_p_new = 1.0 / ((4 * F * sp * cp) / (sigma_r * Ct) - 1)
            else:
                a_p_new = 0
            a_p_new = max(-0.5, min(a_p_new, 2.0))
            a_new = max(0, min(a_new, 0.95))
            
            if abs(a_new - a) < 1e-6 and abs(a_p_new - a_p) < 1e-6 and iteration > 5:
                a, a_p = a_new, a_p_new
                converged = True
                break
                
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
            'alpha_deg': alpha_deg, 'Cl': Cl, 'Cd': Cd, 'V_rel': V_rel,
            'dT': dT, 'dQ': dQ, 'a': a, 'a_p': a_p
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
        'max_aoa': max_aoa, 'min_aoa': min_aoa,
        'stall_count': stall_count, 'status': status, 'elements': elements
    }

def simulate_electrical_full(mech_power_W, rotor_rpm, gear_ratio, load_ohm,
                              gen_v_max=10.0, gen_i_max=0.3, gen_rated_rpm=1500.0,
                              gear_eff=0.95, gen_eff=0.85):
    gen_rpm = rotor_rpm * gear_ratio
    P_shaft = mech_power_W * gear_eff * gen_eff
    R_int = gen_v_max / gen_i_max if gen_i_max > 0 else 999
    V_emf = (gen_rpm / gen_rated_rpm) * gen_v_max if gen_rated_rpm > 0 else 0
    I_raw = V_emf / (R_int + load_ohm) if (R_int + load_ohm) > 0 else 0
    P_raw = I_raw**2 * load_ohm
    
    if P_shaft <= 0:
        return {'gen_rpm': gen_rpm, 'V_emf': V_emf, 'V_terminal': 0,
                'current_A': 0, 'P_elec_W': 0, 'P_elec_mW': 0,
                'R_internal': R_int, 'overall_efficiency': 0, 'stalled': True}
                
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
        'gen_rpm': gen_rpm, 'V_emf': V_emf, 'V_terminal': V_t,
        'current_A': I, 'P_elec_W': P_e, 'P_elec_mW': P_e * 1000,
        'R_internal': R_int, 'overall_efficiency': min(eff, 1.0),
        'stalled': stalled
    }

def calculate_optimal_twist(V_wind, rpm, R_mm, target_aoa=5.0):
    R = R_mm / 1000.0
    omega = rpm * 2 * np.pi / 60.0
    if omega <= 0 or R <= 0:
        return 15.0, 2.0
    r_root = R * 0.15
    rt = np.degrees(np.arctan2(V_wind, omega * r_root)) - target_aoa
    r_tip = R * 0.95
    tt = np.degrees(np.arctan2(V_wind, omega * r_tip)) - target_aoa
    return round(max(-10, min(rt, 45)), 2), round(max(-10, min(tt, 30)), 2)

def get_naca_coords(code, chord_mm, points):
    cs = str(code).strip().zfill(4)
    if len(cs) != 4 or not cs.isdigit(): return [], []
    m = int(cs[0]) / 100.0
    p = int(cs[1]) / 10.0
    t = int(cs[2:]) / 100.0
    beta = np.linspace(0, np.pi, points)
    x = 0.5 * (1 - np.cos(beta))
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 +
                  0.2843 * x**3 - 0.1015 * x**4)
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

# ============================================================
# APP UI & TABS
# ============================================================
st.title("🌪️ Wind Turbine Blade Simulator V8 (Ultimate Precision)")

tab1, tab2, tab3, tab4 = st.tabs([
    "📏 Design & 3D", 
    "🌪️ BEM & Generator Simulation", 
    "🎯 Smart Reverse Design", 
    "📖 Manual & Equations"
])

# ---------------------------------------------------------
# TAB 1: DESIGN & 3D
# ---------------------------------------------------------
with tab1:
    col_input, col_plot = st.columns([1, 2])
    
    with col_input:
        st.subheader("Airfoil & Dimensions")
        naca_code = st.text_input("NACA Code", value="4412")
        blade_L = st.number_input("Span (Length) [mm]", value=st.session_state.blade_L, step=10.0)
        root_c = st.number_input("Root Chord [mm]", value=st.session_state.root_c, step=1.0)
        tip_c = st.number_input("Tip Chord [mm]", value=st.session_state.tip_c, step=1.0)
        
        st.subheader("Twist [deg]")
        root_t = st.number_input("Root Twist", value=st.session_state.root_t, step=0.5)
        tip_t = st.number_input("Tip Twist", value=st.session_state.tip_t, step=0.5)
        
        st.subheader("Mesh Options")
        spacing = st.number_input("Rib Spacing [mm]", value=15.0, step=1.0)
        pts = st.number_input("Points/Rib", value=80, step=10)
        
        # Update session state
        st.session_state.blade_L = blade_L
        st.session_state.root_c = root_c
        st.session_state.tip_c = tip_c
        st.session_state.root_t = root_t
        st.session_state.tip_t = tip_t

    # Generate 3D Coordinates
    rib_data = []
    Xs, Ys, Zs = [], [], []
    if spacing > 0 and blade_L > 0:
        n_ribs = int(np.ceil(blade_L / spacing)) + 1
        z_stations = np.linspace(0, blade_L, n_ribs)
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
                
                rib_data.append({'X': X1, 'Y': Y1, 'Z': Z1})
                Xs.extend(X1)
                Ys.extend(Y1)
                Zs.extend(Z1)
        st.session_state.rib_data = rib_data

    with col_plot:
        st.subheader("3D Blade Model")
        if len(Xs) > 0:
            fig3d = go.Figure(data=[go.Scatter3d(
                x=Xs, y=Zs, z=Ys,
                mode='markers',
                marker=dict(size=1.5, color=Zs, colorscale='Blues', opacity=0.8)
            )])
            fig3d.update_layout(
                margin=dict(l=0, r=0, b=0, t=0),
                scene=dict(
                    xaxis_title='X [mm]',
                    yaxis_title='Span Z [mm]',
                    zaxis_title='Thickness Y [mm]',
                    aspectmode='data'
                )
            )
            st.plotly_chart(fig3d, use_container_width=True)
            
            # Export CSV block (Fusion 360 format)
            st.subheader("💾 Export for Fusion 360")
            csv_buffer = StringIO()
            writer = csv.writer(csv_buffer)
            writer.writerow(["X", "Y", "Z"])
            for rib in rib_data:
                for x, y, z in zip(rib['X'], rib['Y'], rib['Z']):
                    writer.writerow([f"{x:.4f}", f"{y:.4f}", f"{z:.4f}"])
            st.download_button(
                label="⬇️ Download Point Cloud CSV",
                data=csv_buffer.getvalue(),
                file_name=f"Blade_Fusion360_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )

# ---------------------------------------------------------
# TAB 2: SIMULATE & GENERATOR
# ---------------------------------------------------------
with tab2:
    scol1, scol2 = st.columns([1, 2])
    with scol1:
        st.subheader("Environment & Specs")
        v_wind = st.number_input("Wind Speed [m/s]", value=st.session_state.v_wind)
        rpm = st.number_input("Rotor RPM", value=st.session_state.rpm)
        rho = st.number_input("Air Density [kg/m³]", value=1.225)
        num_blades = st.number_input("Number of Blades", value=3, min_value=1, max_value=6)
        
        st.session_state.v_wind = v_wind
        st.session_state.rpm = rpm
        
        st.markdown("---")
        st.subheader("Generator & Load")
        gear_ratio = st.number_input("Gear Ratio (1:X)", value=4.0)
        gen_v = st.number_input("Gen Max Voltage [V]", value=10.0)
        gen_i = st.number_input("Gen Max Current [A]", value=0.3)
        gen_rpm = st.number_input("Gen Rated RPM", value=1500.0)
        load_ohm = st.number_input("Load Resistance [Ω]", value=10.0)
        
        if st.button("🚀 Run BEM Simulation", type="primary"):
            st.session_state.bem_results = run_bem_high_precision(
                v_wind, rpm, st.session_state.blade_L, st.session_state.root_c, st.session_state.tip_c,
                st.session_state.root_t, st.session_state.tip_t, num_blades, rho, 20
            )

    with scol2:
        if st.session_state.bem_results:
            res = st.session_state.bem_results
            total_mech = res['power_W'] * num_blades
            elec = simulate_electrical_full(total_mech, rpm, gear_ratio, load_ohm, gen_v, gen_i, gen_rpm)
            
            st.subheader("📊 Diagnostic Report")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Status", res['status'])
            c2.metric("TSR (λ)", f"{res['TSR']:.2f}")
            c3.metric("Cp (Eff)", f"{res['Cp']:.4f}")
            c4.metric("Mech Power", f"{total_mech*1000:.2f} mW")
            
            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Gen RPM", f"{elec['gen_rpm']:.0f}")
            c6.metric("Elec Power", f"{elec['P_elec_mW']:.2f} mW")
            c7.metric("Terminal V", f"{elec['V_terminal']:.3f} V")
            c8.metric("Current mA", f"{elec['current_A']*1000:.2f} mA")
            
            st.markdown("---")
            st.subheader("📈 Load Matching Curve")
            r_range = np.linspace(0.5, max(50, load_ohm*2), 200)
            p_shaft = total_mech * 0.95 * 0.85
            p_curve = []
            for rr_val in r_range:
                if p_shaft <= 0:
                    p_curve.append(0)
                    continue
                i_r = elec['V_emf'] / (elec['R_internal'] + rr_val)
                p_raw = i_r**2 * rr_val
                p_curve.append(min(p_raw, p_shaft) * 1000)
                
            fig_load, ax = plt.subplots(figsize=(8, 4))
            ax.plot(r_range, p_curve, label='Available Electrical Power (mW)', color='blue')
            ax.scatter([load_ohm], [elec['P_elec_mW']], color='red', s=100, label=f'Operating Point ({load_ohm}Ω)', zorder=5)
            ax.axvline(elec['R_internal'], color='gray', linestyle='--', label=f'Internal R ({elec["R_internal"]:.1f}Ω)')
            ax.set_xlabel('Load Resistance [Ω]')
            ax.set_ylabel('Power [mW]')
            ax.grid(True, linestyle='--', alpha=0.6)
            ax.legend()
            st.pyplot(fig_load)

# ---------------------------------------------------------
# TAB 3: SMART REVERSE DESIGN
# ---------------------------------------------------------
with tab3:
    st.subheader("🎯 Smart Reverse Design (Target Load/Power -> Blade)")
    rcol1, rcol2 = st.columns([1, 2])
    
    with rcol1:
        s_wind = st.number_input("Target Wind [m/s]", value=3.0, key='sw')
        s_load = st.number_input("Target Load [Ω]", value=10.0, key='sl')
        s_target_p = st.number_input("Target Power [mW] (0=Max)", value=0.0, key='sp')
        s_max_L = st.number_input("Max Blade Length [mm]", value=400.0, key='sml')
        
        search_pressed = st.button("🔍 Run Live BEM Search", type="primary")

    with rcol2:
        if search_pressed:
            st.info("Running permutations (this might take a few seconds...)")
            blade_lens = [100, 150, 200, 250, 300, 350, 400]
            root_chords = [30, 50, 70, 90]
            tip_chords = [15, 25, 35, 50]
            rpms = [150, 250, 400, 550]
            
            live_results = []
            for L in [l for l in blade_lens if l <= s_max_L]:
                for rc in root_chords:
                    for tc in tip_chords:
                        if tc >= rc: continue
                        for rpm in rpms:
                            rt, tt = calculate_optimal_twist(s_wind, rpm, L)
                            aero = run_bem_high_precision(s_wind, rpm, L, rc, tc, rt, tt, num_elements=15)
                            if aero['power_W'] <= 0 or aero['status'] in ('SEVERE_STALL', 'NO_POWER', 'INVALID'): continue
                            
                            elec = simulate_electrical_full(aero['power_W'], rpm, 4.0, s_load)
                            if elec['stalled'] or elec['P_elec_mW'] <= 0: continue
                            
                            live_results.append({
                                'Length': L, 'RootChord': rc, 'TipChord': tc, 
                                'RootTwist': rt, 'TipTwist': tt, 'RPM': rpm,
                                'MechPower_mW': aero['power_W']*1000,
                                'ElecPower_mW': elec['P_elec_mW'],
                                'Cp': aero['Cp'],
                                'Status': aero['status']
                            })
            
            if live_results:
                for d in live_results:
                    if s_target_p > 0: d['_score'] = -abs(d['ElecPower_mW'] - s_target_p)
                    else: d['_score'] = d['ElecPower_mW']
                
                live_results.sort(key=lambda x: x['_score'], reverse=True)
                df_res = pd.DataFrame(live_results[:5]).drop(columns=['_score'])
                st.success(f"Found {len(live_results)} valid designs. Showing top 5:")
                st.dataframe(df_res)
                
                best = live_results[0]
                st.markdown(f"### 🏆 Best Design Specs:")
                st.code(f"""
Blade Length: {best['Length']} mm
Root Chord:   {best['RootChord']} mm
Tip Chord:    {best['TipChord']} mm
Root Twist:   {best['RootTwist']}°
Tip Twist:    {best['TipTwist']}°
RPM:          {best['RPM']}
Power (Elec): {best['ElecPower_mW']:.2f} mW
Cp:           {best['Cp']:.4f}
                """, language="text")
            else:
                st.error("No valid design found for these constraints. Try increasing wind or length.")

# ---------------------------------------------------------
# TAB 4: MANUAL & EQUATIONS
# ---------------------------------------------------------
with tab4:
    st.subheader("📖 Equations Reference")
    st.markdown("""
    This simulator implements **Blade Element Momentum (BEM)** theory, the standard method for horizontal-axis wind turbine design.

    ### 1. Fundamental Relationships
    * **Tip Speed Ratio (λ):** `λ = ωR / V_wind`
    * **Available Wind Power:** `P_available = ½ · ρ · A · V_wind³`
    * **Betz Limit:** `Cp_max = 16/27 ≈ 0.5926`

    ### 2. Aerodynamic Coefficients (NACA 4412)
    * Linear Attached Flow: `Cl = 0.45 + (2π) · α_rad`
    * Drag Polar: `Cd = 0.0095 + 0.0007·α + 0.00012·α²`
    * Viterna Post-stall extrapolation is applied for `|α| > 14°`

    ### 3. Induction Factors & BEM
    * **Axial Induction (a):** `a = 1 / (4·F·sin²(φ) / (σ·Cn) + 1)`
    * **Tangential Induction (a'):** `a' = 1 / (4·F·sin(φ)·cos(φ) / (σ·Ct) - 1)`
    * **Prandtl Loss (F):** `F = F_tip · F_hub`
    * **Buhl Correction:** Applied when `a > 0.4` to prevent momentum breakdown.

    ### 4. Generator & Electrical
    * **Energy Conservation Constraint:** `P_elec ≤ P_mech · η_gear · η_gen`
    * **Terminal Voltage:** `V_terminal = I · R_load`
    """)