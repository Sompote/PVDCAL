"""
Streamlit App for PVD Consolidation Analysis
Shows settlement vs time curve with vacuum and staged loading support
"""

import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from pvd_consolidation import PVDConsolidation, SoilLayer, PVDProperties, LoadingStage

# Page configuration
st.set_page_config(
    page_title="PVD Consolidation Analysis", page_icon="📊", layout="wide"
)

# Title
st.title("🏗️ PVD Consolidation Analysis")
st.markdown("### Settlement vs Time Calculator")

# Sidebar for inputs
st.sidebar.header("Input Parameters")

# Soil Layer Inputs
st.sidebar.subheader("Soil Properties")
n_layers = st.sidebar.number_input(
    "Number of Layers", min_value=1, max_value=10, value=1
)

layers = []
for i in range(n_layers):
    st.sidebar.markdown(f"**Layer {i + 1}**")
    col1, col2 = st.sidebar.columns(2)

    with col1:
        thickness = st.number_input(
            f"Thickness (m)", min_value=0.1, value=10.0, step=0.5, key=f"thick_{i}"
        )
        Cv = st.number_input(
            f"Cv (m²/yr)", min_value=0.01, value=0.5, step=0.1, key=f"cv_{i}"
        )
        Ch = st.number_input(
            f"Ch (m²/yr)", min_value=0.01, value=2.0, step=0.1, key=f"ch_{i}"
        )
        RR = st.number_input(
            f"RR", min_value=0.001, value=0.05, step=0.01, format="%.3f", key=f"rr_{i}"
        )
        CR = st.number_input(
            f"CR", min_value=0.001, value=0.30, step=0.01, format="%.3f", key=f"cr_{i}"
        )

    with col2:
        sigma_ini = st.number_input(
            f"σ'ini (kPa)", min_value=1.0, value=50.0, step=5.0, key=f"sini_{i}"
        )
        sigma_p = st.number_input(
            f"σ'p (kPa)", min_value=1.0, value=80.0, step=5.0, key=f"sp_{i}"
        )
        kh = st.number_input(
            f"kh (m/yr)", min_value=0.1, value=2.0, step=0.1, key=f"kh_{i}"
        )
        ks = st.number_input(
            f"ks (m/yr)", min_value=0.1, value=1.0, step=0.1, key=f"ks_{i}"
        )

    layers.append(
        SoilLayer(
            thickness=thickness,
            Cv=Cv,
            Ch=Ch,
            RR=RR,
            CR=CR,
            sigma_ini=sigma_ini,
            sigma_p=sigma_p,
            kh=kh,
            ks=ks,
        )
    )

    st.sidebar.markdown("---")

# PVD Properties
st.sidebar.subheader("PVD Properties")

dw = st.sidebar.number_input(
    "Drain diameter dw (m)", min_value=0.01, value=0.05, step=0.01, format="%.3f"
)
ds = st.sidebar.number_input(
    "Smear zone ds (m)", min_value=0.01, value=0.15, step=0.01, format="%.3f"
)
De = st.sidebar.number_input("Unit cell De (m)", min_value=0.1, value=1.5, step=0.1)
L_drain = st.sidebar.number_input(
    "Drain length L (m)", min_value=1.0, value=10.0, step=1.0
)

well_resistance = st.sidebar.selectbox(
    "Well Resistance", ["Negligible (qw → ∞)", "Custom qw"]
)

if well_resistance == "Custom qw":
    qw = st.sidebar.number_input("qw (m³/yr)", min_value=1.0, value=100.0, step=10.0)
else:
    qw = 1e12  # Very large value for negligible resistance

# Vacuum depth loss option
vacuum_depth_loss = st.sidebar.checkbox(
    "Vacuum loss with depth",
    value=True,
    help="If checked: vacuum decreases with depth (u(z) = u_surface × exp(-z/L)). If unchecked: uniform vacuum at all depths.",
)

pvd = PVDProperties(
    dw=dw, ds=ds, De=De, L_drain=L_drain, qw=qw, vacuum_depth_loss=vacuum_depth_loss
)

# Loading Configuration
st.sidebar.subheader("Loading Configuration")

loading_type = st.sidebar.radio(
    "Loading Type",
    ["Single Stage", "Multi-Stage"],
    help="Single stage: One loading condition. Multi-stage: Multiple loading stages over time.",
)

if loading_type == "Single Stage":
    st.sidebar.markdown("**Single Stage Loading**")
    surcharge = st.sidebar.number_input(
        "Surcharge (kPa)", min_value=0.0, value=100.0, step=10.0
    )
    vacuum = st.sidebar.number_input(
        "Vacuum (kPa)",
        min_value=0.0,
        value=0.0,
        step=10.0,
        help="Vacuum pressure (positive value). 80 kPa vacuum ≈ 80 kPa surcharge",
    )
    loading_stages = None

else:  # Multi-Stage
    st.sidebar.markdown("**Multi-Stage Loading**")
    n_stages = st.sidebar.number_input(
        "Number of Stages", min_value=1, max_value=10, value=2
    )

    loading_stages = []
    for i in range(n_stages):
        st.sidebar.markdown(f"**Stage {i + 1}**")
        col1, col2, col3 = st.sidebar.columns(3)

        with col1:
            start_time = st.number_input(
                f"Time (y)",
                min_value=0.0,
                value=float(i * 0.5),
                step=0.1,
                key=f"time_{i}",
            )
        with col2:
            stage_surcharge = st.number_input(
                f"Surcharge", min_value=0.0, value=50.0, step=10.0, key=f"sur_{i}"
            )
        with col3:
            stage_vacuum = st.number_input(
                f"Vacuum", min_value=0.0, value=0.0, step=10.0, key=f"vac_{i}"
            )

        loading_stages.append(LoadingStage(start_time, stage_surcharge, stage_vacuum))

    surcharge = 0.0  # Not used in multi-stage
    vacuum = 0.0

# Analysis Parameters
st.sidebar.subheader("Analysis Parameters")
t_max = st.sidebar.number_input("Max Time (years)", min_value=0.1, value=2.0, step=0.1)
dt = st.sidebar.number_input(
    "Time Step (years)", min_value=0.001, value=0.01, step=0.001, format="%.3f"
)

# Run Analysis Button
if st.sidebar.button("🚀 Run Analysis", type="primary"):
    with st.spinner("Calculating consolidation..."):
        try:
            # Create analysis with vacuum and staged loading support
            if loading_stages:
                analysis = PVDConsolidation(
                    layers, pvd, loading_stages=loading_stages, dt=dt
                )
            else:
                analysis = PVDConsolidation(
                    layers, pvd, surcharge=surcharge, vacuum=vacuum, dt=dt
                )

            # Calculate settlement vs time
            time, settlement = analysis.settlement_vs_time(t_max=t_max, n_points=200)
            settlement_mm = settlement * 1000  # Convert to mm

            # Get PVD factors
            Fn, Fs, Fr = analysis.calculate_pvd_factors()
            F_total = Fn + Fs + Fr

            # Store results in session state
            st.session_state.time = time
            st.session_state.settlement = settlement_mm
            st.session_state.Fn = Fn
            st.session_state.Fs = Fs
            st.session_state.Fr = Fr
            st.session_state.F_total = F_total
            st.session_state.analysis_done = True

        except Exception as e:
            st.error(f"Error in analysis: {str(e)}")
            st.session_state.analysis_done = False

# Display Results
if hasattr(st.session_state, "analysis_done") and st.session_state.analysis_done:
    # Main plot - Settlement vs Time
    st.subheader("Settlement vs Time")

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(st.session_state.time, st.session_state.settlement, "b-", linewidth=2.5)
    ax.set_xlabel("Time (years)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Settlement (mm)", fontsize=14, fontweight="bold")
    ax.set_title(
        "PVD Consolidation - Settlement vs Time", fontsize=16, fontweight="bold"
    )
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=12)

    # Invert y-axis (settlement goes down)
    ax.invert_yaxis()

    # Add final settlement annotation
    final_settlement = st.session_state.settlement[-1]
    ax.axhline(
        y=final_settlement,
        color="r",
        linestyle="--",
        alpha=0.5,
        label=f"Final Settlement = {final_settlement:.1f} mm",
    )
    ax.legend(fontsize=12)

    st.pyplot(fig)

    # Summary statistics
    st.subheader("Analysis Summary")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Final Settlement", f"{final_settlement:.1f} mm")

    with col2:
        # Find time to 90% consolidation
        target_settlement = 0.9 * final_settlement
        idx_90 = np.argmax(st.session_state.settlement >= target_settlement)
        t_90 = (
            st.session_state.time[idx_90] if idx_90 > 0 else st.session_state.time[-1]
        )
        st.metric("Time to 90% U", f"{t_90:.2f} years")

    with col3:
        # Find time to 50% consolidation
        target_settlement = 0.5 * final_settlement
        idx_50 = np.argmax(st.session_state.settlement >= target_settlement)
        t_50 = (
            st.session_state.time[idx_50] if idx_50 > 0 else st.session_state.time[-1]
        )
        st.metric("Time to 50% U", f"{t_50:.2f} years")

    with col4:
        st.metric("Total Resistance F", f"{st.session_state.F_total:.2f}")

    # PVD Factors
    st.subheader("PVD Influence Factors")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Geometric Factor (Fn)", f"{st.session_state.Fn:.3f}")

    with col2:
        st.metric("Smear Factor (Fs)", f"{st.session_state.Fs:.3f}")

    with col3:
        st.metric("Well Resistance (Fr)", f"{st.session_state.Fr:.3f}")

    # Download data
    st.subheader("Download Results")

    # Prepare CSV data
    csv_data = "Time (years),Settlement (mm)\n"
    for t, s in zip(st.session_state.time, st.session_state.settlement):
        csv_data += f"{t:.4f},{s:.2f}\n"

    st.download_button(
        label="📥 Download Settlement Data (CSV)",
        data=csv_data,
        file_name="settlement_data.csv",
        mime="text/csv",
    )

else:
    # Initial instructions
    st.info(
        "👈 Configure your parameters in the sidebar and click **Run Analysis** to see results"
    )

    st.markdown("""
    ### Instructions:

    1. **Soil Properties**: Enter the properties for each soil layer
       - Thickness (m)
       - Cv: Vertical coefficient of consolidation (m²/year)
       - Ch: Horizontal coefficient of consolidation (m²/year)
       - RR: Recompression ratio
       - CR: Compression ratio
       - σ'ini: Initial effective stress (kPa)
       - σ'p: Preconsolidation pressure (kPa)

    2. **PVD Properties**: Configure the drain installation
       - dw: Equivalent drain diameter (m)
       - ds: Smear zone diameter (m)
       - De: Equivalent unit cell diameter (m)
       - L: Drain length (m)
       - kh: Horizontal permeability (m/year)
       - ks: Smear zone permeability (m/year)
       - qw: Well discharge capacity (m³/year)

    3. **Analysis Parameters**:
       - Surcharge: Applied load (kPa)
       - Max Time: Analysis duration (years)
       - Time Step: Calculation step size (years)

    4. Click **Run Analysis** to calculate and visualize results
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("**PVD Consolidation Calculator**")
st.sidebar.markdown("Version 1.0")
