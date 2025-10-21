"""
Multilayer Prefabricated Vertical Drain (PVD) Consolidation Analysis
Calculates settlement vs time using finite difference method
"""

import numpy as np
import matplotlib.pyplot as plt
import yaml
import argparse
import os
from dataclasses import dataclass
from typing import List, Tuple, Dict, Any


@dataclass
class LoadingStage:
    """Loading stage definition for staged loading"""

    start_time: float  # Time when this stage starts (years)
    surcharge: float  # Surcharge load (kPa)
    vacuum: float  # Vacuum pressure (kPa, positive value, will be applied as negative)

    def __post_init__(self):
        """Validate inputs"""
        if self.vacuum < 0:
            raise ValueError(
                "Vacuum should be specified as positive value (e.g., 80 for -80 kPa)"
            )


@dataclass
class SoilLayer:
    """Soil layer properties"""

    thickness: float  # Layer thickness (m)
    Cv: float  # Vertical coefficient of consolidation (m²/year)
    Ch: float  # Horizontal coefficient of consolidation (m²/year)
    RR: float  # Recompression ratio
    CR: float  # Compression ratio
    sigma_ini: float  # Initial effective stress (kPa)
    sigma_p: float  # Preconsolidation pressure (kPa)
    kh: float  # Horizontal permeability (m/year)
    ks: float  # Smear zone permeability (m/year)


@dataclass
class PVDProperties:
    """PVD installation properties"""

    dw: float  # Equivalent diameter of drain (m)
    ds: float  # Smear zone diameter (m)
    De: float  # Equivalent diameter of unit cell (m)
    L_drain: float  # Total drain spacing for two-way drainage (m)
    qw: float  # Well discharge capacity (m³/year)
    r_influence: float = None  # Vacuum influence radius (m), default = De/4
    vacuum_depth_loss: bool = True  # Use vacuum loss with depth (default: True)

    def __post_init__(self):
        """Set default vacuum influence radius"""
        if self.r_influence is None:
            self.r_influence = self.De / 4


class PVDConsolidation:
    """
    Multilayer PVD consolidation analysis using finite difference method
    """

    def __init__(
        self,
        soil_layers: List[SoilLayer],
        pvd: PVDProperties,
        surcharge: float = 0.0,
        vacuum: float = 0.0,
        loading_stages: List[LoadingStage] = None,
        dt: float = 0.01,
    ):
        """
        Initialize PVD consolidation analysis

        Parameters:
        -----------
        soil_layers : List[SoilLayer]
            List of soil layers from top to bottom
        pvd : PVDProperties
            PVD installation properties
        surcharge : float
            Applied surcharge load (kPa) - for single-stage loading
        vacuum : float
            Applied vacuum pressure (kPa, positive value) - for single-stage loading
        loading_stages : List[LoadingStage]
            List of loading stages for multi-stage loading (overrides surcharge/vacuum)
        dt : float
            Time step for finite difference (years)
        """
        self.layers = soil_layers
        self.pvd = pvd
        self.dt = dt

        # Setup loading
        if loading_stages is not None:
            self.loading_stages = sorted(loading_stages, key=lambda x: x.start_time)
            self.use_staged_loading = True
        else:
            # Single stage loading
            self.loading_stages = [
                LoadingStage(start_time=0.0, surcharge=surcharge, vacuum=vacuum)
            ]
            self.use_staged_loading = False

        # For compatibility
        self.surcharge = surcharge
        self.vacuum = vacuum

        # Calculate total thickness
        self.total_thickness = sum(layer.thickness for layer in soil_layers)

        # Initialize mesh
        self._setup_mesh()

    def _setup_mesh(self, nodes_per_meter: int = 10):
        """Setup finite difference mesh"""
        self.nodes_per_meter = nodes_per_meter

        # Create mesh for each layer
        self.z_coords = []
        self.layer_indices = []
        self.Cv_profile = []
        self.Ch_profile = []
        self.kh_profile = []
        self.ks_profile = []

        z = 0
        for i, layer in enumerate(self.layers):
            n_nodes = max(int(layer.thickness * nodes_per_meter), 2)
            z_layer = np.linspace(z, z + layer.thickness, n_nodes, endpoint=False)

            self.z_coords.extend(z_layer)
            self.layer_indices.extend([i] * len(z_layer))
            self.Cv_profile.extend([layer.Cv] * len(z_layer))
            self.Ch_profile.extend([layer.Ch] * len(z_layer))
            self.kh_profile.extend([layer.kh] * len(z_layer))
            self.ks_profile.extend([layer.ks] * len(z_layer))

            z += layer.thickness

        # Add final node
        self.z_coords.append(self.total_thickness)
        self.layer_indices.append(len(self.layers) - 1)
        self.Cv_profile.append(self.layers[-1].Cv)
        self.Ch_profile.append(self.layers[-1].Ch)
        self.kh_profile.append(self.layers[-1].kh)
        self.ks_profile.append(self.layers[-1].ks)

        self.z_coords = np.array(self.z_coords)
        self.layer_indices = np.array(self.layer_indices)
        self.Cv_profile = np.array(self.Cv_profile)
        self.Ch_profile = np.array(self.Ch_profile)
        self.kh_profile = np.array(self.kh_profile)
        self.ks_profile = np.array(self.ks_profile)

        self.n_nodes = len(self.z_coords)
        self.dz = np.diff(self.z_coords)

    def get_current_loading(self, t: float) -> Tuple[float, float]:
        """
        Get current surcharge and vacuum at time t

        Parameters:
        -----------
        t : float
            Current time (years)

        Returns:
        --------
        surcharge, vacuum : float, float
            Current surcharge and vacuum values (kPa)
        """
        # Find the active loading stage
        active_stage = self.loading_stages[0]
        for stage in self.loading_stages:
            if t >= stage.start_time:
                active_stage = stage
            else:
                break

        return active_stage.surcharge, active_stage.vacuum

    def calculate_vacuum_at_radius(self, r: float, vacuum_magnitude: float) -> float:
        """
        Calculate vacuum pressure at radial distance r from drain

        u(r) = u_drain × exp(-r/r_influence)

        Parameters:
        -----------
        r : float
            Radial distance from drain (m)
        vacuum_magnitude : float
            Vacuum at drain (kPa, positive value)

        Returns:
        --------
        vacuum_at_r : float
            Vacuum pressure at distance r (kPa, positive value)
        """
        if vacuum_magnitude == 0:
            return 0.0

        # Vacuum decreases exponentially with distance
        vacuum_at_r = vacuum_magnitude * np.exp(-r / self.pvd.r_influence)

        return vacuum_at_r

    def calculate_layer_average_vacuum(
        self, layer_idx: int, vacuum_surface: float
    ) -> float:
        """
        Calculate average vacuum pressure for a specific soil layer

        If vacuum_depth_loss = True:
            Vacuum is full at surface and decreases with depth
            u(z) = u_surface × exp(-z/L_drain)

        If vacuum_depth_loss = False:
            Uniform vacuum throughout (full vacuum at all depths within L_drain)

        Parameters:
        -----------
        layer_idx : int
            Layer index
        vacuum_surface : float
            Vacuum at surface (kPa, positive value)

        Returns:
        --------
        avg_vacuum : float
            Average vacuum for this layer (kPa)
        """
        if vacuum_surface == 0:
            return 0.0

        layer = self.layers[layer_idx]

        # Find layer depth range
        depth_top = sum(self.layers[i].thickness for i in range(layer_idx))
        depth_bottom = depth_top + layer.thickness

        # Check if layer is within drain length
        if depth_top >= self.pvd.L_drain:
            # Layer is completely below drain length - no vacuum effect
            return 0.0

        # Option 1: Uniform vacuum (no depth loss)
        if not self.pvd.vacuum_depth_loss:
            # Full vacuum throughout the layer (within drain length)
            return vacuum_surface

        # Option 2: Vacuum loss with depth
        # Adjust bottom depth if it exceeds drain length
        if depth_bottom > self.pvd.L_drain:
            depth_bottom = self.pvd.L_drain
            effective_thickness = depth_bottom - depth_top
        else:
            effective_thickness = layer.thickness

        # Integrate vacuum over layer thickness
        # u(z) = u_surface × exp(-z/L_drain)
        # Average = (1/h) ∫[z_top to z_bottom] u_surface × exp(-z/L_drain) dz

        L = self.pvd.L_drain

        # Analytical integration:
        # ∫ u_surface × exp(-z/L) dz = -u_surface × L × exp(-z/L) + C

        integral_bottom = -vacuum_surface * L * np.exp(-depth_bottom / L)
        integral_top = -vacuum_surface * L * np.exp(-depth_top / L)

        integral = integral_bottom - integral_top

        # Average over effective layer thickness
        avg_vacuum = integral / effective_thickness

        return avg_vacuum

    def calculate_pvd_factors_layer(self, layer_idx: int) -> Tuple[float, float, float]:
        """
        Calculate PVD influence factors for a specific layer

        Parameters:
        -----------
        layer_idx : int
            Layer index

        Returns:
        --------
        Fn, Fs, Fr : float
            Geometric, smear, and well resistance factors for the layer
        """
        layer = self.layers[layer_idx]

        # Geometric factor (n ratio) - same for all layers
        n = self.pvd.De / self.pvd.dw
        Fn = (n**2 / (n**2 - 1)) * np.log(n) - 3 / 4 + 1 / n**2

        # Fs - Smear effect factor (layer-specific)
        s = self.pvd.ds / self.pvd.dw
        Fs = ((layer.kh / layer.ks) - 1) * np.log(s)

        # Fr - Well resistance factor (layer-specific)
        L = self.pvd.L_drain
        if self.pvd.qw > 1e10:  # If qw is very large, assume negligible well resistance
            Fr = 0.0
        else:
            Fr = (np.pi * L**2 * layer.kh) / (8 * self.pvd.qw)

        return Fn, Fs, Fr

    def calculate_pvd_factors(self) -> Tuple[float, float, float]:
        """
        Calculate PVD influence factors using weighted average permeabilities

        Returns:
        --------
        Fn, Fs, Fr : float
            Geometric, smear, and well resistance factors
        """
        # Use thickness-weighted average kh and ks
        kh_avg = np.average(self.kh_profile, weights=np.ones(len(self.kh_profile)))
        ks_avg = np.average(self.ks_profile, weights=np.ones(len(self.ks_profile)))

        # Geometric factor (n ratio)
        n = self.pvd.De / self.pvd.dw

        # Fn - Geometric factor
        Fn = (n**2 / (n**2 - 1)) * np.log(n) - 3 / 4 + 1 / n**2

        # Fs - Smear effect factor
        s = self.pvd.ds / self.pvd.dw
        Fs = ((kh_avg / ks_avg) - 1) * np.log(s)

        # Fr - Well resistance factor
        # For typical band drains: Fr = π(2L-l)l/(qw·kh) where l = L/2
        # Simplified for two-way drainage
        L = self.pvd.L_drain
        if self.pvd.qw > 1e10:  # If qw is very large, assume negligible well resistance
            Fr = 0.0
        else:
            # Using Hansbo formula: Fr = (L²/(8·qw))·(kh)
            # More typical: Fr = π·L²·kh/(8·qw) for two-way drainage
            Fr = (np.pi * L**2 * kh_avg) / (8 * self.pvd.qw)

        return Fn, Fs, Fr

    def calculate_Uh(self, t: float) -> np.ndarray:
        """
        Calculate degree of consolidation in horizontal direction
        using finite difference method with layer-specific PVD factors
        Only applies to layers within drain length L

        Parameters:
        -----------
        t : float
            Time (years)

        Returns:
        --------
        Uh : ndarray
            Degree of horizontal consolidation at each node
            (0 for nodes beyond drain length)
        """
        # Pre-calculate F_total for each layer
        F_total_layers = []
        for layer_idx in range(len(self.layers)):
            Fn, Fs, Fr = self.calculate_pvd_factors_layer(layer_idx)
            F_total_layers.append(Fn + Fs + Fr)

        # Initialize excess pore pressure (normalized)
        u = np.ones(self.n_nodes)  # Initially all excess pore pressure = surcharge
        u_history = [u.copy()]

        # Time stepping
        n_steps = int(t / self.dt)

        for step in range(n_steps):
            u_new = u.copy()

            # Finite difference for radial consolidation
            # ∂u/∂t = Ch * [∂²u/∂r² + (1/r)(∂u/∂r)]
            # Simplified for radial drainage with PVD

            for i in range(1, self.n_nodes - 1):
                # Check if this node is within drain length
                depth = self.z_coords[i]

                if depth <= self.pvd.L_drain:
                    # Within drain length - apply horizontal consolidation
                    Ch = self.Ch_profile[i]
                    layer_idx = self.layer_indices[i]
                    F_total = F_total_layers[layer_idx]

                    # Radial drainage term (simplified)
                    # Using equivalent time factor approach
                    Th = Ch * self.dt / (self.pvd.De**2 / 4)

                    # Update excess pore pressure
                    decay_rate = 8 * Th / F_total
                    u_new[i] = u[i] * np.exp(-decay_rate)
                else:
                    # Beyond drain length - no horizontal consolidation
                    # u remains unchanged (Uh = 0)
                    pass

            # Boundary conditions
            u_new[0] = 0  # Top drainage
            u_new[-1] = 0  # Bottom drainage (if two-way)

            u = u_new

            if step % max(1, n_steps // 100) == 0:
                u_history.append(u.copy())

        # Calculate Uh (degree of consolidation)
        Uh = 1 - u

        return Uh

    def calculate_Uv(self, t: float) -> np.ndarray:
        """
        Calculate degree of consolidation in vertical direction

        For layers below PVD: drainage path is from layer midpoint to bottom of PVD
        For layers within/above PVD: normal two-way drainage

        Parameters:
        -----------
        t : float
            Time (years)

        Returns:
        --------
        Uv : ndarray
            Degree of vertical consolidation at each node
        """
        Uv = np.zeros(self.n_nodes)

        for i in range(self.n_nodes):
            depth = self.z_coords[i]
            Cv = self.Cv_profile[i]
            layer_idx = self.layer_indices[i]

            # Determine drainage path length H
            if depth > self.pvd.L_drain:
                # Below PVD: drainage from this point UP to bottom of PVD
                # For layer below PVD, use distance from layer midpoint to PVD bottom

                # Find layer boundaries
                cumulative_depth = 0
                for idx in range(layer_idx):
                    cumulative_depth += self.layers[idx].thickness
                layer_top = cumulative_depth
                layer_bottom = cumulative_depth + self.layers[layer_idx].thickness
                layer_midpoint = (layer_top + layer_bottom) / 2

                # Drainage path: from layer midpoint to PVD bottom
                H = abs(layer_midpoint - self.pvd.L_drain)

                # One-way drainage (upward only to PVD)
                drainage_type = "one-way"
            else:
                # Within or above PVD: normal two-way drainage
                H = self.total_thickness
                drainage_type = "two-way"

            # Time factor
            if H > 0:
                Tv = Cv * t / H**2
            else:
                Tv = 1e10  # Very large, instant drainage

            # Calculate Uv using Terzaghi's solution
            if drainage_type == "one-way":
                # One-way drainage (for layers below PVD)
                if Tv < 0.05:
                    Uv[i] = np.sqrt(4 * Tv / np.pi)
                else:
                    Uv[i] = 1 - (8 / np.pi**2) * np.exp(-(np.pi**2) * Tv / 4)
            else:
                # Two-way drainage (for layers within PVD zone)
                if Tv < 0.217:
                    Uv[i] = np.sqrt(4 * Tv / np.pi)
                else:
                    Uv[i] = 1 - (8 / np.pi**2) * np.exp(-(np.pi**2) * Tv / 4)

            Uv[i] = min(Uv[i], 1.0)

        return Uv

    def calculate_total_U(self, t: float) -> np.ndarray:
        """
        Calculate total degree of consolidation (combined vertical and horizontal)

        For nodes within drain length: U = 1 - (1 - Uh)(1 - Uv)
        For nodes beyond drain length: U = Uv (only vertical consolidation)

        Parameters:
        -----------
        t : float
            Time (years)

        Returns:
        --------
        U : ndarray
            Total degree of consolidation at each node
        """
        Uh = self.calculate_Uh(t)
        Uv = self.calculate_Uv(t)

        U = np.zeros(self.n_nodes)

        for i in range(self.n_nodes):
            depth = self.z_coords[i]

            if depth <= self.pvd.L_drain:
                # Within drain length - combined vertical and horizontal
                U[i] = 1 - (1 - Uh[i]) * (1 - Uv[i])
            else:
                # Beyond drain length - only vertical consolidation
                U[i] = Uv[i]

        return U

    def calculate_settlement(self, t: float) -> Tuple[float, np.ndarray]:
        """
        Calculate total settlement at time t
        For staged loading: calculates settlement contribution from each stage

        Parameters:
        -----------
        t : float
            Time (years)

        Returns:
        --------
        total_settlement : float
            Total settlement (m)
        layer_settlements : ndarray
            Settlement for each layer (m)
        """
        layer_settlements = np.zeros(len(self.layers))

        # For each loading stage, calculate its contribution to settlement
        for stage_idx, stage in enumerate(self.loading_stages):
            if t < stage.start_time:
                # This stage hasn't started yet
                continue

            # Time since this stage started
            t_stage = t - stage.start_time

            # Calculate consolidation degree for this stage's time duration
            U_stage = self.calculate_total_U(t_stage)

            # Calculate settlement for each layer from this load increment
            for i, layer in enumerate(self.layers):
                # Find nodes in this layer
                layer_mask = self.layer_indices == i
                U_layer = np.mean(U_stage[layer_mask])

                # Get layer-specific vacuum (decreases with depth)
                # Previous stage vacuum for this layer
                if stage_idx == 0:
                    avg_vacuum_prev_layer = 0.0
                else:
                    prev_stage = self.loading_stages[stage_idx - 1]
                    avg_vacuum_prev_layer = self.calculate_layer_average_vacuum(
                        i, prev_stage.vacuum
                    )

                # Current stage vacuum for this layer
                avg_vacuum_current_layer = self.calculate_layer_average_vacuum(
                    i, stage.vacuum
                )

                # Previous stage load for this layer
                if stage_idx == 0:
                    prev_surcharge = 0.0
                    sigma_prev_layer = 0.0
                else:
                    prev_surcharge = self.loading_stages[stage_idx - 1].surcharge
                    sigma_prev_layer = prev_surcharge + avg_vacuum_prev_layer

                # Current stage load for this layer
                sigma_current_layer = stage.surcharge + avg_vacuum_current_layer

                # Load increment for this layer
                delta_sigma_layer = sigma_current_layer - sigma_prev_layer

                if abs(delta_sigma_layer) < 1e-6:
                    # No significant load change in this layer
                    continue

                # Stress before this stage
                sigma_before = layer.sigma_ini + sigma_prev_layer

                # Stress after this stage (if fully consolidated)
                sigma_after = layer.sigma_ini + sigma_current_layer

                # Calculate ultimate settlement for this load increment
                Sc_increment = 0.0

                if delta_sigma_layer > 0:
                    # Loading increment
                    if sigma_after <= layer.sigma_p:
                        # All in recompression range
                        Sc_increment = (
                            layer.RR
                            * np.log10(sigma_after / sigma_before)
                            * layer.thickness
                        )
                    elif sigma_before >= layer.sigma_p:
                        # All in virgin compression range
                        Sc_increment = (
                            layer.CR
                            * np.log10(sigma_after / sigma_before)
                            * layer.thickness
                        )
                    else:
                        # Crosses from recompression to virgin compression
                        Sc_recomp = (
                            layer.RR
                            * np.log10(layer.sigma_p / sigma_before)
                            * layer.thickness
                        )
                        Sc_virgin = (
                            layer.CR
                            * np.log10(sigma_after / layer.sigma_p)
                            * layer.thickness
                        )
                        Sc_increment = Sc_recomp + Sc_virgin
                else:
                    # Unloading (e.g., vacuum removal) - use recompression/swelling
                    Sc_increment = (
                        layer.RR
                        * np.log10(sigma_after / sigma_before)
                        * layer.thickness
                    )

                # Add this stage's contribution (with consolidation degree)
                layer_settlements[i] += U_layer * Sc_increment

        total_settlement = np.sum(layer_settlements)

        return total_settlement, layer_settlements

    def settlement_vs_time(
        self, t_max: float, n_points: int = 100
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate settlement vs time curve

        Parameters:
        -----------
        t_max : float
            Maximum time (years)
        n_points : int
            Number of time points

        Returns:
        --------
        time : ndarray
            Time array (years)
        settlement : ndarray
            Settlement array (m)
        """
        time = np.linspace(0, t_max, n_points)
        settlement = np.zeros(n_points)

        for i, t in enumerate(time):
            if t == 0:
                settlement[i] = 0
            else:
                settlement[i], _ = self.calculate_settlement(t)

        return time, settlement

    def plot_settlement_vs_time(
        self, t_max: float, n_points: int = 100, save_path: str = None
    ):
        """
        Plot settlement vs time curve

        Parameters:
        -----------
        t_max : float
            Maximum time (years)
        n_points : int
            Number of time points
        save_path : str, optional
            Path to save the plot
        """
        time, settlement = self.settlement_vs_time(t_max, n_points)

        # Convert to mm
        settlement_mm = settlement * 1000

        plt.figure(figsize=(10, 6))
        plt.plot(time, settlement_mm, "b-", linewidth=2)
        plt.xlabel("Time (years)", fontsize=12)
        plt.ylabel("Settlement (mm)", fontsize=12)
        plt.title(
            "Settlement vs Time - PVD Consolidation", fontsize=14, fontweight="bold"
        )
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

        return time, settlement_mm

    def plot_degree_of_consolidation(self, t: float, save_path: str = None):
        """
        Plot degree of consolidation profile at time t

        Parameters:
        -----------
        t : float
            Time (years)
        save_path : str, optional
            Path to save the plot
        """
        Uh = self.calculate_Uh(t)
        Uv = self.calculate_Uv(t)
        U = self.calculate_total_U(t)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        # Plot 1: Degree of consolidation profiles
        ax1.plot(Uh * 100, self.z_coords, "r-", linewidth=2, label="Horizontal (Uh)")
        ax1.plot(Uv * 100, self.z_coords, "b-", linewidth=2, label="Vertical (Uv)")
        ax1.plot(U * 100, self.z_coords, "g-", linewidth=2, label="Total (U)")

        # Add drain length line
        ax1.axhline(
            y=self.pvd.L_drain,
            color="orange",
            linestyle="--",
            linewidth=1.5,
            alpha=0.7,
            label=f"Drain Length = {self.pvd.L_drain:.1f} m",
        )

        ax1.set_xlabel("Degree of Consolidation (%)", fontsize=12)
        ax1.set_ylabel("Depth (m)", fontsize=12)
        ax1.set_title(
            f"Consolidation Profile at t = {t:.2f} years",
            fontsize=12,
            fontweight="bold",
        )
        ax1.invert_yaxis()
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Excess pore pressure
        u = 1 - U
        ax2.plot(u * self.surcharge, self.z_coords, "k-", linewidth=2)
        ax2.set_xlabel("Excess Pore Pressure (kPa)", fontsize=12)
        ax2.set_ylabel("Depth (m)", fontsize=12)
        ax2.set_title(
            f"Excess Pore Pressure at t = {t:.2f} years", fontsize=12, fontweight="bold"
        )
        ax2.invert_yaxis()
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")

        plt.show()

    def get_summary_report(self, t_check: List[float]) -> str:
        """
        Generate summary report for specified time points

        Parameters:
        -----------
        t_check : List[float]
            Time points to check (years)

        Returns:
        --------
        report : str
            Summary report
        """
        Fn, Fs, Fr = self.calculate_pvd_factors()

        report = "=" * 70 + "\n"
        report += "PVD CONSOLIDATION ANALYSIS SUMMARY\n"
        report += "=" * 70 + "\n\n"

        report += "PVD PARAMETERS:\n"
        report += f"  Equivalent drain diameter (dw): {self.pvd.dw:.3f} m\n"
        report += f"  Smear zone diameter (ds): {self.pvd.ds:.3f} m\n"
        report += f"  Unit cell diameter (De): {self.pvd.De:.3f} m\n"
        report += f"  Drain length (L): {self.pvd.L_drain:.2f} m\n"
        report += (
            f"  Drain spacing ratio (n = De/dw): {self.pvd.De / self.pvd.dw:.2f}\n"
        )
        report += f"  Geometric factor (Fn): {Fn:.4f}\n"
        report += f"  Smear factor (Fs - avg): {Fs:.4f}\n"
        report += f"  Well resistance (Fr - avg): {Fr:.4f}\n"
        report += f"  Total resistance (F - avg): {Fn + Fs + Fr:.4f}\n\n"

        report += "SOIL PROFILE:\n"
        cumulative_depth = 0
        for i, layer in enumerate(self.layers):
            depth_top = cumulative_depth
            depth_bottom = cumulative_depth + layer.thickness

            # Check if layer is within drain length
            if depth_bottom <= self.pvd.L_drain:
                drain_status = "Full PVD effect (Uh + Uv)"
            elif depth_top >= self.pvd.L_drain:
                drain_status = "No PVD effect (Uv only)"
            else:
                drain_status = "Partial PVD effect"

            report += f"  Layer {i + 1} ({depth_top:.1f}m - {depth_bottom:.1f}m): {drain_status}\n"
            report += f"    Thickness: {layer.thickness:.2f} m\n"
            report += f"    Ch: {layer.Ch:.4f} m²/year\n"
            report += f"    Cv: {layer.Cv:.4f} m²/year\n"
            report += f"    kh: {layer.kh:.4f} m/year\n"
            report += f"    ks: {layer.ks:.4f} m/year\n"
            report += f"    RR: {layer.RR:.4f}\n"
            report += f"    CR: {layer.CR:.4f}\n"
            report += f"    σ'ini: {layer.sigma_ini:.1f} kPa\n"
            report += f"    σ'p: {layer.sigma_p:.1f} kPa\n\n"

            cumulative_depth = depth_bottom

        report += f"Applied surcharge: {self.surcharge:.1f} kPa\n\n"
        report += "=" * 70 + "\n"
        report += "SETTLEMENT vs TIME:\n"
        report += "=" * 70 + "\n"
        report += f"{'Time (years)':<15} {'Settlement (mm)':<20} {'U (%)':<15}\n"
        report += "-" * 70 + "\n"

        for t in t_check:
            settlement, _ = self.calculate_settlement(t)
            U = self.calculate_total_U(t)
            U_avg = np.mean(U)
            report += f"{t:<15.2f} {settlement * 1000:<20.2f} {U_avg * 100:<15.1f}\n"

        report += "=" * 70 + "\n"

        return report


def example_usage():
    """Example usage of PVD consolidation analysis"""

    # Define soil layers (from top to bottom)
    layers = [
        SoilLayer(
            thickness=5.0,  # 5 m thick
            Cv=0.5,  # 0.5 m²/year vertical consolidation
            Ch=1.5,  # 1.5 m²/year horizontal consolidation
            RR=0.05,  # Recompression ratio
            CR=0.30,  # Compression ratio
            sigma_ini=50.0,  # Initial effective stress 50 kPa
            sigma_p=80.0,  # Preconsolidation pressure 80 kPa
            kh=2.0,  # 2 m/year horizontal permeability
            ks=1.0,  # 1 m/year smear zone permeability
        ),
        SoilLayer(
            thickness=8.0,  # 8 m thick
            Cv=0.3,
            Ch=1.0,
            RR=0.04,
            CR=0.35,
            sigma_ini=90.0,
            sigma_p=90.0,
            kh=1.5,  # Lower permeability
            ks=0.75,
        ),
        SoilLayer(
            thickness=7.0,  # 7 m thick
            Cv=0.4,
            Ch=1.2,
            RR=0.045,
            CR=0.32,
            sigma_ini=140.0,
            sigma_p=150.0,
            kh=1.8,
            ks=0.9,
        ),
    ]

    # Define PVD properties
    pvd = PVDProperties(
        dw=0.05,  # 50 mm equivalent drain diameter
        ds=0.15,  # 150 mm smear zone diameter
        De=1.5,  # 1.5 m equivalent unit cell diameter (triangular spacing)
        L_drain=20.0,  # 20 m total drain length (two-way drainage)
        qw=100.0,  # 100 m³/year well discharge capacity
    )

    # Applied surcharge
    surcharge = 100.0  # 100 kPa

    # Create analysis object
    analysis = PVDConsolidation(layers, pvd, surcharge, dt=0.01)

    # Generate summary report
    t_check = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0]
    print(analysis.get_summary_report(t_check))

    # Plot settlement vs time
    print("\nGenerating settlement vs time plot...")
    time, settlement = analysis.plot_settlement_vs_time(
        t_max=20.0, n_points=200, save_path="settlement_vs_time.png"
    )

    # Plot consolidation profiles at different times
    print("Generating consolidation profile plots...")
    for t in [0.5, 2.0, 10.0]:
        analysis.plot_degree_of_consolidation(
            t, save_path=f"consolidation_profile_t{t:.1f}.png"
        )

    print("\nAnalysis complete!")


def load_yaml_data(yaml_file: str) -> Dict[str, Any]:
    """
    Load PVD analysis data from YAML file

    Parameters:
    -----------
    yaml_file : str
        Path to YAML file

    Returns:
    --------
    data : dict
        Dictionary containing analysis parameters
    """
    with open(yaml_file, "r") as f:
        data = yaml.safe_load(f)
    return data


def create_analysis_from_yaml(yaml_file: str) -> PVDConsolidation:
    """
    Create PVDConsolidation object from YAML file

    Parameters:
    -----------
    yaml_file : str
        Path to YAML file

    Returns:
    --------
    analysis : PVDConsolidation
        PVD consolidation analysis object
    """
    data = load_yaml_data(yaml_file)

    # Create soil layers
    layers = []
    for layer_data in data["soil_layers"]:
        layer = SoilLayer(
            thickness=layer_data["thickness"],
            Cv=layer_data["Cv"],
            Ch=layer_data["Ch"],
            RR=layer_data["RR"],
            CR=layer_data["CR"],
            sigma_ini=layer_data["sigma_ini"],
            sigma_p=layer_data["sigma_p"],
            kh=layer_data["kh"],
            ks=layer_data["ks"],
        )
        layers.append(layer)

    # Create PVD properties
    pvd_data = data["pvd"]
    pvd = PVDProperties(
        dw=pvd_data["dw"],
        ds=pvd_data["ds"],
        De=pvd_data["De"],
        L_drain=pvd_data["L_drain"],
        qw=pvd_data["qw"],
    )

    # Get analysis parameters
    surcharge = data["analysis"]["surcharge"]
    dt = data["analysis"].get("dt", 0.01)

    # Create analysis object
    analysis = PVDConsolidation(layers, pvd, surcharge, dt)

    return analysis, data


def run_analysis_from_yaml(yaml_file: str, output_dir: str = None):
    """
    Run complete PVD analysis from YAML file

    Parameters:
    -----------
    yaml_file : str
        Path to YAML input file
    output_dir : str, optional
        Directory to save output files
    """
    print(f"Loading data from: {yaml_file}")
    analysis, data = create_analysis_from_yaml(yaml_file)

    # Create output directory
    if output_dir is None:
        output_dir = os.path.dirname(yaml_file) or "."
    os.makedirs(output_dir, exist_ok=True)

    # Get analysis parameters
    analysis_params = data["analysis"]
    t_max = analysis_params.get("t_max", 20.0)
    n_points = analysis_params.get("n_points", 200)
    t_check = analysis_params.get("t_check", [0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0])
    t_profiles = analysis_params.get("t_profiles", [0.5, 2.0, 10.0])

    # Generate summary report
    print("\n" + "=" * 70)
    print("RUNNING PVD CONSOLIDATION ANALYSIS")
    print("=" * 70 + "\n")

    report = analysis.get_summary_report(t_check)
    print(report)

    # Save report to file
    report_file = os.path.join(output_dir, "pvd_analysis_report.txt")
    with open(report_file, "w") as f:
        f.write(report)
    print(f"\nReport saved to: {report_file}")

    # Plot settlement vs time
    print("\nGenerating settlement vs time plot...")
    settlement_plot = os.path.join(output_dir, "settlement_vs_time.png")
    time, settlement = analysis.plot_settlement_vs_time(
        t_max=t_max, n_points=n_points, save_path=settlement_plot
    )
    print(f"Plot saved to: {settlement_plot}")

    # Save settlement data to CSV
    csv_file = os.path.join(output_dir, "settlement_data.csv")
    np.savetxt(
        csv_file,
        np.column_stack((time, settlement * 1000)),
        delimiter=",",
        header="Time (years),Settlement (mm)",
        comments="",
    )
    print(f"Data saved to: {csv_file}")

    # Plot consolidation profiles at different times
    print("\nGenerating consolidation profile plots...")
    for t in t_profiles:
        profile_plot = os.path.join(output_dir, f"consolidation_profile_t{t:.1f}y.png")
        analysis.plot_degree_of_consolidation(t, save_path=profile_plot)
        print(f"Profile at t={t:.1f} years saved to: {profile_plot}")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE!")
    print("=" * 70)


def main():
    """Main CLI interface"""
    parser = argparse.ArgumentParser(
        description="PVD Consolidation Analysis - Settlement vs Time Calculator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run analysis from YAML file
  python pvd_consolidation.py --data input.yaml

  # Specify output directory
  python pvd_consolidation.py --data input.yaml --output results/

  # Run example analysis
  python pvd_consolidation.py --example
        """,
    )

    parser.add_argument("--data", type=str, help="Path to YAML input file")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for results (default: same as input file)",
    )
    parser.add_argument(
        "--example",
        action="store_true",
        help="Run example analysis with default parameters",
    )

    args = parser.parse_args()

    if args.example:
        print("Running example analysis...")
        example_usage()
    elif args.data:
        if not os.path.exists(args.data):
            print(f"Error: Input file '{args.data}' not found!")
            return
        run_analysis_from_yaml(args.data, args.output)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
