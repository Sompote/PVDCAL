# PVD Consolidation Analysis

Multilayer Prefabricated Vertical Drain (PVD) consolidation analysis tool that calculates settlement vs time using finite difference method.

## 🌐 Try It Online

**[Launch Interactive App on Hugging Face](https://huggingface.co/spaces/Sompote/PVDcal)** 🚀

![Streamlit App Screenshot](Screenshot%202568-10-21%20at%2017.51.11.png)

## Features

- **Multilayer soil profile support** - Handle multiple soil layers with different properties
- **PVD parameter calculations** - Automatically computes Fn, Fs, Fr factors
- **Finite difference solver** - Solves the governing consolidation equation
- **Combined consolidation** - Calculates both horizontal and vertical drainage: U = 1 - (1 - Uh)(1 - Uv)
- **Settlement calculation** - Computes time-dependent settlement using RR and CR indices
- **Visualization tools** - Generates settlement vs time and consolidation profile plots

## Theory

### Governing Equation
The radial consolidation equation for PVD:

```
∂u/∂t = Ch[∂²u/∂r² + (1/r)(∂u/∂r)]
```

Where:
- u = excess pore water pressure
- r = radial distance from drain centerline
- t = time
- Ch = horizontal coefficient of consolidation

### PVD Influence Factors

**Geometric Factor (Fn):**
```
Fn = (n²/(n²-1))ln(n) - 3/4 + 1/n²
```
where n = De/dw

**Smear Effect Factor (Fs):**
```
Fs = [kh/ks - 1]ln(ds/dw)
```

**Well Resistance Factor (Fr):**
```
Fr = πz(L-z)(kh/qw)
```
where z = L/2 for two-way drainage

### Combined Consolidation

```
U = 1 - (1 - Uh)(1 - Uv)
```

### Settlement Calculation

```
Sc = (RR·log(σ'p/σ'ini) + CR·log(σ'final/σ'p))H
```

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Command Line Interface (CLI)

The tool provides a CLI for running analyses from YAML input files:

```bash
# Run analysis from YAML file
python pvd_consolidation.py --data example_input.yaml

# Specify output directory
python pvd_consolidation.py --data example_input.yaml --output results/

# Run example analysis with default parameters
python pvd_consolidation.py --example

# Show help
python pvd_consolidation.py --help
```

### YAML Input File Format

Create a YAML file with your project parameters:

```yaml
# Soil layers (from top to bottom)
soil_layers:
  - thickness: 5.0         # m
    Cv: 0.5                # m²/year
    Ch: 1.5                # m²/year
    RR: 0.05               # Recompression ratio
    CR: 0.30               # Compression ratio
    sigma_ini: 50.0        # kPa
    sigma_p: 80.0          # kPa
    
  - thickness: 8.0
    Cv: 0.3
    Ch: 1.0
    RR: 0.04
    CR: 0.35
    sigma_ini: 90.0
    sigma_p: 90.0

# PVD installation properties
pvd:
  dw: 0.05               # Drain diameter (m)
  ds: 0.15               # Smear zone diameter (m)
  De: 1.5                # Unit cell diameter (m)
  L_drain: 20.0          # Drain length (m)
  kh: 2.0                # Horizontal permeability (m/year)
  ks: 0.5                # Smear zone permeability (m/year)
  qw: 50.0               # Well discharge capacity (m³/year)

# Analysis parameters
analysis:
  surcharge: 100.0       # Applied load (kPa)
  dt: 0.01               # Time step (years)
  t_max: 20.0            # Max time (years)
  n_points: 200          # Number of time points
  
  # Time points for report (years)
  t_check:
    - 0.5
    - 1.0
    - 2.0
    - 5.0
    - 10.0
  
  # Times for consolidation profiles (years)
  t_profiles:
    - 0.5
    - 2.0
    - 10.0
```

See `example_input.yaml` for a complete example with detailed comments.

### Python API - Basic Example

```python
from pvd_consolidation import PVDConsolidation, SoilLayer, PVDProperties

# Define soil layers
layers = [
    SoilLayer(
        thickness=5.0,      # m
        Cv=0.5,            # m²/year
        Ch=1.5,            # m²/year
        RR=0.05,           # Recompression ratio
        CR=0.30,           # Compression ratio
        sigma_ini=50.0,    # kPa
        sigma_p=80.0       # kPa
    ),
    SoilLayer(
        thickness=8.0,
        Cv=0.3,
        Ch=1.0,
        RR=0.04,
        CR=0.35,
        sigma_ini=90.0,
        sigma_p=90.0
    )
]

# Define PVD properties
pvd = PVDProperties(
    dw=0.05,           # 50 mm drain diameter
    ds=0.15,           # 150 mm smear zone
    De=1.5,            # 1.5 m unit cell
    L_drain=20.0,      # 20 m drain length
    kh=2.0,            # 2 m/year
    ks=0.5,            # 0.5 m/year
    qw=50.0            # 50 m³/year
)

# Create analysis
surcharge = 100.0  # kPa
analysis = PVDConsolidation(layers, pvd, surcharge)

# Calculate settlement at time t
settlement, layer_settlements = analysis.calculate_settlement(t=5.0)
print(f"Settlement at 5 years: {settlement*1000:.2f} mm")

# Plot settlement vs time
time, settlement = analysis.plot_settlement_vs_time(t_max=20.0)

# Plot consolidation profile
analysis.plot_degree_of_consolidation(t=5.0)

# Generate report
t_check = [0.5, 1.0, 2.0, 5.0, 10.0]
print(analysis.get_summary_report(t_check))
```

### Quick Start

```bash
# Run the example
python pvd_consolidation.py --example

# Or run with your own YAML file
python pvd_consolidation.py --data your_input.yaml
```

## Input Parameters

### SoilLayer
- `thickness` (m) - Layer thickness
- `Cv` (m²/year) - Vertical coefficient of consolidation
- `Ch` (m²/year) - Horizontal coefficient of consolidation
- `RR` - Recompression ratio
- `CR` - Compression ratio
- `sigma_ini` (kPa) - Initial effective stress
- `sigma_p` (kPa) - Preconsolidation pressure

### PVDProperties
- `dw` (m) - Equivalent diameter of drain
- `ds` (m) - Smear zone diameter
- `ds` (m) - Equivalent diameter of unit cell (De = 1.05·S for triangular, De = 1.13·S for square)
- `L_drain` (m) - Total drain length (for two-way: actual length; for one-way: 2×length)
- `kh` (m/year) - Horizontal permeability
- `ks` (m/year) - Smear zone permeability
- `qw` (m³/year) - Well discharge capacity

## Output

When using CLI with `--data`, the tool generates:

1. **pvd_analysis_report.txt** - Summary report with PVD factors and settlement at time points
2. **settlement_vs_time.png** - Settlement development curve
3. **settlement_data.csv** - Time vs settlement data in CSV format
4. **consolidation_profile_tX.Xy.png** - Consolidation profiles (Uh, Uv, U) at specified times
5. **Excess pore pressure profiles** included in consolidation profile plots

## Methods

### Main Methods

- `calculate_pvd_factors()` - Returns Fn, Fs, Fr
- `calculate_Uh(t)` - Horizontal degree of consolidation
- `calculate_Uv(t)` - Vertical degree of consolidation
- `calculate_total_U(t)` - Combined degree of consolidation
- `calculate_settlement(t)` - Settlement at time t
- `settlement_vs_time(t_max)` - Settlement time series
- `plot_settlement_vs_time(t_max)` - Generate settlement plot
- `plot_degree_of_consolidation(t)` - Generate consolidation profiles
- `get_summary_report(t_check)` - Generate text report

## Examples

### Example 1: Single Layer

```python
layers = [
    SoilLayer(
        thickness=10.0,
        Cv=1.0,
        Ch=3.0,
        RR=0.05,
        CR=0.30,
        sigma_ini=80.0,
        sigma_p=100.0
    )
]

pvd = PVDProperties(dw=0.05, ds=0.15, De=1.5, L_drain=10.0,
                    kh=3.0, ks=1.0, qw=100.0)

analysis = PVDConsolidation(layers, pvd, surcharge=120.0)
time, settlement = analysis.settlement_vs_time(t_max=10.0)
```

### Example 2: Calculate Settlement at Specific Times

```python
times = [0.5, 1.0, 2.0, 5.0, 10.0]
settlements = []

for t in times:
    s, _ = analysis.calculate_settlement(t)
    settlements.append(s * 1000)  # Convert to mm
    print(f"t = {t} years: {s*1000:.2f} mm")
```

### Example 3: Get PVD Factors

```python
Fn, Fs, Fr = analysis.calculate_pvd_factors()
print(f"Geometric factor (Fn): {Fn:.4f}")
print(f"Smear factor (Fs): {Fs:.4f}")
print(f"Well resistance (Fr): {Fr:.4f}")
print(f"Total resistance (F): {Fn+Fs+Fr:.4f}")
```

## References

1. Barron, R.A. (1948). "Consolidation of fine-grained soils by drain wells"
2. Hansbo, S. (1981). "Consolidation of fine-grained soils by prefabricated drains"
3. Indraratna, B. et al. (2005). "Performance of test embankment constructed to failure on soft marine clay"

## License

MIT License

## Author

Geotechnical Engineering Analysis Tool
