# Vacuum Preloading - User Guide

## Overview

The PVD consolidation tool now supports **vacuum preloading** with:
- ✅ Vacuum-only loading
- ✅ Combined surcharge + vacuum
- ✅ Staged loading (multiple time stages)
- ✅ Vacuum loss with radial distance from drain
- ✅ Layer-specific vacuum application (only in PVD zone)

## Quick Start Examples

### Example 1: Vacuum Only

**Python API:**
```python
from pvd_consolidation import PVDConsolidation, SoilLayer, PVDProperties

layers = [SoilLayer(...)]
pvd = PVDProperties(dw=0.05, ds=0.15, De=1.5, L_drain=12.0, qw=1e12)

# Vacuum only (no surcharge)
analysis = PVDConsolidation(
    layers, 
    pvd,
    surcharge=0.0,
    vacuum=80.0  # 80 kPa vacuum
)

time, settlement = analysis.settlement_vs_time(t_max=2.0)
```

**YAML File:**
```yaml
loading_stages:
  - start_time: 0.0
    surcharge: 0.0
    vacuum: 80.0

# Run: python pvd_consolidation.py --data example_vacuum_simple.yaml
```

### Example 2: Combined Surcharge + Vacuum

**Python API:**
```python
analysis = PVDConsolidation(
    layers, 
    pvd,
    surcharge=50.0,  # 50 kPa fill
    vacuum=70.0      # 70 kPa vacuum
)
# Total equivalent load = 120 kPa
```

**YAML File:**
```yaml
loading_stages:
  - start_time: 0.0
    surcharge: 50.0
    vacuum: 70.0
```

### Example 3: Staged Loading

**YAML File:**
```yaml
loading_stages:
  # Stage 1: Start with vacuum only
  - start_time: 0.0
    surcharge: 0.0
    vacuum: 80.0
    
  # Stage 2: Add surcharge at 3 months
  - start_time: 0.25
    surcharge: 40.0
    vacuum: 80.0
    
  # Stage 3: Increase surcharge at 6 months
  - start_time: 0.5
    surcharge: 70.0
    vacuum: 80.0
    
  # Stage 4: Remove vacuum, full surcharge at 1.5 years
  - start_time: 1.5
    surcharge: 100.0
    vacuum: 0.0
```

**Python API:**
```python
from pvd_consolidation import LoadingStage

stages = [
    LoadingStage(start_time=0.0, surcharge=0.0, vacuum=80.0),
    LoadingStage(start_time=0.25, surcharge=40.0, vacuum=80.0),
    LoadingStage(start_time=0.5, surcharge=70.0, vacuum=80.0),
    LoadingStage(start_time=1.5, surcharge=100.0, vacuum=0.0),
]

analysis = PVDConsolidation(
    layers, 
    pvd,
    loading_stages=stages
)
```

## Vacuum Distribution with Distance

### Default Behavior

Vacuum decreases exponentially with distance from drain:

```
u(r) = u_drain × exp(-r/r_influence)
```

**At different distances:**
- r = 0 (at drain): 100% vacuum (80 kPa)
- r = r_influence/2: 60.6% vacuum (48.5 kPa)
- r = r_influence: 36.8% vacuum (29.4 kPa)
- r = 2×r_influence: 13.5% vacuum (10.8 kPa)
- r = De/2: ~5-10% vacuum

### Setting Influence Radius

**Default:** `r_influence = De/4`

**Custom value:**
```yaml
pvd:
  De: 1.5
  r_influence: 0.4  # Custom influence radius (m)
```

**Guidelines:**
- Dense drain spacing: `r_influence = De/3`
- Standard spacing: `r_influence = De/4` (default)
- Wide spacing: `r_influence = De/5`

## Physical Behavior

### Surcharge vs Vacuum Comparison

| Aspect | Surcharge | Vacuum |
|--------|-----------|--------|
| Application | Top-down (gravity) | Inside drains (suction) |
| Pore pressure | Positive (+) | Negative (-) |
| Effective stress | σ' = σ + Δσ | σ' = σ - (-vacuum) = σ + vacuum |
| Settlement | Same | **Same magnitude!** |
| Stability | May cause failure | No additional weight |
| Coverage | All layers | **Only PVD zone** |

**Key insight:** 80 kPa vacuum ≈ 80 kPa surcharge (same settlement!)

### Layer Application

```
Soil Profile (25m):

0-15m:  WITH PVD
        - Gets vacuum effect ✓
        - Gets surcharge ✓
        - Fast consolidation (Uh + Uv)

15-25m: BELOW PVD  
        - NO vacuum effect ✗
        - Gets surcharge only ✓
        - Slower consolidation (Uv only)
```

## Design Strategies

### Strategy 1: Vacuum First, Then Surcharge

**Benefits:**
- Improve soil strength before loading
- Reduce settlement under final load
- Safe for very weak soils

**Timeline:**
```
t=0:     Install PVD + vacuum (0 + 80 kPa)
t=0.5y:  Add partial fill (40 + 80 kPa)
t=1.0y:  Full fill, remove vacuum (100 + 0 kPa)
```

### Strategy 2: Combined from Start

**Benefits:**
- Maximum equivalent load
- Fastest consolidation
- Cost-effective if soil can handle

**Timeline:**
```
t=0:     PVD + vacuum + fill (60 + 80 kPa = 140 kPa equivalent)
t=1.5y:  Remove vacuum (100 + 0 kPa)
```

### Strategy 3: Staged Build-Up

**Benefits:**
- Safest approach
- Monitor and adjust
- Good for uncertain conditions

**Timeline:**
```
t=0:     Small fill (20 kPa)
t=0.2y:  Add vacuum (20 + 60 kPa)
t=0.5y:  Increase both (50 + 80 kPa)
t=1.0y:  Full load (80 + 80 kPa)
t=2.0y:  Remove vacuum (100 + 0 kPa)
```

## Practical Considerations

### Vacuum Magnitude

**Typical range:** 60-90 kPa

**Factors:**
- Atmospheric pressure limit: ~101 kPa
- Practical maximum: 90 kPa
- Conservative design: 70-80 kPa
- Membrane efficiency: 70-85%

**Recommendation:** Use 70-80 kPa for design

### Drain Spacing Effect

**Dense spacing (De = 1.0-1.2 m):**
- Better vacuum distribution
- Higher cost
- Faster consolidation

**Standard spacing (De = 1.5 m):**
- Good balance
- Most economical
- Adequate vacuum coverage

**Wide spacing (De = 2.0+ m):**
- Reduced vacuum efficiency
- Lower cost
- Longer consolidation time

### Membrane Requirements

For vacuum preloading:
- Ground surface must be sealed
- Typical: HDPE geomembrane
- Critical: Airtight seal at edges
- Maintenance: Repair leaks quickly

## Expected Results

### Vacuum-Only System

**Input:**
- 12m soft clay
- PVD to 12m
- 80 kPa vacuum

**Expected output:**
- 50% consolidation: ~0.15 years (2 months)
- 90% consolidation: ~0.5 years (6 months)
- Same as 80 kPa surcharge but safer!

### Combined System

**Input:**
- 15m soft clay  
- PVD to 15m
- 50 kPa fill + 70 kPa vacuum = 120 kPa equivalent

**Expected output:**
- 50% consolidation: ~0.1 years (1.2 months)
- 90% consolidation: ~0.3 years (3.6 months)
- Faster than either alone!

### Staged Loading

**Input:**
```
t=0:   0 + 80 kPa
t=0.25: 40 + 80 kPa  
t=0.5: 70 + 80 kPa
t=1.5: 100 + 0 kPa
```

**Expected output:**
- Progressive settlement increase at each stage
- Step-wise consolidation curve
- Final settlement ~same as 100 kPa surcharge
- But achieved more safely!

## Troubleshooting

### Issue: Settlement same as surcharge-only

**Cause:** Vacuum not being applied in PVD zone

**Check:**
1. Is vacuum > 0 in loading_stages?
2. Are layers within L_drain?
3. Is r_influence reasonable?

### Issue: Very slow consolidation with vacuum

**Cause:** Vacuum influence radius too small

**Solution:**
- Increase r_influence (try De/3 instead of De/4)
- Check drain spacing (may be too wide)

### Issue: Unrealistic fast consolidation

**Cause:** Vacuum applied beyond PVD zone

**Check:**
- Verify L_drain setting
- Check layer depths vs drain length

## File Format Reference

### YAML Structure

```yaml
soil_layers:
  - thickness: float
    Cv: float
    Ch: float
    # ... other properties
    kh: float  # Required for vacuum
    ks: float  # Required for vacuum

pvd:
  dw: float
  ds: float
  De: float
  L_drain: float
  qw: float
  r_influence: float  # Optional, default = De/4

loading_stages:  # Can have 1 or many stages
  - start_time: float  # years
    surcharge: float   # kPa
    vacuum: float      # kPa (positive value)
```

### Python API

```python
# Simple loading
analysis = PVDConsolidation(
    soil_layers,
    pvd,
    surcharge=50.0,
    vacuum=70.0
)

# Staged loading
stages = [
    LoadingStage(start_time=0.0, surcharge=0, vacuum=80),
    LoadingStage(start_time=1.0, surcharge=100, vacuum=0),
]
analysis = PVDConsolidation(
    soil_layers,
    pvd,
    loading_stages=stages
)
```

## Summary

✅ **Vacuum preloading implemented with:**
- Radial distance decay: u(r) = u_drain × exp(-r/r_influence)
- Staged loading support
- Combined surcharge + vacuum
- Layer-specific application (PVD zone only)

✅ **Use vacuum when:**
- Soil is very weak (surcharge would cause failure)
- Fast consolidation needed
- Limited headroom for embankment
- PVD system already planned

✅ **Design guidelines:**
- Vacuum magnitude: 70-80 kPa typical
- Influence radius: De/4 default
- Stage duration: 0.2-0.5 years per stage
- Monitor and adjust based on field data

For examples, see:
- `example_vacuum_simple.yaml` - Basic vacuum usage
- `example_vacuum.yaml` - Staged loading with vacuum

Happy consolidating! 🎯
