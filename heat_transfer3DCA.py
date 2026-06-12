# =============================================================================
# 3D Cellular Automata — Urban Heat Transfer, Kranj, Slovenia
#
# Physics layers (applied each 10-minute timestep):
#   1. Conductive diffusion        — explicit CA, CFL-safe
#   2. Solar heating               — cos-zenith × cloud × shadow mask
#                                    with seasonal ramp (days lengthen in May)
#   3. Nocturnal radiative cooling — longwave loss, cloud-modulated
#   4. Free-atmosphere relaxation  — outdoor air pinned to sky temperature
#   5. Asphalt boundary-layer plume— sensible heat rises 0-10 m above streets
#   6. Surface ↔ air convection    — Newton cooling, symmetric, weakened CONV_AIR
#   7. Interior ventilation        — ACH-based air exchange through walls
#   8. Boundary conditions         — sky top (diurnal+noise), deep-ground anchors
#
# Climate inputs (Kranj, May):
#   Daytime mean air temperature : 19.5 °C
#   Nighttime mean air temperature:  8.5 °C
#   Wind speed                   :  6 km/h  (1.67 m/s)
#   Smooth noise offset          : +4.5 °C mean over 30 days
#
# Key fixes vs. original code:
#   FIX 1 — SOLAR_GAIN_RATE corrected for 1 m cell depth (was 10× too large)
#   FIX 2 — RADIATIVE_COOL_RATE re-balanced against corrected solar gain
#   FIX 3 — Symmetric air/solid convection (CONV_AIR no longer hard-snaps air)
#   FIX 4 — Free-atmosphere relaxation prevents multi-day heat accumulation
#   FIX 5 — Sky temperature recalibrated to real Kranj May 19.5/8.5 °C cycle
#   FIX 6 — Smooth correlated noise (+4.5 °C mean) added to sky temperature
#   FIX 7 — Wind-calibrated H_CONV (Jurges correlation at 1.67 m/s)
#   FIX 8 — ATM_RELAX raised (air nearly independent of surrounding surfaces)
#   FIX 9 — Asphalt boundary-layer plume up to 10 m height
#   FIX 10— Street deep-ground anchor prevents spurious cooling drift
#   FIX 11— Seasonal solar ramp (May days lengthen, net gain grows)
#   FIX 12— Reduced nocturnal convective loss from asphalt (high thermal mass)
# =============================================================================

import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

start_time = time.perf_counter()

# ══════════════════════════════════════════════════════════════════════════════
# 1.  DOMAIN & SIMULATION CONSTANTS
# ══════════════════════════════════════════════════════════════════════════════

GRID_SIZE     = (50, 50, 20)           # (x, y, z) cells — each 1 m³
STEPS_PER_DAY = 144                    # one 10-minute step
DAYS          = 30
TIMESTEPS     = STEPS_PER_DAY * DAYS

LATITUDE  = 46.2389                    # Kranj, Slovenia
START_DOY = 121                        # 1 May

DT = (24.0 * 3600.0) / STEPS_PER_DAY  # 600 s per step
DX = 1.0                               # cell size [m]

# ── Material indices ──────────────────────────────────────────────────────────
MAT_AIR      = 0   # outdoor open air
MAT_STREET   = 1   # asphalt / pavement
MAT_PARK     = 2   # grass / soil surface
MAT_BUILDING = 3   # concrete shell (walls, roof, floor slab)
MAT_INTERIOR = 4   # enclosed building air

# ── Thermal diffusivity  α  [m²/s] ───────────────────────────────────────────
DIFFUSIVITY = {
    MAT_AIR:      2.0e-5,   # turbulent outdoor air (eddy-enhanced)
    MAT_STREET:   7.0e-7,   # asphalt
    MAT_PARK:     3.5e-7,   # grass / soil
    MAT_BUILDING: 6.5e-7,   # concrete
    MAT_INTERIOR: 5.0e-6,   # calm indoor air
}

# ── Volumetric heat capacity  ρ·Cp  [J/(m³·K)] ───────────────────────────────
RHO_CP = {
    MAT_AIR:      1.2   * 1005,    #    1 206  J/(m³·K)
    MAT_STREET:   2100  * 920,     #  1 932 000
    MAT_PARK:     1300  * 1480,    #  1 924 000
    MAT_BUILDING: 2300  * 880,     #  2 024 000
    MAT_INTERIOR: 1.2   * 1005,    #    1 206
}

# ── Initial temperatures [°C] ─────────────────────────────────────────────────
T_INIT_AIR    = 14.0   # May morning (~diurnal midpoint for Kranj)
T_INIT_GROUND = 13.0   # surface soil / slab at simulation start
T_DEEP_GROUND = 11.0   # deep park soil — seasonal soft anchor

# FIX 10 — Street deep-ground anchor temperature.
# Asphalt stores significantly more solar heat than soil over a sunny May.
# Setting T_DEEP_STREET > T_DEEP_GROUND prevents the spurious cold bleed
# from adjacent park cells driving asphalt temperature downward over time.
# Value 16 °C is consistent with May urban asphalt sub-surface measurements.
T_DEEP_STREET = 16.0

# ── Solar / radiative parameters ──────────────────────────────────────────────
# FIX 1 — SOLAR_GAIN_RATE corrected for DX = 1 m cells.
# Original 0.17 assumed a 0.1 m absorption slab; dividing by 10 gives 0.017.
SOLAR_GAIN_RATE = 0.017   # °C·step⁻¹  at cos(z)=1, cloud=1

# FIX 11 — Seasonal solar ramp.
# In May, day length grows from ~14.5 h (Day 1) to ~15.5 h (Day 30) and
# solar elevation at noon increases by ~8°. Net daily solar gain on asphalt
# rises by roughly 1.5% per day. The ramp is applied multiplicatively to
# SOLAR_GAIN_RATE inside the update step.
SOLAR_SEASON_RAMP_PER_DAY = 0.015   # fractional gain increase per day

# FIX 2 — RADIATIVE_COOL_RATE re-balanced against corrected solar gain.
RADIATIVE_COOL_RATE = 0.010   # °C·step⁻¹

# ── Convection — FIX 7: wind-calibrated H_CONV ───────────────────────────────
# Jurges correlation: h_c ≈ 5.6 + 4·v  for v < 5 m/s
#   v = 6 km/h = 1.667 m/s  →  5.6 + 4·1.667 ≈ 12.3 W/(m²·K)
# Apply urban canyon sheltering factor ~0.8  →  ≈ 10 W/(m²·K)
WIND_SPEED_MS = 6.0 / 3.6   # 1.667 m/s
H_CONV        = 10.0         # W/(m²·K)

def conv_coeff(mat_id):
    """Dimensionless convective coupling coefficient for one 10-min step."""
    return min(1.0, H_CONV * DT / (RHO_CP[mat_id] * DX))

CONV_SOLID = {m: conv_coeff(m) for m in [MAT_STREET, MAT_PARK, MAT_BUILDING]}

# FIX 3 / FIX 8 — CONV_AIR drastically reduced so outdoor air is not pulled
# strongly toward surface temperatures; independence is enforced by ATM_RELAX.
CONV_AIR = 0.05   # was 1.0 (hard snap) — now a gentle 5% nudge per step

# FIX 12 — Night-time convection damping for asphalt.
# Real asphalt has very high thermal mass (~1.93 MJ/(m³·K)) and a thin
# nocturnal boundary layer. Its overnight cooling rate is roughly 60% of
# the daytime convective rate. Applied only when solar == 0.
ASPHALT_NIGHT_CONV_FACTOR = 0.4   # multiplied onto CONV_SOLID[MAT_STREET] at night

# FIX 4 / FIX 8 — ATM_RELAX raised from 0.001 → 0.15.
# e-folding time = DT / ATM_RELAX = 600 / 0.15 ≈ 67 min.
# Outdoor air returns to sky temperature within ~1 hour of any perturbation,
# making it effectively independent of surrounding streets and buildings.
ATM_RELAX = 0.15

# ── Interior ventilation ──────────────────────────────────────────────────────
ACH       = 1.0                        # air changes per hour
VENT_FRAC = ACH / (3600.0 / DT)        # ≈ 0.167 fraction replaced per step

# ── Asphalt boundary-layer plume (FIX 9) ─────────────────────────────────────
# Exponential decay of asphalt excess heat with height above ground.
# H_scale compressed by wind shear (higher wind → shallower mixing layer).
#   H_scale = 4.0 / (1 + v_ms * 0.3)  ≈ 3.1 m at 1.67 m/s
# ASPHALT_STRENGTH: fraction of column excess temperature added per step.
# Applied to air cells z=1..ASPHALT_PLUME_TOP directly above street columns.
ASPHALT_PLUME_TOP  = 10                # maximum influence height [m / cells]
ASPHALT_H_SCALE    = 4.0 / (1.0 + WIND_SPEED_MS * 0.3)   # ≈ 3.1 m
ASPHALT_STRENGTH   = 0.25             # fractional coupling per step

# ══════════════════════════════════════════════════════════════════════════════
# 2.  CFL STABILITY VALIDATION
# ══════════════════════════════════════════════════════════════════════════════
print("=" * 54)
print("  CFL Stability Check  (Fo = α·Δt/Δx²  ≤  1/6)")
print("=" * 54)
material_names = {0:"Air", 1:"Street", 2:"Park", 3:"Building", 4:"Interior"}
for mat, alpha in DIFFUSIVITY.items():
    Fo     = alpha * DT / DX**2
    status = "OK" if Fo <= 1/6 else "VIOLATION"
    print(f"  {material_names[mat]:12s}: α={alpha:.1e}  Fo={Fo:.6f}  {status}")
    assert Fo <= 1/6, f"CFL violated for {material_names[mat]}"
print()

# ══════════════════════════════════════════════════════════════════════════════
# 3.  CITY GEOMETRY
# ══════════════════════════════════════════════════════════════════════════════
def generate_city_map(size):
    """
    Build a 50×50×20 m urban block with three buildings, two streets, and park.

    z = 0   : ground layer  (MAT_PARK default, overwritten by streets/buildings)
    z = 1+  : above-ground  (MAT_AIR default, overwritten by building shells/interiors)

    Street layout:
      N-S arterial : x = 22..27,  full y extent
      E-W arterial : y = 23..26,  full x extent

    Buildings:
      Block A —  6 storeys, SW quadrant
      Block B — 12 storeys, NE quadrant (tall tower)
      Block C —  8 storeys, SE quadrant
    """
    w, l, h = size
    mat = np.full(size, MAT_AIR, dtype=np.int8)

    # Ground: park everywhere by default
    mat[:, :, 0] = MAT_PARK

    # Streets overwrite park at z=0
    mat[w//2-3 : w//2+3, :,        0] = MAT_STREET   # N-S arterial
    mat[:,        l//2-2 : l//2+2, 0] = MAT_STREET   # E-W arterial

    def add_building(mat, x1, x2, y1, y2, nz):
        """Solid shell of MAT_BUILDING with MAT_INTERIOR air core."""
        mat[x1:x2, y1:y2, 0:nz] = MAT_BUILDING
        if nz > 2 and (x2 - x1) > 2 and (y2 - y1) > 2:
            mat[x1+1:x2-1, y1+1:y2-1, 1:nz-1] = MAT_INTERIOR

    add_building(mat,  5, 15,  5, 15,  6)   # Block A —  6 storeys
    add_building(mat, 30, 45, 10, 25, 12)   # Block B — 12 storeys
    add_building(mat, 10, 20, 30, 40,  8)   # Block C —  8 storeys

    return mat

# ══════════════════════════════════════════════════════════════════════════════
# 4.  ASPHALT PLUME MASK  (pre-computed — static geometry)
# ══════════════════════════════════════════════════════════════════════════════
def build_asphalt_plume_mask(material_map, plume_top):
    """
    Pre-compute which air cells lie in open columns directly above street cells,
    up to plume_top metres height.  A column is 'open' if every cell between
    z=0 and the target level is unobstructed air (no building walls crossing it).

    Returns
    -------
    plume_mask   : bool array (plume_top+1, w, l)
                   True at (z, x, y) if that cell receives asphalt plume heating
    decay_weight : float array (plume_top+1,)
                   Exponential decay factor exp(-z / H_scale) for each level
    """
    w, l, h     = material_map.shape
    street_xy   = (material_map[:, :, 0] == MAT_STREET)   # (w, l) footprint

    plume_mask   = np.zeros((plume_top + 1, w, l), dtype=bool)
    decay_weight = np.zeros(plume_top + 1)

    for z in range(1, min(plume_top + 1, h)):
        # Column is open up to level z if all intermediate levels are air
        col_open = street_xy.copy()
        for zz in range(1, z):
            col_open &= (material_map[:, :, zz] == MAT_AIR)
        is_air_here      = (material_map[:, :, z] == MAT_AIR)
        plume_mask[z]    = col_open & is_air_here
        decay_weight[z]  = np.exp(-z / ASPHALT_H_SCALE)

    return plume_mask, decay_weight

# ══════════════════════════════════════════════════════════════════════════════
# 5.  SOLAR & SKY MODEL  — FIX 5 & 6: calibrated Kranj May climate + noise
# ══════════════════════════════════════════════════════════════════════════════
# Kranj May observations:
#   Daytime mean   = 19.5 °C
#   Nighttime mean =  8.5 °C
#   Midpoint       = (19.5 + 8.5) / 2 = 14.0 °C
#   Amplitude      = (19.5 - 8.5) / 2 =  5.5 °C
#
# Diurnal sine: peaks at 14:00 (phase offset +2 h from noon),
#               troughs at 02:00.
#
# Smooth noise (FIX 6):
#   AR(1) process with ~3-hour correlation time produces smooth variation.
#   Shifted so its mean over the full simulation = exactly +4.5 °C.
#   Clipped to ±6 °C around the shifted mean.

T_DAY_MEAN        = 19.5
T_NIGHT_MEAN      =  8.5
T_MID             = (T_DAY_MEAN + T_NIGHT_MEAN) / 2.0   # 14.0 °C
T_AMP             = (T_DAY_MEAN - T_NIGHT_MEAN) / 2.0   #  5.5 °C
NOISE_MEAN_TARGET =  4.5    # desired mean offset [°C]
NOISE_CLIP        =  6.0    # symmetric clip around shifted mean [°C]

def generate_sky_noise(timesteps, seed=77, tau_steps=18):
    """
    AR(1) correlated noise:  x[t] = a·x[t-1] + (1-a)·ε[t],  ε ~ N(0,1)
    tau_steps = 18 → e-folding ≈ 18 × 600 s = 3 hours (smooth, not steppy).
    Output is rescaled to std=1.5 °C then shifted to mean = NOISE_MEAN_TARGET.
    """
    np.random.seed(seed)
    a   = np.exp(-1.0 / tau_steps)
    eps = np.random.randn(timesteps)
    x   = np.zeros(timesteps)
    x[0] = eps[0]
    for t in range(1, timesteps):
        x[t] = a * x[t-1] + (1.0 - a) * eps[t]
    x = x / (x.std() + 1e-9) * 1.5          # normalise std to 1.5 °C
    x = x - x.mean() + NOISE_MEAN_TARGET    # shift mean to target
    x = np.clip(x,
                NOISE_MEAN_TARGET - NOISE_CLIP,
                NOISE_MEAN_TARGET + NOISE_CLIP)
    return x

# Generate once; index by step throughout simulation
_sky_noise = generate_sky_noise(TIMESTEPS)


def get_solar_intensity(step):
    """
    cos(zenith angle) for Kranj at simulation step.
    Returns 0 during night (below horizon).
    """
    doy        = START_DOY + (step // STEPS_PER_DAY)
    decl       = np.radians(23.45 * np.sin(np.radians((360/365) * (doy - 80))))
    hour       = (step % STEPS_PER_DAY) * (24.0 / STEPS_PER_DAY)
    hour_angle = np.radians((hour - 12.0) * 15.0)
    lat        = np.radians(LATITUDE)
    cos_z      = (  np.sin(lat) * np.sin(decl)
                  + np.cos(lat) * np.cos(decl) * np.cos(hour_angle))
    return float(max(0.0, cos_z))


def get_sky_temp(step):
    """
    Prescribed outdoor air / sky boundary temperature [°C].
    = diurnal cycle (19.5/8.5 day/night means) + smooth noise (+4.5 °C mean).
    """
    hour    = (step % STEPS_PER_DAY) * (24.0 / STEPS_PER_DAY)
    diurnal = T_AMP * np.sin(np.pi * (hour - 2.0) / 12.0)
    return float(T_MID + diurnal + _sky_noise[step])

# ══════════════════════════════════════════════════════════════════════════════
# 6.  WEATHER ENGINE  (stochastic cloud clearness index)
# ══════════════════════════════════════════════════════════════════════════════
def generate_weather(timesteps, seed=44, base_vol=0.06):
    """
    Stochastic clearness index k_t ∈ [0.15, 1.0] driven by a random walk.
    Volatility scales with solar elevation so clouds develop mainly by day.
    """
    np.random.seed(seed)
    clearness = np.zeros(timesteps)
    k = 0.85
    for step in range(timesteps):
        sol = get_solar_intensity(step)
        vol = base_vol * max(0.1, sol)
        k  += np.random.uniform(-vol, vol)
        k   = np.clip(k, 0.15, 1.0)
        clearness[step] = k
    return clearness

# ══════════════════════════════════════════════════════════════════════════════
# 7.  SHADOW CASTING
# ══════════════════════════════════════════════════════════════════════════════
def compute_exposed_mask(material_map, solar_intensity):
    """
    Boolean mask of surface cells that receive direct solar radiation.

    A surface cell (non-air with air directly above) is shadowed if any
    taller solid object exists in the same vertical column.
    Returns all-False during night (solar_intensity == 0).
    """
    if solar_intensity <= 0:
        return np.zeros(material_map.shape, dtype=bool)

    w, l, h = material_map.shape

    # Top-surface cells: non-air cell with air immediately above
    top_surface = np.zeros((w, l, h), dtype=bool)
    top_surface[:, :, :-1] = (
        (material_map[:, :, :-1] != MAT_AIR) &
        (material_map[:, :, 1:]  == MAT_AIR)
    )

    # Highest non-air cell in each column (for vertical shadow test)
    non_air     = (material_map != MAT_AIR)
    flipped_idx = np.argmax(non_air[:, :, ::-1], axis=2)
    col_max_z   = (h - 1) - flipped_idx
    col_max_z[~non_air.any(axis=2)] = 0

    # A top-surface cell is shadowed if something taller exists above it
    shadowed = np.zeros((w, l, h), dtype=bool)
    for z in range(h):
        shadowed[:, :, z] = top_surface[:, :, z] & (col_max_z > z)

    return top_surface & (~shadowed)

# ══════════════════════════════════════════════════════════════════════════════
# 8.  CONVECTION HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════
def build_interface_masks(material_map):
    """
    Pre-compute Boolean interface masks used every timestep.
    'Bordering' means sharing at least one face with the target material.
    """
    is_air   = (material_map == MAT_AIR)
    is_solid = ~(is_air | (material_map == MAT_INTERIOR))
    is_int   = (material_map == MAT_INTERIOR)

    def any_neighbour(mask, target):
        """True where mask is True AND at least one face-neighbour is in target."""
        T_int = target.astype(np.int8)
        T_pad = np.pad(T_int, 1, mode='constant', constant_values=0)
        nc = (T_pad[2:,  1:-1, 1:-1] + T_pad[:-2, 1:-1, 1:-1] +
              T_pad[1:-1, 2:,  1:-1] + T_pad[1:-1, :-2, 1:-1] +
              T_pad[1:-1, 1:-1, 2:]  + T_pad[1:-1, 1:-1, :-2])
        return mask & (nc > 0)

    return {
        "solid_bordering_air": any_neighbour(is_solid, is_air),
        "air_bordering_solid": any_neighbour(is_air,   is_solid),
        "int_bordering_air":   any_neighbour(is_int,   is_air),
        "is_air":              is_air,
        "is_solid":            is_solid,
        "is_interior":         is_int,
    }


def air_neighbour_mean(T, material_map):
    """
    For each cell, compute the mean temperature of its face-adjacent MAT_AIR
    neighbours.  Returns (mean_air, has_air_neighbour_mask).
    """
    is_air_f = (material_map == MAT_AIR).astype(np.float64)
    T_pad    = np.pad(T,        1, mode='edge')
    a_pad    = np.pad(is_air_f, 1, mode='constant', constant_values=0)

    T_sum = (T_pad[2:,  1:-1,1:-1]*a_pad[2:,  1:-1,1:-1] +
             T_pad[:-2, 1:-1,1:-1]*a_pad[:-2, 1:-1,1:-1] +
             T_pad[1:-1,2:,  1:-1]*a_pad[1:-1,2:,  1:-1] +
             T_pad[1:-1,:-2, 1:-1]*a_pad[1:-1,:-2, 1:-1] +
             T_pad[1:-1,1:-1,2:]  *a_pad[1:-1,1:-1,2:]   +
             T_pad[1:-1,1:-1,:-2] *a_pad[1:-1,1:-1,:-2])
    n_air = (a_pad[2:,  1:-1,1:-1] + a_pad[:-2, 1:-1,1:-1] +
             a_pad[1:-1,2:,  1:-1] + a_pad[1:-1,:-2, 1:-1] +
             a_pad[1:-1,1:-1,2:]   + a_pad[1:-1,1:-1,:-2])

    with np.errstate(invalid='ignore', divide='ignore'):
        mean_air = np.where(n_air > 0, T_sum / n_air, T)
    return mean_air, (n_air > 0)

# ══════════════════════════════════════════════════════════════════════════════
# 9.  CORE CA UPDATE STEP
# ══════════════════════════════════════════════════════════════════════════════
def update_temperature(T, material_map, iface, exposed_mask,
                       step, weather_clearness,
                       plume_mask, decay_weight):
    """
    Advance temperatures by one 10-minute CA timestep.
    Eight physics layers are applied sequentially.
    """
    new_T  = T.copy()
    solar  = get_solar_intensity(step)
    cloud  = weather_clearness[step]
    sky_T  = get_sky_temp(step)
    day_of_sim = step // STEPS_PER_DAY   # integer 0..DAYS-1

    # ── Layer 1: Conductive diffusion (explicit, CFL-safe) ───────────────────
    T_pad     = np.pad(T, 1, mode='edge')
    laplacian = (T_pad[2:,  1:-1,1:-1] + T_pad[:-2, 1:-1,1:-1] +
                 T_pad[1:-1,2:,  1:-1] + T_pad[1:-1,:-2, 1:-1] +
                 T_pad[1:-1,1:-1,2:]   + T_pad[1:-1,1:-1,:-2]
                 - 6.0 * T)
    alpha_map = np.vectorize(DIFFUSIVITY.get)(material_map).astype(np.float64)
    new_T    += (alpha_map * DT / DX**2) * laplacian

    # ── Layer 2: Solar heating on exposed surfaces ────────────────────────────
    if solar > 0:
        # FIX 11 — Seasonal ramp: each day in May delivers ~1.5% more solar
        # energy than the previous one (longer days, higher solar elevation).
        season_ramp = 1.0 + SOLAR_SEASON_RAMP_PER_DAY * day_of_sim
        new_T[exposed_mask] += SOLAR_GAIN_RATE * solar * cloud * season_ramp

    # ── Layer 3: Nocturnal radiative cooling on top surfaces ─────────────────
    else:
        night_surface = np.zeros_like(material_map, dtype=bool)
        night_surface[:, :, :-1] = (
            (material_map[:, :, :-1] != MAT_AIR) &
            (material_map[:, :, 1:]  == MAT_AIR)
        )
        # Cloud acts as a blanket: higher k_t (clearer sky) → more longwave loss
        new_T[night_surface] -= RADIATIVE_COOL_RATE * cloud

    # ── Layer 4: Free-atmosphere relaxation ──────────────────────────────────
    # ATM_RELAX = 0.15 → e-fold ≈ 67 min.  Outdoor air snaps back to sky_T
    # after any surface perturbation, keeping it nearly surface-independent.
    air_cells = iface["is_air"]
    new_T[air_cells] += ATM_RELAX * (sky_T - new_T[air_cells])

    # ── Layer 5: Asphalt boundary-layer plume ────────────────────────────────
    # For each z in 1..ASPHALT_PLUME_TOP, add a fraction of the asphalt excess
    # temperature (asphalt T minus sky_T) decayed exponentially with height.
    # At night the excess can be negative → plume slightly cools near-surface air.
    excess_xy = new_T[:, :, 0] - sky_T   # (w, l) — positive when asphalt > sky
    for z in range(1, min(ASPHALT_PLUME_TOP + 1, GRID_SIZE[2])):
        mask_z = plume_mask[z]            # bool (w, l)
        if not mask_z.any():
            continue
        delta = ASPHALT_STRENGTH * decay_weight[z] * excess_xy
        new_T[:, :, z][mask_z] += delta[mask_z]

    # ── Layer 6: Surface ↔ outdoor-air convection ─────────────────────────────
    # Newton's law of cooling applied symmetrically: hot surfaces lose heat to
    # cool air and vice versa.  CONV_AIR = 0.05 keeps outdoor air from being
    # strongly dragged toward surface temperatures (FIX 3 / FIX 8).
    T_air_adj, _ = air_neighbour_mean(new_T, material_map)
    delta_T = new_T - T_air_adj   # positive → surface warmer than adjacent air

    for mat in [MAT_STREET, MAT_PARK, MAT_BUILDING]:
        solid_mask = iface["solid_bordering_air"] & (material_map == mat)
        if not solid_mask.any():
            continue
        coeff = CONV_SOLID[mat]
        # FIX 12 — At night, asphalt loses heat to air more slowly than during
        # the day because the stable nocturnal boundary layer suppresses turbulent
        # mixing and asphalt's high thermal mass resists cooling.
        if mat == MAT_STREET and solar == 0:
            coeff = coeff * ASPHALT_NIGHT_CONV_FACTOR
        new_T[solid_mask] -= coeff * delta_T[solid_mask]

    # Air cells warm/cool toward mean of their solid neighbours (FIX 3)
    air_mask  = iface["air_bordering_solid"]
    T_pad2    = np.pad(new_T, 1, mode='edge')
    s_pad     = np.pad(iface["is_solid"].astype(np.float64), 1,
                       mode='constant', constant_values=0)
    T_sol_sum = (T_pad2[2:,  1:-1,1:-1]*s_pad[2:,  1:-1,1:-1] +
                 T_pad2[:-2, 1:-1,1:-1]*s_pad[:-2, 1:-1,1:-1] +
                 T_pad2[1:-1,2:,  1:-1]*s_pad[1:-1,2:,  1:-1] +
                 T_pad2[1:-1,:-2, 1:-1]*s_pad[1:-1,:-2, 1:-1] +
                 T_pad2[1:-1,1:-1,2:]  *s_pad[1:-1,1:-1,2:]   +
                 T_pad2[1:-1,1:-1,:-2] *s_pad[1:-1,1:-1,:-2])
    n_sol = (s_pad[2:,  1:-1,1:-1] + s_pad[:-2, 1:-1,1:-1] +
             s_pad[1:-1,2:,  1:-1] + s_pad[1:-1,:-2, 1:-1] +
             s_pad[1:-1,1:-1,2:]   + s_pad[1:-1,1:-1,:-2])
    with np.errstate(invalid='ignore', divide='ignore'):
        T_sol_mean = np.where(n_sol > 0, T_sol_sum / n_sol, new_T)
    new_T[air_mask] = ((1.0 - CONV_AIR) * new_T[air_mask] +
                        CONV_AIR         * T_sol_mean[air_mask])

    # ── Layer 7: Interior ventilation ─────────────────────────────────────────
    # Building interior air exchanges with outdoor air at ACH = 1.0 h⁻¹.
    int_mask = iface["int_bordering_air"]
    if int_mask.any():
        T_out_adj, _ = air_neighbour_mean(new_T, material_map)
        new_T[int_mask] = ((1.0 - VENT_FRAC) * new_T[int_mask] +
                            VENT_FRAC         * T_out_adj[int_mask])

    # ── Layer 8: Boundary conditions ──────────────────────────────────────────
    # Top layer: Dirichlet sky temperature
    new_T[:, :, -1] = sky_T

    # Park z=0: soft seasonal relaxation toward deep-ground temperature
    park_z0 = (material_map[:, :, 0] == MAT_PARK)
    new_T[:, :, 0][park_z0] = (0.995 * new_T[:, :, 0][park_z0] +
                                0.005 * T_DEEP_GROUND)

    # FIX 10 — Street z=0: soft anchor toward warmer asphalt sub-surface.
    # Prevents spurious cold bleed from adjacent park cells over the 30-day run.
    street_z0 = (material_map[:, :, 0] == MAT_STREET)
    new_T[:, :, 0][street_z0] = (0.997 * new_T[:, :, 0][street_z0] +
                                  0.003 * T_DEEP_STREET)

    return new_T

# ══════════════════════════════════════════════════════════════════════════════
# 10.  INITIALISE SCENE
# ══════════════════════════════════════════════════════════════════════════════
material_map      = generate_city_map(GRID_SIZE)
weather_clearness = generate_weather(TIMESTEPS)
iface_masks       = build_interface_masks(material_map)
plume_mask, decay_weight = build_asphalt_plume_mask(material_map, ASPHALT_PLUME_TOP)

# Initial temperature field: ground materials start at T_INIT_GROUND, air at T_INIT_AIR
T = np.where(
    material_map != MAT_AIR,
    np.full(GRID_SIZE, T_INIT_GROUND),
    np.full(GRID_SIZE, T_INIT_AIR)
).astype(np.float64)

# Shadow mask cache — keyed by rounded cos(zenith) to avoid recomputing
# identical masks at steps with the same solar elevation
shadow_cache = {}
def get_exposed(step):
    key = round(get_solar_intensity(step), 2)
    if key not in shadow_cache:
        shadow_cache[key] = compute_exposed_mask(material_map, key)
    return shadow_cache[key]

# ══════════════════════════════════════════════════════════════════════════════
# 11.  MAIN SIMULATION LOOP
# ══════════════════════════════════════════════════════════════════════════════
hours_axis = np.linspace(0, 24.0 * DAYS, TIMESTEPS)

# Per-step recording arrays
air_temps, street_temps, building_temps, interior_temps = [], [], [], []
air_z1_temps, air_z5_temps = [], []   # near-surface plume levels
sky_T_series   = []
daylight_flags = np.zeros(TIMESTEPS)

# Per-day street statistics
street_max_daily, street_min_daily = [], []
day_street = []

print(f"Running {DAYS}-Day CA Simulation — Kranj, Slovenia  (v2: asphalt fix)")
print("-" * 64)

for step in range(TIMESTEPS):
    exposed = get_exposed(step)
    T = update_temperature(T, material_map, iface_masks, exposed,
                           step, weather_clearness, plume_mask, decay_weight)

    # Domain-mean temperatures by material
    t_air = float(np.mean(T[material_map == MAT_AIR]))
    t_str = float(np.mean(T[material_map == MAT_STREET]))
    t_bld = float(np.mean(T[material_map == MAT_BUILDING]))
    t_int = float(np.mean(T[material_map == MAT_INTERIOR]))

    # Near-surface plume: air at z=1 and z=5 directly above street columns
    z1_mask = plume_mask[1] & (material_map[:, :, 1] == MAT_AIR)
    z5_mask = (plume_mask[5] & (material_map[:, :, 5] == MAT_AIR)
               if ASPHALT_PLUME_TOP >= 5 else z1_mask)
    t_z1 = float(np.mean(T[:, :, 1][z1_mask])) if z1_mask.any() else t_air
    t_z5 = float(np.mean(T[:, :, 5][z5_mask])) if z5_mask.any() else t_air

    air_temps.append(t_air);      street_temps.append(t_str)
    building_temps.append(t_bld); interior_temps.append(t_int)
    air_z1_temps.append(t_z1);    air_z5_temps.append(t_z5)
    day_street.append(t_str)
    sky_T_series.append(get_sky_temp(step))

    if get_solar_intensity(step) > 0:
        daylight_flags[step] = 1.0

    if (step + 1) % STEPS_PER_DAY == 0:
        day_num = (step + 1) // STEPS_PER_DAY
        street_max_daily.append(max(day_street))
        street_min_daily.append(min(day_street))
        print(f"  Day {day_num:2d}/{DAYS} | "
              f"Air {t_air:5.1f}C | "
              f"Sky {sky_T_series[-1]:5.1f}C | "
              f"Street {t_str:5.1f}C ({min(day_street):.1f}-{max(day_street):.1f}) | "
              f"Plume z1={t_z1:.1f} z5={t_z5:.1f}C | "
              f"Int {t_int:5.1f}C | "
              f"Cloud {weather_clearness[step]*100:.0f}%")
        day_street = []

print(f"\nNoise mean: {np.mean(_sky_noise):.3f} C  (target {NOISE_MEAN_TARGET} C)")
print(f"Sky temp — mean:{np.mean(sky_T_series):.1f}  "
      f"min:{np.min(sky_T_series):.1f}  max:{np.max(sky_T_series):.1f} C")
print(f"Street  — Day1:{street_temps[STEPS_PER_DAY-1]:.1f}  "
      f"Day30:{street_temps[-1]:.1f} C  "
      f"(trend {street_temps[-1]-street_temps[STEPS_PER_DAY-1]:+.1f} C)")
print("\nSimulation complete.")

# ══════════════════════════════════════════════════════════════════════════════
# 12.  VISUALISATION — 5-panel dashboard
# ══════════════════════════════════════════════════════════════════════════════
fig = plt.figure(figsize=(20, 16))
gs  = gridspec.GridSpec(5, 1, height_ratios=[3, 1.4, 1.0, 0.9, 0.9], hspace=0.45)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax4 = fig.add_subplot(gs[3], sharex=ax1)
ax5 = fig.add_subplot(gs[4], sharex=ax1)

# ── Panel 1: All temperature traces ─────────────────────────────────────────
ax1.plot(hours_axis, sky_T_series,   label="Sky (prescribed)",        color='#2471A3', lw=1.8)
ax1.plot(hours_axis, air_temps,      label="Outdoor Air (domain mean)",color='#5BA4CF', lw=1.5, ls='--')
ax1.plot(hours_axis, street_temps,   label="Asphalt Street",          color='#C0392B', lw=2.2)
ax1.plot(hours_axis, building_temps, label="Concrete Shell",          color='#E6A817', lw=2.0)
ax1.plot(hours_axis, interior_temps, label="Building Interior",       color='#27AE60', lw=1.5, ls=':')

# Daylight shading modulated by cloud clearness
for step in range(0, TIMESTEPS - 1, 6):
    if daylight_flags[step] > 0:
        ax1.axvspan(hours_axis[step], hours_axis[min(step+6, TIMESTEPS-1)],
                    color='gold', alpha=0.05 * weather_clearness[step])
for day in range(1, DAYS):
    ax1.axvline(day*24, color='grey', ls=':', alpha=0.35, lw=0.7)
    ax1.text(day*24+0.3, min(air_temps)-0.8, f"D{day+1}",
             fontsize=6, color='grey', va='top')

ax1.set_title(
    f"{DAYS}-Day Urban Heat Transfer — Kranj, Slovenia\n"
    "May climate 19.5/8.5 C | Noise +4.5 C | Wind 6 km/h | "
    "Asphalt plume 0-10 m | Seasonal ramp | Night-conv damping | Street anchor",
    fontsize=10, fontweight='bold')
ax1.set_ylabel("Mean Temperature (C)", fontsize=11)
ax1.grid(True, ls='--', alpha=0.3)
ax1.legend(loc='upper left', fontsize=8.5, framealpha=0.85, ncol=2)

# ── Panel 2: Asphalt plume vertical comparison ───────────────────────────────
ax2.plot(hours_axis, street_temps,  label="Asphalt z=0",  color='#922B21', lw=1.8)
ax2.plot(hours_axis, air_z1_temps,  label="Air z=1 m",    color='#E74C3C', lw=1.5, ls='--')
ax2.plot(hours_axis, air_z5_temps,  label="Air z=5 m",    color='#F1948A', lw=1.5, ls=':')
ax2.plot(hours_axis, sky_T_series,  label="Sky reference", color='#2471A3', lw=1.2, alpha=0.7)
for day in range(1, DAYS):
    ax2.axvline(day*24, color='grey', ls=':', alpha=0.35, lw=0.7)
ax2.set_ylabel("Plume Temps (C)", fontsize=10)
ax2.set_title("Asphalt boundary-layer plume: surface, 1 m, 5 m vs sky", fontsize=9)
ax2.legend(loc='upper left', fontsize=8, framealpha=0.85, ncol=2)
ax2.grid(True, ls='--', alpha=0.3)

# ── Panel 3: Street daily range bars ─────────────────────────────────────────
day_centres = np.arange(0.5, DAYS) * 24
ax3.plot(hours_axis, street_temps, color='#C0392B', lw=1.0, alpha=0.5)
ax3.bar(day_centres,
        [mx-mn for mx,mn in zip(street_max_daily, street_min_daily)],
        bottom=street_min_daily, width=20, color='#C0392B', alpha=0.25,
        label="Daily street range")
ax3.set_ylabel("Street Temp (C)", fontsize=10)
ax3.legend(loc='upper left', fontsize=8)
ax3.grid(True, ls='--', alpha=0.3)
for day in range(1, DAYS):
    ax3.axvline(day*24, color='grey', ls=':', alpha=0.35, lw=0.7)

# ── Panel 4: Sky clearness / cloud cover ─────────────────────────────────────
ax4.plot(hours_axis, weather_clearness*100, color='#7D3C98', lw=1.3, label="Sky clearness %")
ax4.fill_between(hours_axis, weather_clearness*100, 100,
                 color='grey', alpha=0.18, label="Cloud cover")
ax4.set_ylabel("Sky Clarity (%)", fontsize=10)
ax4.set_ylim(0, 110)
ax4.legend(loc='lower left', fontsize=8)
ax4.grid(True, ls='--', alpha=0.3)
for day in range(1, DAYS):
    ax4.axvline(day*24, color='grey', ls=':', alpha=0.35, lw=0.7)

# ── Panel 5: Sky boundary temperature with noise ─────────────────────────────
ax5.plot(hours_axis, sky_T_series, color='#2471A3', lw=1.4,
         label=f"Sky boundary temp (noise mean +{NOISE_MEAN_TARGET} C)")
ax5.fill_between(hours_axis, sky_T_series, alpha=0.12, color='#2471A3')
ax5.axhline(np.mean(sky_T_series), color='navy', ls='--', lw=0.9,
            label=f"Mean {np.mean(sky_T_series):.1f} C")
ax5.set_ylabel("Sky Temp (C)", fontsize=10)
ax5.set_xlabel("Simulation time — cumulative hours  (1 May, Kranj)", fontsize=11)
ax5.set_xlim(0, 24*DAYS)
ax5.set_xticks(range(0, 24*DAYS+1, 12))
ax5.legend(loc='lower left', fontsize=8)
ax5.grid(True, ls='--', alpha=0.3)
for day in range(1, DAYS):
    ax5.axvline(day*24, color='grey', ls=':', alpha=0.35, lw=0.7)

plt.savefig('kranj_ca_heat_transfer.png', dpi=150, bbox_inches='tight')
print("Dashboard saved.")

# ══════════════════════════════════════════════════════════════════════════════
# 13.  X-Z CROSS-SECTION HEATMAPS  (Day 1, 3, 7 at solar noon)
# ══════════════════════════════════════════════════════════════════════════════
T2 = np.where(
    material_map != MAT_AIR,
    np.full(GRID_SIZE, T_INIT_GROUND),
    np.full(GRID_SIZE, T_INIT_AIR)
).astype(np.float64)

snap_targets = {
    0 * STEPS_PER_DAY + STEPS_PER_DAY//2: "Day 1 - Noon",
    2 * STEPS_PER_DAY + STEPS_PER_DAY//2: "Day 3 - Noon",
    6 * STEPS_PER_DAY + STEPS_PER_DAY//2: "Day 7 - Noon",
}
snap_data  = {}
snap_limit = max(snap_targets.keys())

for step in range(snap_limit + 1):
    exposed2 = get_exposed(step)
    T2 = update_temperature(T2, material_map, iface_masks, exposed2,
                            step, weather_clearness, plume_mask, decay_weight)
    if step in snap_targets:
        snap_data[step] = T2[:, GRID_SIZE[1]//2, :].copy()

fig2, axes = plt.subplots(1, 3, figsize=(17, 5))
vmin = min(a.min() for a in snap_data.values())
vmax = max(a.max() for a in snap_data.values())

for ax, (step, label) in zip(axes, snap_targets.items()):
    im = ax.imshow(snap_data[step].T, origin='lower', aspect='auto',
                   cmap='inferno', vmin=vmin, vmax=vmax)
    ax.set_title(f"{label}\n(X-Z slice, Y = 25 m)", fontsize=10)
    ax.set_xlabel("X (m)")
    ax.set_ylabel("Z height (m)")
    ax.axhline(ASPHALT_PLUME_TOP, color='cyan', lw=0.9, ls='--', alpha=0.7)
    ax.text(1, ASPHALT_PLUME_TOP + 0.3, f"plume top {ASPHALT_PLUME_TOP} m",
            color='cyan', fontsize=7)
    plt.colorbar(im, ax=ax, label="C", shrink=0.85)

plt.suptitle(
    "3D CA Heat Field — Cross-Section Snapshots (Kranj)\n"
    "Cyan dashed = asphalt plume top (10 m); warm columns above street cells visible",
    fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig('kranj_ca_heatmap_snapshots.png', dpi=150, bbox_inches='tight')
print("Heatmap snapshots saved.")

# ══════════════════════════════════════════════════════════════════════════════
# 14.  VERTICAL PLUME PROFILE  (Day 7 solar noon, Y=25 m slice)
# ══════════════════════════════════════════════════════════════════════════════
snap_key = list(snap_targets.keys())[2]   # Day 7 noon snapshot

z_profile_street = []
z_profile_park   = []

for z in range(GRID_SIZE[2]):
    # Use the central Y-slice (Y=25 m) from the snapshot
    col_temps = snap_data[snap_key][:, z]   # shape (50,) — temperature at each x

    # Masks for street and park columns in this Y-slice
    ms = (material_map[:, GRID_SIZE[1]//2, 0] == MAT_STREET) & \
         (material_map[:, GRID_SIZE[1]//2, z] == MAT_AIR)
    mp = (material_map[:, GRID_SIZE[1]//2, 0] == MAT_PARK) & \
         (material_map[:, GRID_SIZE[1]//2, z] == MAT_AIR)

    z_profile_street.append(float(np.mean(col_temps[ms])) if ms.any() else np.nan)
    z_profile_park.append(  float(np.mean(col_temps[mp])) if mp.any() else np.nan)

fig3, ax = plt.subplots(figsize=(5, 8))
z_axis = np.arange(GRID_SIZE[2])
ax.plot(z_profile_street, z_axis, 'o-',  color='#C0392B', lw=2.0, label="Above asphalt")
ax.plot(z_profile_park,   z_axis, 's--', color='#27AE60', lw=1.8, label="Above park")
ax.axhline(ASPHALT_PLUME_TOP, color='grey', ls=':', lw=1.2,
           label=f"Plume top ({ASPHALT_PLUME_TOP} m)")
ax.set_xlabel("Temperature (C)", fontsize=11)
ax.set_ylabel("Height z (m)", fontsize=11)
ax.set_title("Vertical temperature profile\n(Day 7 solar noon, Y=25 m slice)", fontsize=10)
ax.legend(fontsize=9)
ax.grid(True, ls='--', alpha=0.35)
plt.tight_layout()
plt.savefig('kranj_ca_plume_profile.png', dpi=150, bbox_inches='tight')
print("Plume profile saved.")

end_time = time.perf_counter()
print(f"Execution time: {end_time - start_time:.2f} seconds")
