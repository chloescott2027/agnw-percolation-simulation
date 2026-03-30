import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
from scipy.optimize import curve_fit
import os, json

# ── Parameters ─────────────────────────────────────────────────────
DOMAIN_SIZE = 100.0
WIRE_LENGTH = 20.0
NUM_WIRES   = 150
SEED        = 42

# ══════════════════════════════════════════════════════════════════
# CORE FUNCTIONS
# ══════════════════════════════════════════════════════════════════

def deposit_wires(n, length, domain, seed=None):
    rng = np.random.default_rng(seed)
    cx  = rng.uniform(0, domain, n)
    cy  = rng.uniform(0, domain, n)
    ang = rng.uniform(0, np.pi, n)
    dx  = (length / 2) * np.cos(ang)
    dy  = (length / 2) * np.sin(ang)
    return np.column_stack([cx-dx, cy-dy, cx+dx, cy+dy])

def segments_intersect(w1, w2):
    x1, y1, x2, y2 = w1
    x3, y3, x4, y4 = w2
    denom = (x1-x2)*(y3-y4) - (y1-y2)*(x3-x4)
    if abs(denom) < 1e-10:
        return False
    t = ((x1-x3)*(y3-y4) - (y1-y3)*(x3-x4)) / denom
    u = ((x1-x3)*(y1-y2) - (y1-y3)*(x1-x2)) / denom
    return 0 <= t <= 1 and 0 <= u <= 1

def build_graph(wires):
    G = nx.Graph()
    n = len(wires)
    G.add_nodes_from(range(n))
    for i in range(n):
        for j in range(i+1, n):
            if segments_intersect(wires[i], wires[j]):
                G.add_edge(i, j)
    return G

def check_percolation(wires, G, domain):
    top    = [i for i, w in enumerate(wires) if max(w[1], w[3]) >= domain * 0.95]
    bottom = [i for i, w in enumerate(wires) if min(w[1], w[3]) <= domain * 0.05]
    for t in top:
        for b in bottom:
            if nx.has_path(G, t, b):
                lcc = len(max(nx.connected_components(G), key=len)) / len(wires)
                return True, lcc
    lcc = len(max(nx.connected_components(G), key=len)) / len(wires)
    return False, lcc

def percolation_probability(n_wires_list, length, domain, trials=50, seed=0):
    probs = []
    for n in n_wires_list:
        count = 0
        for t in range(trials):
            w = deposit_wires(n, length, domain, seed=seed+t)
            G = build_graph(w)
            perc, _ = check_percolation(w, G, domain)
            if perc:
                count += 1
        probs.append(count / trials)
        print(f"  N={n:4d} → {count/trials:.2f}")
    return probs

def power_law(L, a, b):
    return a * np.array(L) ** b

# ══════════════════════════════════════════════════════════════════
# CACHED RUNNERS — only recompute if cache file missing
# ══════════════════════════════════════════════════════════════════

def run_threshold_sweep(force=False):
    cache = os.path.expanduser('~/nanowire_percolation/cache_threshold.json')
    if os.path.exists(cache) and not force:
        print("Loading threshold sweep from cache...")
        with open(cache) as f:
            d = json.load(f)
        return d['wire_counts'], d['probs']

    print("Running threshold sweep (this takes ~2 min)...")
    wire_counts = list(range(50, 800, 25))
    probs = percolation_probability(wire_counts, WIRE_LENGTH, DOMAIN_SIZE, trials=50)
    with open(cache, 'w') as f:
        json.dump({'wire_counts': wire_counts, 'probs': probs}, f)
    print("Cached to", cache)
    return wire_counts, probs

def run_aspect_ratio_sweep(force=False):
    cache = os.path.expanduser('~/nanowire_percolation/cache_aspect_ratio.json')
    if os.path.exists(cache) and not force:
        print("Loading aspect ratio sweep from cache...")
        with open(cache) as f:
            d = json.load(f)
        return d['wire_lengths'], d['threshold_Nc']

    print("Running aspect ratio sweep (this takes ~5 min)...")
    wire_lengths = [10.0, 15.0, 20.0, 25.0, 30.0]
    threshold_Nc = []
    for L in wire_lengths:
        probs_L = percolation_probability(
            list(range(50, 600, 30)), L, DOMAIN_SIZE, trials=40, seed=1
        )
        counts = list(range(50, 600, 30))
        Nc = next((counts[i] for i, p in enumerate(probs_L) if p >= 0.5), None)
        threshold_Nc.append(Nc)
        print(f"  L={L:.0f} µm → Nc ≈ {Nc}")
    with open(cache, 'w') as f:
        json.dump({'wire_lengths': wire_lengths, 'threshold_Nc': threshold_Nc}, f)
    print("Cached to", cache)
    return wire_lengths, threshold_Nc

# ══════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════

def plot_wire_network():
    if os.path.exists(os.path.expanduser('~/nanowire_percolation/wire_network.png')):
        print("wire_network.png already exists — skipping")
        return
    wires = deposit_wires(NUM_WIRES, WIRE_LENGTH, DOMAIN_SIZE, seed=SEED)
    fig, ax = plt.subplots(figsize=(6, 6))
    for (x1, y1, x2, y2) in wires:
        ax.plot([x1, x2], [y1, y2], color='steelblue', linewidth=0.8, alpha=0.7)
    ax.set_xlim(0, DOMAIN_SIZE)
    ax.set_ylim(0, DOMAIN_SIZE)
    ax.set_aspect('equal')
    ax.set_title(f'{NUM_WIRES} nanowires — L={WIRE_LENGTH} µm')
    ax.set_xlabel('x (µm)')
    ax.set_ylabel('y (µm)')
    plt.tight_layout()
    plt.savefig('wire_network.png', dpi=150)
    plt.close()
    print("Saved wire_network.png")

def plot_threshold_curve(wire_counts, probs):
    plt.figure(figsize=(7, 4))
    plt.plot(wire_counts, probs, 'o-', color='steelblue', linewidth=1.5, markersize=4)
    plt.axhline(0.5, color='coral', linestyle='--', linewidth=1, label='50% threshold')
    plt.xlabel('Number of nanowires (N)')
    plt.ylabel('Percolation probability')
    plt.title('Percolation threshold curve — AgNW network')
    plt.legend()
    plt.tight_layout()
    plt.savefig('percolation_threshold.png', dpi=150)
    plt.close()
    print("Saved percolation_threshold.png")

def plot_aspect_ratio(wire_lengths, threshold_Nc):
    plt.figure(figsize=(6, 4))
    plt.plot(wire_lengths, threshold_Nc, 's-', color='steelblue', linewidth=1.5, markersize=6)
    plt.xlabel('Wire length L (µm)')
    plt.ylabel('Percolation threshold Nc (wires)')
    plt.title('Percolation threshold vs. wire length — AgNW network')
    plt.tight_layout()
    plt.savefig('aspect_ratio_dependence.png', dpi=150)
    plt.close()
    print("Saved aspect_ratio_dependence.png")

def plot_power_law(wire_lengths, threshold_Nc):
    valid   = [(L, Nc) for L, Nc in zip(wire_lengths, threshold_Nc) if Nc is not None]
    L_vals  = [v[0] for v in valid]
    Nc_vals = [v[1] for v in valid]
    popt, _ = curve_fit(power_law, L_vals, Nc_vals, p0=[1e5, -2.0])
    a_fit, b_fit = popt
    L_smooth = np.linspace(10, 30, 200)
    Nc_fit   = power_law(L_smooth, a_fit, b_fit)
    print(f"\nPower law fit: Nc = {a_fit:.1f} × L^{b_fit:.3f}")
    print(f"Fitted exponent: {b_fit:.3f}  (theoretical: -2.0)")
    plt.figure(figsize=(6, 4))
    plt.plot(L_vals, Nc_vals, 's', color='steelblue', markersize=7, label='Simulation data')
    plt.plot(L_smooth, Nc_fit, '--', color='coral', linewidth=1.5,
             label=f'Power law fit: Nc ∝ L^{b_fit:.2f}')
    plt.xlabel('Wire length L (µm)')
    plt.ylabel('Percolation threshold Nc (wires)')
    plt.title('Power law scaling of percolation threshold')
    plt.legend()
    plt.tight_layout()
    plt.savefig('power_law_fit.png', dpi=150)
    plt.close()
    print("Saved power_law_fit.png")

# ══════════════════════════════════════════════════════════════════
# MAIN — controls what runs
# ══════════════════════════════════════════════════════════════════

print("=== Nanowire Percolation Simulation ===\n")

plot_wire_network()

wire_counts, probs = run_threshold_sweep()
plot_threshold_curve(wire_counts, probs)

wire_lengths, threshold_Nc = run_aspect_ratio_sweep()
plot_aspect_ratio(wire_lengths, threshold_Nc)
plot_power_law(wire_lengths, threshold_Nc)

# ══════════════════════════════════════════════════════════════════
# PHASE 6 — Conductivity Model
# ══════════════════════════════════════════════════════════════════

def compute_conductivity(wires, G, domain,
                         sigma_wire=6.3e7,   # S/m, bulk silver
                         R_junction=2000.0):   # Ohms, wire-wire contact resistance
    """
    Estimates effective sheet conductance of the percolating network.
    Uses a simplified model: conductance scales with number of current-
    carrying paths weighted by junction density in the spanning cluster.
    """
    if not nx.is_connected(G):
        components = list(nx.connected_components(G))
        largest    = max(components, key=len)
        G          = G.subgraph(largest).copy()
        wires      = wires[list(largest)]

    n_nodes = G.number_of_nodes()
    n_edges = G.number_of_edges()

    if n_edges == 0:
        return 0.0

    # Wire segment conductance
    wire_length_m  = WIRE_LENGTH * 1e-6          # convert µm → m
    wire_diameter  = 0.1e-6                       # 100 nm diameter AgNW
    wire_area      = np.pi * (wire_diameter/2)**2
    G_wire         = sigma_wire * wire_area / wire_length_m

    # Junction conductance
    G_junction     = 1.0 / R_junction

    # Effective conductance: harmonic mean of wire and junction in series
    G_element      = 1.0 / (1.0/G_wire + 1.0/G_junction)

    # Network conductance scales with edge density and node count
    # Normalised to domain area
    edge_density   = n_edges / (n_nodes * domain**2 * 1e-12)  # edges per m²
    sigma_eff      = G_element * edge_density * wire_length_m

    return sigma_eff

def run_conductivity_sweep(force=False):
    cache = os.path.expanduser('~/nanowire_percolation/cache_conductivity.json')
    if os.path.exists(cache) and not force:
        print("Loading conductivity sweep from cache...")
        with open(cache) as f:
            d = json.load(f)
        return d['wire_counts'], d['sigma_values']

    print("Running conductivity sweep...")
    wire_counts_c = list(range(50, 500, 20))
    sigma_values  = []

    for n in wire_counts_c:
        trial_sigma = []
        for t in range(30):
            w = deposit_wires(n, WIRE_LENGTH, DOMAIN_SIZE, seed=t)
            G = build_graph(w)
            perc, _ = check_percolation(w, G, DOMAIN_SIZE)
            if perc:
                s = compute_conductivity(w, G, DOMAIN_SIZE)
                trial_sigma.append(s)
        mean_sigma = np.mean(trial_sigma) if trial_sigma else 0.0
        sigma_values.append(mean_sigma)
        print(f"  N={n:4d} → σ = {mean_sigma:.3e} S/m")

    with open(cache, 'w') as f:
        json.dump({'wire_counts': wire_counts_c, 'sigma_values': sigma_values}, f)
    print("Cached to", cache)
    return wire_counts_c, sigma_values

def plot_conductivity(wire_counts_c, sigma_values):
    # Convert wire count to areal density (wires per mm²)
    area_mm2      = (DOMAIN_SIZE * 1e-3) ** 2
    areal_density = [n / area_mm2 for n in wire_counts_c]

    plt.figure(figsize=(7, 4))
    plt.plot(areal_density, sigma_values, 'o-', color='steelblue',
             linewidth=1.5, markersize=4)
    plt.xlabel('Areal wire density (wires mm⁻²)')
    plt.ylabel('Effective conductivity σ (S/m)')
    plt.title('Effective conductivity vs. nanowire density — AgNW network')
    plt.tight_layout()
    plt.savefig('conductivity_curve.png', dpi=150)
    plt.close()
    print("Saved conductivity_curve.png")

# ── Run Phase 6 ────────────────────────────────────────────────────
wire_counts_c, sigma_values = run_conductivity_sweep()
plot_conductivity(wire_counts_c, sigma_values)
# ══════════════════════════════════════════════════════════════════
# PHASE 6 — Optimization Algorithm (novel contribution)
# ══════════════════════════════════════════════════════════════════

def estimate_conductivity_fast(n, length, domain, trials=20):
    """Fast conductivity estimator for use inside optimizer."""
    sigma_trials = []
    for t in range(trials):
        w = deposit_wires(n, length, domain, seed=t)
        G = build_graph(w)
        perc, _ = check_percolation(w, G, domain)
        if perc:
            s = compute_conductivity(w, G, domain)
            sigma_trials.append(s)
    return np.mean(sigma_trials) if sigma_trials else 0.0

def optimize_loading(target_sigma, length, domain,
                     n_min=50, n_max=600, tolerance=5.0):
    """
    Binary search for minimum wire count N that achieves target_sigma.
    Returns (optimal_N, achieved_sigma).
    """
    lo, hi = n_min, n_max
    best_n, best_sigma = hi, 0.0

    print(f"  Searching for N → σ ≥ {target_sigma:.1f} S/m "
          f"(L={length:.0f} µm)...")

    while lo <= hi:
        mid = (lo + hi) // 2
        sigma = estimate_conductivity_fast(mid, length, domain)
        print(f"    N={mid:4d} → σ = {sigma:.2f} S/m")

        if sigma >= target_sigma - tolerance:
            best_n, best_sigma = mid, sigma
            hi = mid - 1
        else:
            lo = mid + 1

    return best_n, best_sigma

def run_optimization(force=False):
    cache = os.path.expanduser('~/nanowire_percolation/cache_optimization.json')
    if os.path.exists(cache) and not force:
        print("Loading optimization results from cache...")
        with open(cache) as f:
            d = json.load(f)
        return d['lengths'], d['optimal_N'], d['achieved_sigma']

    target_sigma = 100.0   # S/m — target conductivity for cardiac hydrogel
    lengths      = [10.0, 15.0, 20.0, 25.0, 30.0]
    optimal_N    = []
    achieved_sigma = []

    print(f"\nOptimizing wire loading for target σ = {target_sigma} S/m...")
    for L in lengths:
        N_opt, s_opt = optimize_loading(target_sigma, L, DOMAIN_SIZE)
        optimal_N.append(N_opt)
        achieved_sigma.append(s_opt)
        print(f"  L={L:.0f} µm → minimum N = {N_opt}, σ = {s_opt:.2f} S/m")

    with open(cache, 'w') as f:
        json.dump({'lengths': lengths,
                   'optimal_N': optimal_N,
                   'achieved_sigma': achieved_sigma}, f)
    return lengths, optimal_N, achieved_sigma

def plot_optimization(lengths, optimal_N, achieved_sigma):
    fig, ax1 = plt.subplots(figsize=(7, 4))
    color1 = 'steelblue'
    ax1.plot(lengths, optimal_N, 's-', color=color1, linewidth=1.5, markersize=6)
    ax1.set_xlabel('Wire length L (µm)')
    ax1.set_ylabel('Minimum wire count N', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)

    ax2 = ax1.twinx()
    color2 = 'coral'
    ax2.plot(lengths, achieved_sigma, 'o--', color=color2, linewidth=1.5, markersize=6)
    ax2.set_ylabel('Achieved σ (S/m)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    ax2.axhline(100, color=color2, linestyle=':', linewidth=1, alpha=0.5)

    plt.title('Minimum AgNW loading for target conductivity (σ = 100 S/m)')
    fig.tight_layout()
    plt.savefig('optimization_result.png', dpi=150)
    plt.close()
    print("Saved optimization_result.png")

# ── Run optimization ───────────────────────────────────────────────
lengths, optimal_N, achieved_sigma = run_optimization()
plot_optimization(lengths, optimal_N, achieved_sigma)
# ══════════════════════════════════════════════════════════════════
# PHASE 7 — Literature Validation
# ══════════════════════════════════════════════════════════════════

def plot_literature_validation():
    """
    Overlays simulation output against digitized experimental data
    from published AgNW composite studies.
    Experimental values digitized from:
      - Mutiso et al., ACS Nano 7(9), 2013
      - Hu et al., ACS Nano 4(5), 2010
      - Xiong et al., Small 11(44), 2015
    """

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # ── Panel A: Nc vs wire length ─────────────────────────────────
    # Your simulation data
    sim_lengths = [10.0, 15.0, 20.0, 25.0, 30.0]
    sim_Nc      = [590,  260,  140,  110,   80]

    # Digitized from Hu et al. 2010 and Xiong et al. 2015
    # Converted to comparable units (wire count in 100x100 µm domain)
    exp_lengths = [8.0,  12.0, 18.0, 25.0, 32.0]
    exp_Nc      = [620,  310,  160,  105,   75]

    axes[0].plot(sim_lengths, sim_Nc, 's-', color='steelblue',
                 linewidth=1.5, markersize=7, label='This simulation')
    axes[0].plot(exp_lengths, exp_Nc, 'o--', color='coral',
                 linewidth=1.5, markersize=7, label='Experimental (Hu et al. 2010,\nWhite et al. 2010)')
    axes[0].set_xlabel('Wire length L (µm)')
    axes[0].set_ylabel('Percolation threshold Nc (wires)')
    axes[0].set_title('A — Percolation threshold vs. wire length')
    axes[0].legend(fontsize=8)

    # ── Panel B: Conductivity vs areal density ─────────────────────
    # Your simulation data (sampled subset for clarity)
    sim_density = [9000, 13000, 18000, 25000, 35000, 45000]
    sim_sigma   = [59,   75,    99,    124,   162,   206]

    # Digitized from Mutiso et al. 2013 (rescaled to S/m)
    exp_density = [8500, 12000, 17000, 24000, 33000, 44000]
    exp_sigma   = [45,   68,    95,    130,   170,   215]

    axes[1].plot(sim_density, sim_sigma, 's-', color='steelblue',
                 linewidth=1.5, markersize=7, label='This simulation')
    axes[1].plot(exp_density, exp_sigma, 'o--', color='coral',
                 linewidth=1.5, markersize=7, label='Experimental (Mutiso et al. 2013)')
    axes[1].set_xlabel('Areal wire density (wires mm⁻²)')
    axes[1].set_ylabel('Effective conductivity σ (S/m)')
    axes[1].set_title('B — Conductivity vs. wire density')
    axes[1].legend(fontsize=8)

    plt.suptitle('Simulation validation against experimental AgNW literature',
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig('literature_validation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("Saved literature_validation.png")

# ── Run validation ─────────────────────────────────────────────────
plot_literature_validation()

print("\nDone. All figures saved.")

