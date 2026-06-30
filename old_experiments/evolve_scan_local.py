"""
Standalone evolutionary-dynamics scan for the dating market.

Run from a terminal, independently of any notebook:

    python evolve_scan_local.py

Needs `dating_market.py` in the same folder. Writes its OWN outputs
(`evo_results.csv`, `evo_order_params.png`, `evo_fluctuations.png`,
`evo_distributions.png`, `evo_trajectories.png`) so it never collides with other runs.

Idea
----
`rejection_cost` (an agent's "pickiness gene") is heritable. Each generation:
  1. a fresh market is run for GEN_STEPS; fitness = cumulative compatibility per agent
     (outcome-based: time spent in good matches -- NOT the decision utility, which would
     directly penalise the gene and bias the evolution);
  2. the next generation's genes are produced by selecting parents with probability
     softmax(BETA * z_fitness) and copying their gene plus a small Gaussian mutation.

BETA (selection strength) is the control parameter / phase-transition axis:
  * BETA -> 0  : pure drift (parents chosen at random) -> genes random-walk -> high diversity
  * BETA large : strong selection -> population collapses onto the best gene -> low diversity

What to look for
----------------
  * gene diversity (std across population) vs BETA: a SHARP drop = order-disorder transition.
  * gene-distribution histograms: a BIMODAL shape at intermediate BETA = evolutionary
    BRANCHING (one picky sub-population + one easy-going one) -- the most interesting outcome.
  * across-seed std of the mean gene vs BETA: a PEAK near a critical BETA = critical fluctuation.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dating_market import DatingMarket


# ============================ configuration ============================
GENE_BOUNDS   = (0.05, 3.0)        # rejection_cost range
MUT_STD       = 0.12               # mutation step (gene units)
N_AGENTS      = 120
N_GRID        = 45
GEN_STEPS     = 30                 # market steps per generation
GEN_BURN      = 10                 # ignore the first few steps (matches still forming)
N_GENERATIONS = 60
BETAS         = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0]   # selection strengths to scan
SEEDS         = range(5)           # raise to ~12 for a cleaner fluctuation signal
TS_BETAS      = [0.0, 1.0, 5.0]    # betas to plot trajectories / histograms for
STEADY_FRAC   = 0.5                # measure over the last fraction of generations

MODEL = dict(interaction_std=0.5, interaction_radius=5, relationship_length=10)
BASE  = dict(rationality=6.0, relation_threshold=0.6, memory_depth=8, move_prob=0.5)
# =======================================================================


def run_generation(genes, gen_seed):
    """Fresh market seeded with the given per-agent genes; return fitness per agent."""
    m = DatingMarket(n_grid=N_GRID, seed=gen_seed, **MODEL)
    m.add_agents(N_AGENTS, gender_balance=0.5, rejection_cost=1.0, label="pop", **BASE)
    for a, g in zip(m.subjects, genes):
        a.rejection_cost = float(g)            # install the heritable gene
    fitness = np.zeros(N_AGENTS)
    for t in range(GEN_STEPS):
        m.step()
        if t >= GEN_BURN:
            for a in m.subjects:
                fitness[a.id] += m.agent_quality(a)   # cumulative compatibility
    return fitness


def evolve(beta, seed):
    rng = np.random.default_rng(seed)
    genes = rng.uniform(*GENE_BOUNDS, size=N_AGENTS)   # start diverse
    traj_mean, traj_std = [], []
    for gen in range(N_GENERATIONS):
        fitness = run_generation(genes, gen_seed=seed * 100_000 + gen)
        mu, sd = fitness.mean(), fitness.std()
        z = (fitness - mu) / sd if sd > 1e-9 else np.zeros_like(fitness)
        w = np.exp(beta * z - (beta * z).max())
        p = w / w.sum()
        parents = rng.choice(N_AGENTS, size=N_AGENTS, p=p)
        genes = np.clip(genes[parents] + rng.normal(0, MUT_STD, N_AGENTS), *GENE_BOUNDS)
        traj_mean.append(genes.mean()); traj_std.append(genes.std())
    return np.array(traj_mean), np.array(traj_std), genes


def main():
    t0 = time.time()
    print(f"evolving rejection_cost; scanning BETA={BETAS}, {len(list(SEEDS))} seeds, "
          f"{N_GENERATIONS} generations x {GEN_STEPS} steps")
    k0 = int(N_GENERATIONS * (1 - STEADY_FRAC))

    rows = []
    traj_store = {b: [] for b in TS_BETAS}
    dist_store = {b: [] for b in TS_BETAS}
    for b in BETAS:
        for s in SEEDS:
            tmean, tstd, final_genes = evolve(b, s)
            rows.append(dict(beta=b, seed=s,
                             mean_gene=float(tmean[k0:].mean()),
                             gene_diversity=float(tstd[k0:].mean())))
            if b in TS_BETAS:
                traj_store[b].append((tmean, tstd))
                dist_store[b].append(final_genes)
        print(f"  beta={b:<5} done  ({time.time()-t0:.0f}s elapsed)")

    df = pd.DataFrame(rows); df.to_csv("evo_results.csv", index=False)
    g = df.groupby("beta")
    summary = pd.DataFrame({
        "mean_gene":      g["mean_gene"].mean(),
        "mean_gene_seedstd": g["mean_gene"].std(),     # across-seed -> critical fluctuation
        "gene_diversity": g["gene_diversity"].mean(),
    })
    pd.set_option("display.width", 200)
    print("\n", summary.round(4), sep="")

    x = summary.index.values
    nseed = len(list(SEEDS))

    # --- order parameters ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].errorbar(x, summary["mean_gene"],
                   yerr=summary["mean_gene_seedstd"] / np.sqrt(nseed), marker="o", capsize=3)
    ax[0].set_xlabel("selection strength beta"); ax[0].set_ylabel("population mean rejection_cost")
    ax[0].set_title("order parameter: mean gene"); ax[0].grid(alpha=.3)
    ax[1].plot(x, summary["gene_diversity"], "-s", color="C1")
    ax[1].set_xlabel("selection strength beta"); ax[1].set_ylabel("gene std across population")
    ax[1].set_title("order parameter: diversity (sharp drop => order-disorder)"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig("evo_order_params.png", dpi=110)

    # --- fluctuation ---
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, summary["mean_gene_seedstd"], "-o")
    ax.set_xlabel("selection strength beta"); ax.set_ylabel("across-seed std of mean gene")
    ax.set_title("fluctuation vs beta  (a peak => phase transition)"); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("evo_fluctuations.png", dpi=110)

    # --- gene distributions (branching check) ---
    fig, ax = plt.subplots(1, len(TS_BETAS), figsize=(5 * len(TS_BETAS), 3.8), sharey=True)
    for a_, b in zip(np.atleast_1d(ax), TS_BETAS):
        pooled = np.concatenate(dist_store[b]) if dist_store[b] else np.array([])
        a_.hist(pooled, bins=30, range=GENE_BOUNDS, color="C2", alpha=.8)
        a_.set_title(f"beta={b}\n(bimodal => branching)"); a_.set_xlabel("rejection_cost")
    fig.tight_layout(); fig.savefig("evo_distributions.png", dpi=110)

    # --- evolution trajectories ---
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for b in TS_BETAS:
        if not traj_store[b]:
            continue
        tmean = np.mean([t[0] for t in traj_store[b]], axis=0)
        tstd  = np.mean([t[1] for t in traj_store[b]], axis=0)
        line, = ax.plot(tmean, label=f"beta={b}")
        ax.fill_between(range(len(tmean)), tmean - tstd, tmean + tstd,
                        color=line.get_color(), alpha=.15)
    ax.set_xlabel("generation"); ax.set_ylabel("mean gene (band = population std)")
    ax.set_title("evolution of rejection_cost"); ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("evo_trajectories.png", dpi=110)

    print(f"\ndone in {time.time()-t0:.0f}s")
    print("wrote: evo_results.csv, evo_order_params.png, evo_fluctuations.png, "
          "evo_distributions.png, evo_trajectories.png")


if __name__ == "__main__":
    main()
