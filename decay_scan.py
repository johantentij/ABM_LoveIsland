"""
Standalone phase-transition scan for the threshold-decay mechanism.

Run from a terminal, independently of any notebook:

    python decay_scan_local.py

Needs `dating_market.py` in the same folder. Writes its OWN outputs
(`decay_results.csv`, `decay_order_params.png`, `decay_fluctuations.png`,
`decay_timeseries.png`) so it never collides with the sensitivity-analysis
notebook's files. Runs as a separate process — your SA run keeps going.

Mechanism: an agent's effective threshold falls the longer it stays single, and
resets automatically when it matches (time_since_single = t - engaged_until, and
engaged_until jumps forward on every match). `decay_rate` is the control parameter.

What to look for in the output:
  * order params (quality, matched fraction) vs decay -> a SHARP drop/jump = transition;
    a gentle monotonic slope = no transition.
  * across-seed std vs decay -> a PEAK near a critical decay = critical fluctuation,
    the hallmark of a real phase transition.
"""

from __future__ import annotations

import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")                      # no display needed; saves PNGs
import matplotlib.pyplot as plt
from scipy.stats import ttest_1samp

from dating_market import Agent, DatingMarket


# ============================ configuration ============================
# The picky / jammed regime is where a positive-feedback cascade (and thus a
# transition) is most likely. Lower THRESHOLD to 0.6 to re-check the easy market.
THRESHOLD   = 0.85
DECAYS      = [0.0, 0.005, 0.01, 0.02, 0.03, 0.05, 0.08, 0.12]
SEEDS       = range(8)                     # raise to ~20 for a clean fluctuation signal
N_AGENTS    = 120
N_GRID      = 45
N_STEPS     = 220
BURN_IN     = 110                          # measure only after this (steady state)
TS_DECAYS   = [0.0, 0.02, 0.08]            # decays to capture full time series for (oscillation check)

MODEL = dict(interaction_std=0.5, interaction_radius=5, relationship_length=10)
BASE  = dict(rejection_cost=1.0, rationality=6.0, memory_depth=8, move_prob=0.5)
# =======================================================================


class DecayAgent(Agent):
    decay_rate = 0.0
    threshold_floor = 0.0

    def _effective_threshold(self) -> float:
        dur = max(0, self.model.t - self.engaged_until)
        return max(self.threshold_floor, self.relation_threshold - self.decay_rate * dur)

    def _expected_utility(self, other_id):
        if other_id not in self._cnt:
            return None
        s = self.samples(other_id)
        if len(s) < 2:
            return None
        res = ttest_1samp(s, self._effective_threshold(), alternative="greater")
        if not np.isfinite(res.pvalue):
            return None
        p = 1.0 - res.pvalue
        return p * (1.0 + self.rejection_cost) - self.rejection_cost


class DecayMarket(DatingMarket):
    def add_agents(self, n, *, decay_rate=0.0, threshold_floor=0.0, **kw):
        sid = super().add_agents(n, **kw)
        for a in self.subjects:
            if a.strategy_id == sid:
                a.__class__ = DecayAgent          # upgrade in place (same attributes)
                a.decay_rate = decay_rate
                a.threshold_floor = threshold_floor
        return sid


def run_decay(decay_rate, seed, keep_series=False):
    m = DecayMarket(n_grid=N_GRID, seed=seed, **MODEL)
    m.add_agents(N_AGENTS, gender_balance=0.5, decay_rate=decay_rate,
                 threshold_floor=0.0, relation_threshold=THRESHOLD, label="pop", **BASE)
    q_series, mf_series, qs, mfs = [], [], [], []
    for t in range(N_STEPS):
        m.step()
        q = m.relationship_quality()
        mf = float(np.mean([not a.is_single for a in m.subjects]))
        if keep_series:
            q_series.append(q); mf_series.append(mf)
        if t >= BURN_IN:
            qs.append(q); mfs.append(mf)
    out = dict(decay=decay_rate, seed=seed,
               quality=float(np.mean(qs)), matched=float(np.mean(mfs)),
               quality_tstd=float(np.std(qs)))
    if keep_series:
        out["q_series"] = q_series; out["mf_series"] = mf_series
    return out


def main():
    t0 = time.time()
    print(f"scanning decay in [{DECAYS[0]}, {DECAYS[-1]}], threshold={THRESHOLD}, "
          f"{len(DECAYS)} points x {len(SEEDS)} seeds")

    rows = []
    for d in DECAYS:
        for s in SEEDS:
            rows.append(run_decay(d, s))
        print(f"  decay={d:<6} done  ({time.time()-t0:.0f}s elapsed)")
    df = pd.DataFrame(rows)
    df.to_csv("decay_results.csv", index=False)

    g = df.groupby("decay")
    summary = pd.DataFrame({
        "quality_mean":    g["quality"].mean(),
        "quality_seedstd": g["quality"].std(),     # across-seed -> critical fluctuation
        "matched_mean":    g["matched"].mean(),
        "matched_seedstd": g["matched"].std(),
        "temporal_std":    g["quality_tstd"].mean(),
    })
    pd.set_option("display.width", 200)
    print("\n", summary.round(4), sep="")

    x = summary.index.values
    nseed = len(SEEDS)

    # --- order parameters vs decay ---
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.2))
    ax[0].errorbar(x, summary["quality_mean"],
                   yerr=summary["quality_seedstd"] / np.sqrt(nseed), marker="o", capsize=3)
    ax[0].set_xlabel("decay_rate"); ax[0].set_ylabel("mean matched quality")
    ax[0].set_title("order parameter: quality"); ax[0].grid(alpha=.3)
    ax[1].errorbar(x, summary["matched_mean"],
                   yerr=summary["matched_seedstd"] / np.sqrt(nseed), marker="s", capsize=3, color="C1")
    ax[1].set_xlabel("decay_rate"); ax[1].set_ylabel("matched fraction")
    ax[1].set_title("order parameter: matched fraction"); ax[1].grid(alpha=.3)
    fig.tight_layout(); fig.savefig("decay_order_params.png", dpi=110)

    # --- fluctuations vs decay (look for a peak) ---
    fig, ax = plt.subplots(figsize=(7, 4.2))
    ax.plot(x, summary["quality_seedstd"], "-o", label="across-seed std (critical fluctuation)")
    ax.plot(x, summary["temporal_std"], "-s", label="within-run temporal std")
    ax.set_xlabel("decay_rate"); ax.set_ylabel("std of quality")
    ax.set_title("fluctuations vs decay  (a peak => phase transition)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("decay_fluctuations.png", dpi=110)

    # --- time series at a few decays (oscillation / settling check) ---
    fig, ax = plt.subplots(figsize=(9, 4.2))
    for d in TS_DECAYS:
        r = run_decay(d, seed=0, keep_series=True)
        ax.plot(r["q_series"], label=f"decay={d}")
    ax.axvline(BURN_IN, color="0.6", lw=.8, ls="--")
    ax.set_xlabel("step"); ax.set_ylabel("matched quality")
    ax.set_title("quality time series (oscillation check; dashed = burn-in)")
    ax.legend(); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig("decay_timeseries.png", dpi=110)

    print(f"\ndone in {time.time()-t0:.0f}s")
    print("wrote: decay_results.csv, decay_order_params.png, "
          "decay_fluctuations.png, decay_timeseries.png")


if __name__ == "__main__":
    main()
