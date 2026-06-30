"""
Stable-matching benchmark (M*) and distance/ratio metrics for the dating market.

This module only adds a subclass, `InstrumentedDatingMarket`, that behaves 
exactly like `DatingMarket` for the simulation itself, but additionally: 
computes the unique stable matching M* (greedy on descending true
compatibility, respecting each agent's own threshold), and metrics that
compare the realized matching to M*: exact-partner overlap, Jaccard over
couples, a welfare ratio, and a blocking-pair count.

"""

from __future__ import annotations

import numpy as np

from dating_market_Alex import DatingMarket


class InstrumentedDatingMarket(DatingMarket):
    def __init__(self, *args, seed: int | None = None, **kwargs):
        super().__init__(*args, seed=seed, **kwargs)
        # Independent RNG for the latent matrix; never touches the dynamics RNG.
        self._true_rng = np.random.default_rng(np.random.SeedSequence(seed).spawn(1)[0])
        self._finalized = False
        self._contacted: set[tuple[int, int]] = set()   # (male_id, female_id) ever in range
        self._stable_global: dict[int, int] | None = None
        self._males: list[int] = []
        self._females: list[int] = []
        self._thr: dict[int, float] = {}

    # -- freeze the true compatibility matrix once --------------------------

    def finalize(self) -> None:
        """Materialize every (male, female) true value with the separate RNG.

        Idempotent. Fires automatically on the first step(); you can also call it
        by hand after the last add_agents() if you want to inspect M* before
        running.
        """
        if self._finalized:
            return
        self._males = [a.id for a in self.subjects if a.is_male]
        self._females = [a.id for a in self.subjects if not a.is_male]
        for m in self._males:
            for f in self._females:
                if (m, f) not in self._compat:        # don't clobber anything pre-drawn
                    self._compat[(m, f)] = float(self._true_rng.random())
        self._thr = {a.id: a.relation_threshold for a in self.subjects}
        self._finalized = True

    def step(self) -> None:
        if not self._finalized:
            self.finalize()
        super().step()

    # -- contact tracking (spatial reachability) ----------------------------

    def sample_compatibility(self, observer_id: int, other_id: int) -> float:
        # Same return value as the base method; we only note the contact.
        if self.subjects[observer_id].is_male:
            self._contacted.add((observer_id, other_id))
        else:
            self._contacted.add((other_id, observer_id))
        return super().sample_compatibility(observer_id, other_id)

    def _c(self, m_id: int, f_id: int) -> float:
        """True compatibility, keyed (male, female) like the base store."""
        return self._compat[(m_id, f_id)]

    # -- stable benchmark M* ------------------------------------------------

    def compute_stable_matching(self, allowed: set[tuple[int, int]] | None = None
                                ) -> dict[int, int]:
        """Unique stable matching by greedy descending edge weight.

        A pair (m, f) is eligible only if true compatibility clears BOTH agents'
        thresholds (each prefers the partner to staying single). `allowed`, if
        given, restricts to a set of (male_id, female_id) pairs -- pass the
        contact set for the spatially-constrained benchmark.

        Returns a dict with BOTH directions: match[m] = f and match[f] = m.
        """
        if not self._finalized:
            self.finalize()
        edges = []
        for m in self._males:
            tm = self._thr[m]
            for f in self._females:
                if allowed is not None and (m, f) not in allowed:
                    continue
                c = self._compat[(m, f)]
                if c >= tm and c >= self._thr[f]:
                    edges.append((c, m, f))
        edges.sort(reverse=True)                      # highest compatibility first
        match: dict[int, int] = {}
        for _, m, f in edges:
            if m not in match and f not in match:
                match[m] = f
                match[f] = m
        return match

    def stable_global(self) -> dict[int, int]:
        """Global M* -- fixed for the whole run, so computed once and cached."""
        if self._stable_global is None:
            self._stable_global = self.compute_stable_matching()
        return self._stable_global

    def stable_spatial(self) -> dict[int, int]:
        """M* restricted to pairs that have ever been in sensing range.

        Recomputed on demand because the contact set grows over time. Call it at
        the end of your steady-state window, not every step.
        """
        return self.compute_stable_matching(allowed=self._contacted)

    # -- realized matching & comparison metrics -----------------------------

    def _realized(self) -> dict[int, int]:
        return {a.id: a.partner for a in self.subjects
                if not a.is_single and a.partner is not None}

    def _percap_welfare(self, match: dict[int, int]) -> float:
        """Mean true compatibility per agent (singles contribute 0)."""
        if not self.subjects:
            return 0.0
        total = 0.0
        for i, j in match.items():
            total += self._compat[(i, j)] if self.subjects[i].is_male else self._compat[(j, i)]
        return (total / 2.0) / len(self.subjects)     # each couple counted twice above

    def count_blocking_pairs(self, realized: dict[int, int] | None = None,
                             allowed: set[tuple[int, int]] | None = None) -> int:
        """Number of (m, f) that would both rather be together than stay put.

        A single agent's reservation value is its own threshold. Pass
        allowed=self._contacted for the spatially-constrained version.
        """
        if realized is None:
            realized = self._realized()
        val: dict[int, float] = {}
        for a in self.subjects:
            if a.id in realized:
                p = realized[a.id]
                val[a.id] = self._compat[(a.id, p)] if a.is_male else self._compat[(p, a.id)]
            else:
                val[a.id] = self._thr[a.id]           # reservation = own threshold
        count = 0
        for m in self._males:
            for f in self._females:
                if allowed is not None and (m, f) not in allowed:
                    continue
                if realized.get(m) == f:
                    continue
                c = self._compat[(m, f)]
                if c > val[m] and c > val[f]:
                    count += 1
        return count

    def matching_metrics(self, benchmark: dict[int, int] | None = None) -> dict:
        """Compare the current realized matching to a benchmark (global M* by default).

        Keys:
          partner_frac_matched : among CURRENTLY MATCHED agents, fraction whose
                                 partner is exactly their benchmark partner.
                                 (rotation/vacancy-robust -- the headline number)
          coverage_of_Mstar    : among agents the benchmark pairs up, fraction now
                                 realized with that exact partner (includes vacancy)
          jaccard              : overlap of couple sets, |R n S| / |R u S|
          welfare_ratio        : realized per-capita quality / benchmark per-capita
                                 quality, in [0, 1]
          blocking_pairs       : blocking pairs over the realized matching
                                 (global; allowed=None)
        """
        if benchmark is None:
            benchmark = self.stable_global()
        realized = self._realized()

        matched_ids = list(realized.keys())
        if matched_ids:
            hit = sum(1 for i in matched_ids if benchmark.get(i) == realized[i])
            partner_frac_matched = hit / len(matched_ids)
        else:
            partner_frac_matched = float("nan")

        if benchmark:
            cov = sum(1 for i in benchmark if realized.get(i) == benchmark[i]) / len(benchmark)
        else:
            cov = float("nan")

        R = {frozenset((i, realized[i])) for i in realized}
        S = {frozenset((i, benchmark[i])) for i in benchmark}
        union = R | S
        jaccard = len(R & S) / len(union) if union else float("nan")

        w_real = self._percap_welfare(realized)
        w_star = self._percap_welfare(benchmark)
        welfare_ratio = (w_real / w_star) if w_star > 0 else float("nan")

        return {
            "partner_frac_matched": partner_frac_matched,
            "coverage_of_Mstar": cov,
            "jaccard": jaccard,
            "welfare_ratio": welfare_ratio,
            "blocking_pairs": self.count_blocking_pairs(realized, allowed=None),
        }

    def spatial_metrics(self) -> dict:
        """Same metrics, but against the spatially-constrained M* and with
        blocking pairs counted only over reachable (contacted) pairs."""
        bench = self.stable_spatial()
        out = self.matching_metrics(benchmark=bench)
        out["blocking_pairs"] = self.count_blocking_pairs(allowed=self._contacted)
        return out


if __name__ == "__main__":
    m = InstrumentedDatingMarket(n_grid=50, interaction_std=0.5, interaction_radius=5,
                                 relationship_length=10, seed=0)
    m.add_agents(240, gender_balance=0.5, rejection_cost=1.0, rationality=6.0,
                 relation_threshold=0.6, memory_depth=8, move_prob=0.5)

    glob_hist, spat_hist = [], []
    for t in range(200):
        m.step()
        if t >= 80:
            glob_hist.append(m.matching_metrics())
            if t % 20 == 0:                            # spatial M* is pricier; sample it
                spat_hist.append(m.spatial_metrics())

    def avg(hist, k):
        xs = [h[k] for h in hist if h[k] == h[k]]      # drop NaN
        return float(np.mean(xs)) if xs else float("nan")

    print("vs GLOBAL M* (steady-state mean):")
    for k in ("partner_frac_matched", "coverage_of_Mstar", "jaccard",
              "welfare_ratio", "blocking_pairs"):
        print(f"  {k:>22} = {avg(glob_hist, k):.3f}")
    print("vs SPATIAL M* (steady-state mean):")
    for k in ("partner_frac_matched", "coverage_of_Mstar", "jaccard",
              "welfare_ratio", "blocking_pairs"):
        print(f"  {k:>22} = {avg(spat_hist, k):.3f}")
