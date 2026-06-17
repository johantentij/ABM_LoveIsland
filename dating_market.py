"""
Agent-based model of a dating market.

Agents live on a toroidal (wrap-around) grid. Each step, single agents may
move, then observe opposite-gender singles within an interaction radius,
gathering *noisy* samples of a latent mutual compatibility. Each agent keeps a
short memory of recent samples per candidate and runs a one-sample t-test to
decide whether it is confident the true compatibility exceeds its personal
threshold. It proposes to its most convincing candidate; a proposal succeeds
only if it is mutual. Matched pairs leave the market for a while (longer when
more compatible), then return single.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import ttest_1samp


class Agent:
    def __init__(
        self,
        model: "DatingMarket",
        agent_id: int,
        pos: tuple[int, int],
        is_male: bool,
        move_prob: float,
        risk_aversion: float,
        n_subjects: int,
        memory_depth: int = 5,
        relation_threshold: float = 0.8,
    ):
        self.model = model
        self.id = agent_id
        self.pos = pos
        self.is_male = is_male
        self.move_prob = move_prob
        self.memory_depth = memory_depth
        self.relation_threshold = relation_threshold

        # Significance level for the t-test. A more risk-averse agent demands
        # stronger statistical evidence (a smaller alpha) before proposing.
        self.alpha = max(1e-3, 1.0 - risk_aversion)

        # Step index at which this agent becomes single again. While
        # model.t < engaged_until the agent is in a relationship.
        self.engaged_until = 0

        # Ring buffer of recent observations per potential partner, plus a count.
        self.observations = np.zeros((n_subjects, memory_depth))
        self.n_observations = np.zeros(n_subjects, dtype=int)

        # Candidates found this step: list of (other_id, p_value).
        self.candidates: list[tuple[int, float]] = []

    @property
    def is_single(self) -> bool:
        return self.model.t >= self.engaged_until

    def observe(self, other_id: int) -> None:
        """Record one fresh noisy sample of compatibility with `other_id`."""
        slot = self.n_observations[other_id] % self.memory_depth
        self.observations[other_id, slot] = self.model.sample_compatibility(self.id, other_id)
        self.n_observations[other_id] += 1

    def evaluate_neighbours(self) -> None:
        self.candidates = []
        for other_id in self.model.neighbours(self.id):
            other = self.model.subjects[other_id]
            if other.is_male == self.is_male or not other.is_single:
                continue

            self.observe(other_id)

            n = min(self.n_observations[other_id], self.memory_depth)
            if n < 2:
                continue  # need >= 2 samples to estimate a variance

            samples = self.observations[other_id, :n]
            result = ttest_1samp(samples, self.relation_threshold, alternative="greater")

            # Small p-value => strong evidence that true compatibility exceeds
            # the threshold. Keep candidates that clear this agent's alpha.
            if np.isfinite(result.pvalue) and result.pvalue <= self.alpha:
                self.candidates.append((other_id, float(result.pvalue)))

    def propose(self) -> None:
        if not self.candidates:
            return
        # Most convincing candidate = smallest p-value.
        best_id, _ = min(self.candidates, key=lambda c: c[1])
        engaged_until = self.model.subjects[best_id].handle_proposal(self.id)
        if engaged_until:  # 0 means the proposal was declined
            self.engaged_until = engaged_until

    def handle_proposal(self, proposer_id: int) -> int:
        # Accept only if the proposer is mutually one of my candidates.
        if any(other_id == proposer_id for other_id, _ in self.candidates):
            duration = self.model.relationship_length(self.id, proposer_id)
            self.engaged_until = self.model.t + duration
            return self.engaged_until
        return 0

    def move(self) -> None:
        if self.model.rng.random() > self.move_prob:
            return
        options = self.model.movement_options(self.id)
        if options:
            new_pos = options[self.model.rng.integers(len(options))]
            self.model.grid[self.pos] = DatingMarket.EMPTY
            self.model.grid[new_pos] = self.id
            self.pos = new_pos


# The all-knowing being that decides who is compatible with whom.
class DatingMarket:
    EMPTY = -1

    def __init__(
        self,
        n_grid: int,
        n_subjects: int,
        gender_balance: float = 0.5,
        move_prob: float = 0.5,
        risk_aversion: float = 0.99,
        interaction_std: float = 0.5,
        interaction_radius: int = 5,
        memory_depth: int = 5,
        relation_threshold: float = 0.8,
        max_relationship_length: int = 10,
        seed: int | None = None,
    ):
        if n_subjects > n_grid * n_grid:
            raise ValueError("more subjects than grid cells")

        self.rng = np.random.default_rng(seed)
        self.n_grid = n_grid
        self.n_subjects = n_subjects
        self.interaction_std = interaction_std
        self.interaction_radius = interaction_radius
        self.max_relationship_length = max_relationship_length
        self.t = 0  # global step clock

        self.n_males = int(round(gender_balance * n_subjects))
        self.n_females = n_subjects - self.n_males

        # Latent mutual compatibility for each (male, female) pair, in [0, 1].
        self.compatibility = self.rng.random((self.n_males, self.n_females))

        self.grid = np.full((n_grid, n_grid), self.EMPTY, dtype=np.int32)
        self.subjects: list[Agent] = []

        # Place everyone on distinct random cells. ids 0..n_males-1 are male,
        # the remaining ids are female.
        free_cells = [(r, c) for r in range(n_grid) for c in range(n_grid)]
        self.rng.shuffle(free_cells)
        for agent_id in range(n_subjects):
            pos = free_cells[agent_id]
            is_male = agent_id < self.n_males
            self.grid[pos] = agent_id
            self.subjects.append(
                Agent(self, agent_id, pos, is_male, move_prob, risk_aversion,
                      n_subjects, memory_depth, relation_threshold)
            )

    # -- compatibility-matrix helpers --------------------------------------

    def _male_female_index(self, id1: int, id2: int) -> tuple[int, int]:
        """Map an (agent, agent) pair to (male_row, female_col)."""
        if self.subjects[id1].is_male:
            male_id, female_id = id1, id2
        else:
            male_id, female_id = id2, id1
        return male_id, female_id - self.n_males

    def sample_compatibility(self, observer_id: int, other_id: int) -> float:
        m, f = self._male_female_index(observer_id, other_id)
        return self.compatibility[m, f] + self.rng.normal(0.0, self.interaction_std)

    def relationship_length(self, id1: int, id2: int) -> int:
        m, f = self._male_female_index(id1, id2)
        return max(1, round(self.max_relationship_length * self.compatibility[m, f]))

    # -- spatial queries ---------------------------------------------------

    def neighbours(self, agent_id: int) -> list[int]:
        r0, c0 = self.subjects[agent_id].pos
        r = self.interaction_radius
        result = []
        for dr in range(-r, r + 1):
            rr = (r0 + dr) % self.n_grid
            for dc in range(-r, r + 1):
                cc = (c0 + dc) % self.n_grid
                if (rr, cc) == (r0, c0):
                    continue
                occupant = self.grid[rr, cc]
                if occupant != self.EMPTY:
                    result.append(int(occupant))
        return result

    def movement_options(self, agent_id: int) -> list[tuple[int, int]]:
        r0, c0 = self.subjects[agent_id].pos
        options = []
        for dr in (-1, 0, 1):
            rr = (r0 + dr) % self.n_grid
            for dc in (-1, 0, 1):
                cc = (c0 + dc) % self.n_grid
                if (rr, cc) == (r0, c0):
                    continue
                if self.grid[rr, cc] == self.EMPTY:
                    options.append((rr, cc))
        return options

    # -- simulation --------------------------------------------------------

    def step(self) -> None:
        for agent_id in self.rng.permutation(self.n_subjects):
            agent = self.subjects[agent_id]
            if agent.is_single:
                agent.move()
                agent.evaluate_neighbours()
                agent.propose()
        self.t += 1

    def stats(self) -> dict[str, int]:
        single = sum(a.is_single for a in self.subjects)
        return {
            "t": self.t,
            "single": single,
            "engaged": self.n_subjects - single,
            "couples": (self.n_subjects - single) // 2,
        }

    def run(self, n_steps: int) -> list[dict[str, int]]:
        history = [self.stats()]
        for _ in range(n_steps):
            self.step()
            history.append(self.stats())
        return history


if __name__ == "__main__":
    market = DatingMarket(
        n_grid=40,
        n_subjects=200,
        move_prob=0.6,
        risk_aversion=0.99,
        interaction_radius=4,
        seed=0,
    )
    history = market.run(80)
    for snapshot in history[::10]:
        print(
            f"t={snapshot['t']:>3}  "
            f"single={snapshot['single']:>3}  "
            f"engaged={snapshot['engaged']:>3}  "
            f"couples={snapshot['couples']:>3}"
        )