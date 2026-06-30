"""
Agent-based model of a dating market (fixed-length relationships).

Each step, agents act one at a time in random order.

  * SINGLE agents sample their opposite-gender neighbours, then choose at most
    one single neighbour to propose to via a logit rule over expected utilities.
    A proposal is resolved immediately: the target evaluates the proposer with
    its OWN t-test and accepts stochastically. On a match, both partners are
    committed for a fixed `relationship_length` (default 10) steps.

  * ENGAGED agents still observe their surroundings (their memory stays current)
    but cannot move, propose, or handle proposals until the relationship ends.

Decision model
--------------
For a candidate, a one-sided one-sample t-test on the agent's recent samples
gives p = 1 - pvalue, the confidence that true compatibility exceeds the
agent's threshold (its estimated probability of acceptance). The decision tree:

        don't propose -> u = 0
        propose       -> u = +1  (reciprocated)   with prob p
                         u = -a  (rejected)        with prob 1 - p

so the expected utility of proposing is  EU = p * (1 + a) - a, where `a` is the
cost of rejection. The proposer logit-chooses among candidate EUs plus a
no-proposal option (EU = 0) with personal rationality beta. When proposed to, a
single target accepts with probability sigmoid(beta * EU_target). Handling a
proposal is also scored: accepting adds +1 utility, rejecting adds 0.

Populations are built with `add_agents(...)`; each call is recorded as a
*strategy* with an id so groups can be compared.
"""

from __future__ import annotations

import numpy as np
import ctypes
import scipy.special.cython_special as cysp
from numba import njit, int64, int32
from numba.types import ListType, UniTuple
from numba.extending import get_cython_function_address

# Grab the stable C address for the Student's T CDF function directly
stdtr_capsule_name = next((k for k in cysp.__pyx_capi__.keys() if "stdtr" in k), None)
addr = get_cython_function_address("scipy.special.cython_special", stdtr_capsule_name)

# Define the explicit C-signature: double stdtr(double, double)
stdtr_sig = ctypes.CFUNCTYPE(ctypes.c_double, ctypes.c_double, ctypes.c_double)
stdtr_c = stdtr_sig(addr)

@njit
def set_numba_seed(SEED):
    np.random.seed(SEED)
    return

@njit
def numba_t_sf(t, df):
    """
    Computes the right-tailed survival function (p-value) using SciPy's C-backend.
    Equivalent to 1.0 - cdf
    """
    # stdtr_c(df, t) returns the left-tail CDF area.
    # Subtracting from 1.0 gives the right-tail Survival Function (SF).
    return 1.0 - stdtr_c(df, t)


@njit
def t_test(data, pop_mean):
    n = len(data)
    sample_mean = np.mean(data)

    # Manually calculate sample standard deviation (ddof=1)
    variance = np.sum((data - sample_mean) ** 2) / (n - 1)
    sample_std = np.sqrt(variance)

    df = n - 1

    # Calculate standard error and the t-statistic
    standard_error = sample_std / np.sqrt(n)
    t_stat = (sample_mean - pop_mean) / standard_error
    p_val = numba_t_sf(t_stat, df)
    return t_stat, p_val

@njit(int64[:](int64, int32[:, :], int64, int64, UniTuple(int64, 2)),cache=True)
def numba_neighbours(EMPTY, grid, n_grid, interaction_radius, current_agent_pos):
    r0, c0 = current_agent_pos[0], current_agent_pos[1]
    r = interaction_radius
    out_list = []
    for dr in range(-r, r + 1):
        rr = (r0 + dr) % n_grid
        for dc in range(-r, r + 1):
            cc = (c0 + dc) % n_grid
            if rr == r0 and cc == c0:
                continue
            occ = grid[rr, cc]
            if occ != EMPTY:
                out_list.append(int64(occ))

    res_array = np.empty(len(out_list), dtype=np.int64)
    for i in range(len(out_list)):
        res_array[i] = out_list[i]

    return res_array


@njit(cache=True)
def numba_sample_compatibility(base_compatibility, interaction_std):
    """
    Fully compiled JIT function that synchronizes with your master random seed.
    """
    # This random draw is now completely deterministic and linked to your seed!
    return base_compatibility + np.random.normal(0.0, interaction_std)

# The all-knowing being that decides who is compatible with whom.
class DatingMarket:
    EMPTY = -1

    def __init__(
        self,
        n_grid: int,
        interaction_std: float = 0.5,
        interaction_radius: int = 5,
        relationship_length: int = 10,
        seed: int | None = None,
    ):
        self.rng = np.random.default_rng(seed)
        set_numba_seed(seed)
        self.n_grid = n_grid
        self.interaction_std = interaction_std
        self.interaction_radius = interaction_radius
        self.relationship_length = relationship_length
        self.t = 0

        self.grid = np.full((n_grid, n_grid), self.EMPTY, dtype=np.int32)
        self.subjects: list[Agent] = []

        self._compat: dict[tuple[int, int], float] = {}

        self.strategies: dict[int, dict] = {}
        self.history: list[dict] = []
        self.strategy_history: dict[int, list[dict]] = {}

        self.rej_cost_matrix_list: list[float] = []

    # -- population building -------------------------------------------------

    def add_agents(
        self,
        n: int,
        *,
        gender_balance: float = 0.5,
        move_prob: float = 0.5,
        rejection_cost: float = 0.5,
        rationality: float = 5.0,
        relation_threshold: float = 0.8,
        memory_depth: int = 10,
        agent_rational: str = 'freq',
        label: str | None = None,
        decay_rate_rej = 1
    ) -> int:
        """Add `n` agents sharing one strategy. Returns the strategy id."""
        strategy_id = len(self.strategies)
        self.strategies[strategy_id] = {
            "label": label or f"strategy_{strategy_id}",
            "n": n,
            "gender_balance": gender_balance,
            "move_prob": move_prob,
            "rejection_cost": rejection_cost,
            "rationality": rationality,
            "relation_threshold": relation_threshold,
            "memory_depth": memory_depth,
            "agent_rational": agent_rational,
            "decay_rate_rej": decay_rate_rej
        }
        self.strategy_history[strategy_id] = []

        # self.rej_cost_matrix_list[strategy_id] = np.full(shape=(n), fill_value=rejection_cost, dtype=np.float64)

        n_male = int(round(gender_balance * n))
        genders = [True] * n_male + [False] * (n - n_male)
        self.rng.shuffle(genders)

        for is_male in genders:
            pos = self._random_free_cell()
            agent_id = len(self.subjects)
            agent = Agent(
                self, agent_id, pos, is_male, strategy_id,
                move_prob, rejection_cost, rationality,
                relation_threshold, memory_depth, agent_rational=agent_rational,decay_rate_rej=decay_rate_rej
            )
            self.grid[pos] = agent_id
            self.subjects.append(agent)
            self.rej_cost_matrix_list.append(rejection_cost)

        return strategy_id

    def _random_free_cell(self) -> tuple[int, int]:
        free = np.argwhere(self.grid == self.EMPTY)
        if len(free) == 0:
            raise RuntimeError("grid is full; use a larger n_grid")
        r, c = free[self.rng.integers(len(free))]
        return (int(r), int(c))

    # -- compatibility -------------------------------------------------------

    def compatibility(self, id_a: int, id_b: int) -> float:
        if self.subjects[id_a].is_male:
            key = (id_a, id_b)
        else:
            key = (id_b, id_a)
        val = self._compat.get(key)
        if val is None:
            val = float(self.rng.random())
            self._compat[key] = val
        return val

    def sample_compatibility(self, observer_id: int, other_id: int) -> float:
        base_comp = self.compatibility(observer_id, other_id)
        return numba_sample_compatibility(base_comp, self.interaction_std)
        # return self.compatibility(observer_id, other_id) + self.rng.normal(0.0, self.interaction_std)

    # -- spatial queries -----------------------------------------------------

    def neighbours(self, agent_id: int) -> list[int]:
        r0, c0 = self.subjects[agent_id].pos
        r = self.interaction_radius
        out = []
        for dr in range(-r, r + 1):
            rr = (r0 + dr) % self.n_grid
            for dc in range(-r, r + 1):
                cc = (c0 + dc) % self.n_grid
                if (rr, cc) == (r0, c0):
                    continue
                occ = self.grid[rr, cc]
                if occ != self.EMPTY:
                    out.append(int(occ))
        return out

    def movement_options(self, agent_id: int) -> list[tuple[int, int]]:
        r0, c0 = self.subjects[agent_id].pos
        out = []
        for dr in (-1, 0, 1):
            rr = (r0 + dr) % self.n_grid
            for dc in (-1, 0, 1):
                cc = (c0 + dc) % self.n_grid
                if (rr, cc) == (r0, c0):
                    continue
                if self.grid[rr, cc] == self.EMPTY:
                    out.append((rr, cc))
        return out

    # -- simulation ----------------------------------------------------------

    def step(self) -> None:
        n = len(self.subjects)
        for i in self.rng.permutation(n):
            a = self.subjects[int(i)]
            if a.is_single:
                a.act()
            else:
                a.observe_surroundings()           # engaged: sense but don't act
        # only single agents may relocate
        for i in self.rng.permutation(n):
            a = self.subjects[int(i)]
            if a.is_single:
                a.move()
        self.t += 1

    # -- statistics ----------------------------------------------------------

    def agent_quality(self, agent: Agent) -> float:
        if agent.is_single:
            return 0.0
        return self.compatibility(agent.id, agent.partner)

    def relationship_quality(self) -> float:
        if not self.subjects:
            return 0.0
        return float(np.mean([self.agent_quality(a) for a in self.subjects]))

    def snapshot(self) -> dict:
        matched = sum(not a.is_single for a in self.subjects)
        n = len(self.subjects)
        return {
            "t": self.t,
            "matched": matched,
            "single": n - matched,
            "couples": matched // 2,
            "mean_quality": self.relationship_quality(),
            "mean_utility": float(np.mean([a.utility for a in self.subjects])) if n else 0.0,
        }

    def strategy_stats(self) -> dict[int, dict]:
        out = {}
        for sid, params in self.strategies.items():
            members = [a for a in self.subjects if a.strategy_id == sid]
            if not members:
                continue
            out[sid] = {
                "label": params["label"],
                "n": len(members),
                "matched_fraction": float(np.mean([not a.is_single for a in members])),
                "mean_quality": float(np.mean([self.agent_quality(a) for a in members])),
                "mean_utility": float(np.mean([a.utility for a in members])),
            }
        return out

    def _record(self) -> None:
        self.history.append(self.snapshot())
        for sid, stats in self.strategy_stats().items():
            self.strategy_history[sid].append({"t": self.t, **stats})

    def run(self, n_steps: int) -> list[dict]:
        if not self.history:
            self._record()
        for _ in range(n_steps):
            self.step()
            self._record()
        return self.history

class Agent:
    def __init__(
        self,
        model: "DatingMarket",
        agent_id: int,
        pos: tuple[int, int],
        is_male: bool,
        strategy_id: int,
        move_prob: float,
        rejection_cost: float,
        rationality: float,
        relation_threshold: float,
        memory_depth: int,
        agent_rational: str,
        decay_rate_rej: float
    ):
        self.model = model
        self.id = agent_id
        self.pos = pos
        self.is_male = is_male
        self.strategy_id = strategy_id

        self.move_prob = move_prob
        self.rejection_cost = rejection_cost      # `a`: rejection cost / standards
        self.rationality = rationality            # logit inverse temperature beta
        self.relation_threshold = relation_threshold
        self.memory_depth = memory_depth
        self.agent_rational = agent_rational

        self.partner: int | None = None
        self.engaged_until = 0                     # committed while t < engaged_until
        self.utility = 0.0

        self._buf: dict[int, np.ndarray] = {}
        self._cnt: dict[int, int] = {}
        self.decay_rate_rej = decay_rate_rej

    @property
    def is_single(self) -> bool:
        return self.model.t >= self.engaged_until

    # -- observation memory -------------------------------------------------

    def observe(self, other_id: int, value: float) -> None:
        if other_id not in self._buf:
            self._buf[other_id] = np.zeros(self.memory_depth)
            self._cnt[other_id] = 0
        slot = self._cnt[other_id] % self.memory_depth
        self._buf[other_id][slot] = value
        self._cnt[other_id] += 1

    def samples(self, other_id: int) -> np.ndarray:
        n = min(self._cnt[other_id], self.memory_depth)
        return self._buf[other_id][:n]

    def observe_surroundings(self) -> None:
        """Engaged agents keep sensing the environment, but take no action."""
        current_agent_pos = self.model.subjects[self.id].pos
        neigbours = numba_neighbours(self.model.EMPTY, self.model.grid,
                                     self.model.n_grid, self.model.interaction_radius,
                                     current_agent_pos)
        # for other_id in self.model.neighbours(self.id):
        for other_id in neigbours:
            if self.model.subjects[other_id].is_male != self.is_male:
                self.observe(other_id, self.model.sample_compatibility(self.id, other_id))

    # -- expected utility ---------------------------------------------------

    def _expected_utility(self, other_id: int) -> float | None:
        """EU of being matched with other_id, or None if too few samples."""
        a = self.model.rej_cost_matrix_list[self.id]
        if other_id not in self._cnt:
            return None
        s = self.samples(other_id)
        if len(s) < 2:
            return None

        #Mean field strategy
        if self.agent_rational == "mean":
            return 2*a if np.mean(s) > self.relation_threshold else -a

        # res = ttest_1samp(s, self.relation_threshold, alternative="greater")
        _, pvalue = t_test(s, self.relation_threshold)
        if not np.isfinite(pvalue):
            return None
        p = 1.0 - pvalue
        return p - (1 - p) * a

    # -- turn (single agents only) ------------------------------------------

    def act(self) -> None:
        self.partner = None                        # drop any finished partnership
        target = self._choose_target()
        if target is None:
            return                                 # utility += 0

        other = self.model.subjects[target]
        if other.consider_proposal(self.id):
            until = self.model.t + self.model.relationship_length + self.model.rng.integers(-2, 3)
            self.partner = target
            other.partner = self.id
            self.engaged_until = until
            other.engaged_until = until
            self.utility += 1.0                    # reciprocated
        else:
            self.utility -= self.rejection_cost    # rejected

    def _choose_target(self) -> int | None:
        target_ids: list[int | None] = []
        utilities: list[float] = []
        current_agent_pos = self.model.subjects[self.id].pos
        neigbours = numba_neighbours(self.model.EMPTY, self.model.grid,
                                     self.model.n_grid, self.model.interaction_radius,
                                     current_agent_pos)

        # for other_id in self.model.neighbours(self.id):
        for other_id in neigbours:
            other = self.model.subjects[other_id]
            if other.is_male == self.is_male:
                continue

            # interact with every opposite-gender neighbour
            self.observe(other_id, self.model.sample_compatibility(self.id, other_id))

            # may only propose to single agents
            if not other.is_single:
                continue

            eu = self._expected_utility(other_id)
            if eu is None:
                continue
            target_ids.append(other_id)
            utilities.append(eu)

        target_ids.append(None)
        utilities.append(0.0)

        z = self.rationality * np.asarray(utilities)
        z -= z.max()
        weights = np.exp(z)
        probs = weights / weights.sum()
        choice = self.model.rng.choice(len(target_ids), p=probs)
        t = target_ids[choice]
        return None if t is None else int(t)

    def consider_proposal(self, proposer_id: int) -> bool:
        """Accept with probability sigmoid(beta * EU); accept -> +1, reject -> 0."""
        if not self.is_single:
            return False                           # engaged agents do not respond
        self.observe(proposer_id, self.model.sample_compatibility(self.id, proposer_id))
        eu = self._expected_utility(proposer_id)
        if eu is None:
            return False
        p_accept = 1.0 / (1.0 + np.exp(-self.rationality * eu))
        if self.model.rng.random() < p_accept:
            self.utility += 1.0
            self.model.rej_cost_matrix_list[proposer_id] = self.model.subjects[proposer_id].rejection_cost
            self.model.rej_cost_matrix_list[self.id] = self.model.subjects[self.id].rejection_cost
            return True
        else:
            self.model.rej_cost_matrix_list[proposer_id] *= self.decay_rate_rej
            # print(self.model.rej_cost_matrix_list[self.strategy_id][proposer_id % (self.strategy_id+1)])
        return False

    def move(self) -> None:
        if self.model.rng.random() > self.move_prob:
            return
        options = self.model.movement_options(self.id)
        if options:
            new_pos = options[self.model.rng.integers(len(options))]
            self.model.grid[self.pos] = DatingMarket.EMPTY
            self.model.grid[new_pos] = self.id
            self.pos = new_pos




if __name__ == "__main__":
    market = DatingMarket(n_grid=40, interaction_radius=4, interaction_std=0.5,
                          relationship_length=10, seed=0)
    market.add_agents(100, rejection_cost=1.5, rationality=6.0, label="cautious")#, agent_rational="mean")
    market.add_agents(100, rejection_cost=0.2, rationality=6.0, label="bold")
    market.add_agents(100, rejection_cost=0.2, rationality=.0, label="random")#, agent_rational="mean")
    market.run(120)

    print("Final per-strategy statistics:")
    for sid, s in market.strategy_stats().items():
        print(
            f"  [{s['label']:>8}] n={s['n']:>3}  "
            f"matched={s['matched_fraction']:.2f}  "
            f"quality={s['mean_quality']:.3f}  "
            f"cum.utility={s['mean_utility']:.1f}"
        )
    print(market.rej_cost_matrix_list[0:100])
    print(market.rej_cost_matrix_list[100:200])
    print(market.rej_cost_matrix_list[200:300])
