"""
Matplotlib animation of the dating-market model.

Left panel: the grid over time, colour-coded single male / single female /
matched (matched is gender-agnostic).

Right panels: per-strategy time series updating live -- matched fraction (top)
and mean relationship quality (bottom), one line per strategy group.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D

from dating_market import DatingMarket

MALE = "#3b82f6"     # blue
FEMALE = "#ec4899"   # pink
MATCHED = "#22c55e"  # green


def _split_positions(market: DatingMarket):
    males, females, matched = [], [], []
    for agent in market.subjects:
        point = (agent.pos[1], agent.pos[0])  # x = col, y = row
        if not agent.is_single:
            matched.append(point)
        elif agent.is_male:
            males.append(point)
        else:
            females.append(point)
    return males, females, matched


def animate(market: DatingMarket, n_steps: int = 150, interval: int = 50):
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 2, figure=fig, width_ratios=[1.3, 1], wspace=0.25, hspace=0.35)
    ax_grid = fig.add_subplot(gs[:, 0])
    ax_util = fig.add_subplot(gs[0, 1])
    ax_qual = fig.add_subplot(gs[1, 1])

    # --- grid panel ---
    ax_grid.set_xlim(-1, market.n_grid)
    ax_grid.set_ylim(-1, market.n_grid)
    ax_grid.set_aspect("equal")
    ax_grid.invert_yaxis()
    ax_grid.set_xticks([])
    ax_grid.set_yticks([])

    size = max(8, 4000 / market.n_grid)
    male_scat = ax_grid.scatter([], [], s=size, c=MALE, edgecolors="none")
    female_scat = ax_grid.scatter([], [], s=size, c=FEMALE, edgecolors="none")
    matched_scat = ax_grid.scatter([], [], s=size, c=MATCHED, edgecolors="none")
    ax_grid.legend(
        handles=[
            Line2D([], [], marker="o", ls="", mfc=MALE, mec="none", ms=9, label="single male"),
            Line2D([], [], marker="o", ls="", mfc=FEMALE, mec="none", ms=9, label="single female"),
            Line2D([], [], marker="o", ls="", mfc=MATCHED, mec="none", ms=9, label="matched"),
        ],
        loc="upper right", framealpha=0.9, fontsize=9,
    )

    # --- comparison panels: one line per strategy ---
    cmap = plt.get_cmap("tab10")
    strategy_ids = list(market.strategies.keys())
    colors = {sid: cmap(i % 10) for i, sid in enumerate(strategy_ids)}
    series = {sid: {"t": [], "util_cum": [], "quality": []} for sid in strategy_ids}

    util_lines, qual_lines = {}, {}
    for sid in strategy_ids:
        label = market.strategies[sid]["label"]
        (util_lines[sid],) = ax_util.plot([], [], color=colors[sid], lw=2, label=label)
        (qual_lines[sid],) = ax_qual.plot([], [], color=colors[sid], lw=2, label=label)

    ax_util.set_xlim(0, n_steps)
    ax_util.set_ylabel("avg utility / step")
    ax_util.legend(fontsize=8, loc="upper right")
    ax_util.grid(alpha=0.25)
    ax_util.axhline(0, color="0.6", lw=0.8)

    ax_qual.set_xlim(0, n_steps)
    ax_qual.set_ylim(0, 1)
    ax_qual.set_xlabel("step")
    ax_qual.set_ylabel("mean quality")
    ax_qual.grid(alpha=0.25)

    def draw_grid():
        males, females, matched = _split_positions(market)
        male_scat.set_offsets(np.array(males).reshape(-1, 2))
        female_scat.set_offsets(np.array(females).reshape(-1, 2))
        matched_scat.set_offsets(np.array(matched).reshape(-1, 2))
        s = market.snapshot()
        ax_grid.set_title(
            f"step {s['t']}   single {s['single']}   "
            f"matched {s['matched']}   couples {s['couples']}",
            fontsize=11,
        )

    window = market.relationship_length  # smoothing window for the slope

    def update(_frame):
        market.step()
        draw_grid()
        for sid, st in market.strategy_stats().items():
            s = series[sid]
            s["t"].append(market.t)
            s["util_cum"].append(st["mean_utility"])
            s["quality"].append(st["mean_quality"])
            qual_lines[sid].set_data(s["t"], s["quality"])

            # utility slope = per-step increment, trailing mean over `window`
            cum = np.asarray(s["util_cum"])
            if len(cum) >= 2:
                inc = np.diff(cum)
                rate = np.array([inc[max(0, k - window + 1): k + 1].mean()
                                 for k in range(len(inc))])
                util_lines[sid].set_data(s["t"][1:], rate)
        ax_util.relim()
        ax_util.autoscale_view(scalex=False)
        return (male_scat, female_scat, matched_scat)

    draw_grid()
    anim = FuncAnimation(fig, update, frames=n_steps, interval=interval, blit=False)
    return fig, anim


if __name__ == "__main__":
    market = DatingMarket(n_grid=60, interaction_radius=4, interaction_std=0.5,
                          relationship_length=10, seed=0)
    market.add_agents(100, rejection_cost=5, rationality=100.0, label="cautious", gender_balance=.75)
    market.add_agents(100, rejection_cost=5, rationality=100.0, label="bold", gender_balance=.75)

    fig, anim = animate(market, n_steps=300)

    plt.show()  # for a live window with a GUI backend
    # anim.save("dating_market.gif", writer="pillow", fps=12, dpi=90)
    # print("saved dating_market.gif")