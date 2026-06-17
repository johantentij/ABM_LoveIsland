"""
Matplotlib animation of the dating-market model.

Renders the grid over time with three colour codings:
    - single male
    - single female
    - engaged (either gender -> one colour)

Run directly to preview; call `animate(...)` to embed or save elsewhere.
"""

from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.lines import Line2D

from dating_market import DatingMarket

# colours
MALE = "#3b82f6"     # blue
FEMALE = "#ec4899"   # pink
ENGAGED = "#22c55e"  # green


def _split_positions(market: DatingMarket):
    """Return (male_xy, female_xy, engaged_xy) as lists of (x, y) points."""
    males, females, engaged = [], [], []
    for agent in market.subjects:
        r, c = agent.pos
        point = (c, r)  # x = column, y = row
        if not agent.is_single:
            engaged.append(point)
        elif agent.is_male:
            males.append(point)
        else:
            females.append(point)
    return males, females, engaged


def animate(market: DatingMarket, n_steps: int = 120, interval: int = 500):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-1, market.n_grid)
    ax.set_ylim(-1, market.n_grid)
    ax.set_aspect("equal")
    ax.invert_yaxis()  # row 0 at the top, like a grid
    ax.set_xticks([])
    ax.set_yticks([])

    size = max(8, 4000 / market.n_grid)
    male_scat = ax.scatter([], [], s=size, c=MALE, edgecolors="none")
    female_scat = ax.scatter([], [], s=size, c=FEMALE, edgecolors="none")
    engaged_scat = ax.scatter([], [], s=size, c=ENGAGED, edgecolors="none")

    legend_handles = [
        Line2D([], [], marker="o", linestyle="", markerfacecolor=MALE,
               markeredgecolor="none", markersize=9, label="single male"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=FEMALE,
               markeredgecolor="none", markersize=9, label="single female"),
        Line2D([], [], marker="o", linestyle="", markerfacecolor=ENGAGED,
               markeredgecolor="none", markersize=9, label="engaged"),
    ]
    ax.legend(handles=legend_handles, loc="upper right",
              framealpha=0.9, fontsize=9)

    def draw():
        males, females, engaged = _split_positions(market)
        male_scat.set_offsets(np.array(males).reshape(-1, 2))
        female_scat.set_offsets(np.array(females).reshape(-1, 2))
        engaged_scat.set_offsets(np.array(engaged).reshape(-1, 2))
        s = market.stats()
        ax.set_title(
            f"step {s['t']}   single {s['single']}   "
            f"engaged {s['engaged']}   couples {s['couples']}",
            fontsize=11,
        )

    def update(_frame):
        market.step()
        draw()
        return male_scat, female_scat, engaged_scat

    draw()  # initial frame at t = 0
    anim = FuncAnimation(fig, update, frames=n_steps,
                         interval=interval, blit=False)
    return fig, anim


if __name__ == "__main__":
    market = DatingMarket(
        n_grid=40,
        n_subjects=200,
        move_prob=0.6,
        risk_aversion=0.8,
        interaction_radius=4,
        seed=0,
    )
    fig, anim = animate(market, n_steps=120)

    # Live window if you have a GUI backend:
    plt.show()

    # Otherwise save to a file:
    # anim.save("dating_market.gif", writer="pillow", fps=12, dpi=90)
    # print("saved dating_market.gif")