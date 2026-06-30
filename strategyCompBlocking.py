import numpy as np
import matplotlib.pyplot as plt
import warnings
from joblib import Parallel, delayed
from dating_market_blocking import DatingMarket

def _run_single_sim(n_cautious, n_bold, n_steps, grid_size, seed):
    """
    Helper function executing a single simulation run.
    Extracted to allow easy parallelization via joblib.
    """
    market = DatingMarket(
        n_grid=grid_size, 
        interaction_radius=4, 
        interaction_std=0.5, 
        seed=seed,
        relationship_length=1000
    )
    
    if n_cautious > 0:
        market.add_agents(
            n_cautious, 
            rejection_cost=10.0, 
            rationality=30.0, 
            label="Cautious Rational",
            decay_rate_rej=1,
            memory_depth=8
        )
        
    if n_bold > 0:
        market.add_agents(
            n_bold, 
            rejection_cost=.1, 
            rationality=30.0, 
            label="Bold Rational",
            decay_rate_rej=1,
            memory_depth=8
        )
        
    market.run(n_steps)
    stats = market.strategy_stats()
    block_stats = market.blocking_pair_stats()
    
    q_c, q_b = np.nan, np.nan
    b_c, b_b = np.nan, np.nan  # fraction of each strategy's agents in >=1 blocking pair
    
    for sid, s in stats.items():
        if s["label"] == "Cautious Rational":
            q_c = s["mean_matched_quality"]
        elif s["label"] == "Bold Rational":
            q_b = s["mean_matched_quality"]

    for sid, s in block_stats["by_strategy"].items():
        if s["label"] == "Cautious Rational":
            b_c = s["frac_agents_in_blocking_pair"]
        elif s["label"] == "Bold Rational":
            b_b = s["frac_agents_in_blocking_pair"]

    overall_block_frac = block_stats["fraction_blocking_pairs"]

    return q_c, q_b, b_c, b_b, overall_block_frac


def run_extreme_sweep(
    total_agents: int = 200, 
    n_steps: int = 200, 
    grid_size: int = 40,
    repeats: int = 100,
    n_jobs: int = -1
):
    """
    Focuses the experimental sweep on the extreme margins (e.g. 1/199 ratios)
    to observe minority strategy effects. Runs stochastic variations in parallel.
    
    Args:
        n_jobs (int): Number of CPU cores to use. -1 means use all available cores.
    """
    
    # Custom array of cautious agent counts focusing on the extreme margins
    cautious_counts = [
        0, 1, 2, 3, 4, 5, 10,  # Extreme Bold dominance
        20, 40, 60, 100, 140, 180, # Middle bulk
        190, 195, 196, 197, 198, 199, 200 # Extreme Cautious dominance
    ]
    
    cautious_fractions = np.array(cautious_counts) / total_agents
    
    q_cautious_mean, q_cautious_std = [], []
    q_bold_mean, q_bold_std = [], []
    b_cautious_mean, b_cautious_std = [], []
    b_bold_mean, b_bold_std = [], []
    b_overall_mean, b_overall_std = [], []
    
    print(f"Running extreme sweep with {total_agents} total agents over {n_steps} steps...")
    print(f"Averaging over {repeats} stochastic runs per parameter set using {n_jobs if n_jobs > 0 else 'all'} CPU cores.")
    
    for n_cautious in cautious_counts:
        n_bold = total_agents - n_cautious
        
        # Pre-generate unique seeds for this batch of runs to avoid multiprocessing RNG collision
        run_seeds = np.random.randint(0, 1000000, size=repeats)
        
        # Execute the inner loop in parallel across your CPU cores
        results = Parallel(n_jobs=n_jobs)(
            delayed(_run_single_sim)(n_cautious, n_bold, n_steps, grid_size, seed)
            for seed in run_seeds
        )
        
        # Unpack the list of tuples returned by Parallel
        run_q_c, run_q_b, run_b_c, run_b_b, run_b_overall = zip(*results)
            
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            mean_c = np.nanmean(run_q_c)
            std_c = np.nanstd(run_q_c)
            mean_b = np.nanmean(run_q_b)
            std_b = np.nanstd(run_q_b)

            mean_bc = np.nanmean(run_b_c)
            std_bc = np.nanstd(run_b_c)
            mean_bb = np.nanmean(run_b_b)
            std_bb = np.nanstd(run_b_b)
            mean_b_overall = np.nanmean(run_b_overall)
            std_b_overall = np.nanstd(run_b_overall)
            
        q_cautious_mean.append(mean_c)
        q_cautious_std.append(std_c)
        q_bold_mean.append(mean_b)
        q_bold_std.append(std_b)

        b_cautious_mean.append(mean_bc)
        b_cautious_std.append(std_bc)
        b_bold_mean.append(mean_bb)
        b_bold_std.append(std_bb)
        b_overall_mean.append(mean_b_overall)
        b_overall_std.append(std_b_overall)
        
        mc_str = f"{mean_c:.3f}" if not np.isnan(mean_c) else "  NaN"
        mb_str = f"{mean_b:.3f}" if not np.isnan(mean_b) else "  NaN"
        bc_str = f"{mean_bc:.3f}" if not np.isnan(mean_bc) else "  NaN"
        bb_str = f"{mean_bb:.3f}" if not np.isnan(mean_bb) else "  NaN"
        
        print(f"  Cautious: {n_cautious:>3} | Bold: {n_bold:>3} --> "
              f"Q_Cautious: {mc_str} | Q_Bold: {mb_str} | "
              f"Block_Cautious: {bc_str} | Block_Bold: {bb_str} | "
              f"Block_Overall: {mean_b_overall:.4f}")

    plot_results(
        cautious_fractions, 
        np.array(q_cautious_mean), 
        np.array(q_bold_mean),
        np.array(q_cautious_std),
        np.array(q_bold_std)
    )

    plot_stability(
        cautious_fractions,
        np.array(b_cautious_mean),
        np.array(b_bold_mean),
        np.array(b_cautious_std),
        np.array(b_bold_std),
        np.array(b_overall_mean),
        np.array(b_overall_std),
    )

def plot_results(fractions, q_c_mean, q_b_mean, q_c_std, q_b_std):
    # Create a figure with a main plot and two zoomed-in subplots for the margins
    fig = plt.figure(figsize=(12, 5))
    
    # 1. Left Margin Zoom (0% to 10%)
    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(fractions, q_c_mean, marker='o', color='blue')
    ax1.plot(fractions, q_b_mean, marker='s', color='red')
    ax1.fill_between(fractions, q_c_mean - q_c_std, q_c_mean + q_c_std, color='blue', alpha=0.15)
    ax1.fill_between(fractions, q_b_mean - q_b_std, q_b_mean + q_b_std, color='red', alpha=0.15)
    ax1.set_xlim(-0.01, 0.1)
    ax1.set_title('Extreme Bold Dominance')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylabel('Mean Quality')
    
    # 2. Main Overall Plot
    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(fractions, q_c_mean, marker='o', color='blue', label='Cautious Rational')
    ax2.plot(fractions, q_b_mean, marker='s', color='red', label='Bold Rational')
    ax2.fill_between(fractions, q_c_mean - q_c_std, q_c_mean + q_c_std, color='blue', alpha=0.15)
    ax2.fill_between(fractions, q_b_mean - q_b_std, q_b_mean + q_b_std, color='red', alpha=0.15)
    ax2.set_xlabel('Fraction of Cautious Agents')
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # 3. Right Margin Zoom (90% to 100%)
    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(fractions, q_c_mean, marker='o', color='blue')
    ax3.plot(fractions, q_b_mean, marker='s', color='red')
    ax3.fill_between(fractions, q_c_mean - q_c_std, q_c_mean + q_c_std, color='blue', alpha=0.15)
    ax3.fill_between(fractions, q_b_mean - q_b_std, q_b_mean + q_b_std, color='red', alpha=0.15)
    ax3.set_xlim(0.9, 1.01)
    ax3.set_title('Extreme Cautious Dominance')
    ax3.grid(True, linestyle='--', alpha=0.7)
    
    # Global legend
    fig.legend(['Cautious Rational', 'Bold Rational'], loc='upper center', ncol=2, fontsize=12)
    
    plt.tight_layout()
    plt.savefig('extreme_margins_sweep.png', dpi=300, bbox_inches='tight')
    plt.show()

def plot_stability(fractions, b_c_mean, b_b_mean, b_c_std, b_b_std, b_overall_mean, b_overall_std):
    """
    Plots the blocking-pair stability metric across the composition sweep.

    b_c_mean / b_b_mean: fraction of Cautious / Bold agents that are part of
        at least one blocking pair (i.e. there exists some currently-available
        agent both they and that agent would rather switch to). Lower = that
        strategy's matches are more individually stable.
    b_overall_mean: fraction of ALL possible male-female pairs that are
        blocking pairs, market-wide. Lower = the matching as a whole is closer
        to a stable matching (0 = fully stable, no incentive to deviate).
    """
    fig = plt.figure(figsize=(12, 5))

    ax1 = plt.subplot(1, 3, 1)
    ax1.plot(fractions, b_c_mean, marker='o', color='blue')
    ax1.plot(fractions, b_b_mean, marker='s', color='red')
    ax1.fill_between(fractions, b_c_mean - b_c_std, b_c_mean + b_c_std, color='blue', alpha=0.15)
    ax1.fill_between(fractions, b_b_mean - b_b_std, b_b_mean + b_b_std, color='red', alpha=0.15)
    ax1.set_xlim(-0.01, 0.1)
    ax1.set_title('Extreme Bold Dominance')
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_ylabel('Frac. agents in a blocking pair')

    ax2 = plt.subplot(1, 3, 2)
    ax2.plot(fractions, b_c_mean, marker='o', color='blue', label='Cautious Rational')
    ax2.plot(fractions, b_b_mean, marker='s', color='red', label='Bold Rational')
    ax2.plot(fractions, b_overall_mean, marker='^', color='green', linestyle='--', label='Market-wide (all pairs)')
    ax2.fill_between(fractions, b_c_mean - b_c_std, b_c_mean + b_c_std, color='blue', alpha=0.15)
    ax2.fill_between(fractions, b_b_mean - b_b_std, b_b_mean + b_b_std, color='red', alpha=0.15)
    ax2.fill_between(fractions, b_overall_mean - b_overall_std, b_overall_mean + b_overall_std, color='green', alpha=0.1)
    ax2.set_xlabel('Fraction of Cautious Agents')
    ax2.grid(True, linestyle='--', alpha=0.7)

    ax3 = plt.subplot(1, 3, 3)
    ax3.plot(fractions, b_c_mean, marker='o', color='blue')
    ax3.plot(fractions, b_b_mean, marker='s', color='red')
    ax3.fill_between(fractions, b_c_mean - b_c_std, b_c_mean + b_c_std, color='blue', alpha=0.15)
    ax3.fill_between(fractions, b_b_mean - b_b_std, b_b_mean + b_b_std, color='red', alpha=0.15)
    ax3.set_xlim(0.9, 1.01)
    ax3.set_title('Extreme Cautious Dominance')
    ax3.grid(True, linestyle='--', alpha=0.7)

    fig.legend(loc='upper center', ncol=3, fontsize=11)

    plt.tight_layout()
    plt.savefig('extreme_margins_stability.png', dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    run_extreme_sweep()