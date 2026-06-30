import cProfile
from dating_market import DatingMarket

market = DatingMarket(n_grid=40, interaction_radius=4, interaction_std=0.5,
                          relationship_length=10, seed=0)
market.add_agents(100, rejection_cost=1.5, rationality=6.0, label="cautious") #, agent_rational="mean")
market.add_agents(100, rejection_cost=0.2, rationality=6.0, label="bold", agent_rational="mean")
market.add_agents(100, rejection_cost=0.2, rationality=.0, label="random") #, agent_rational="mean")

# Profile to file
cProfile.run('market.run(120)', 'profile_results.prof')