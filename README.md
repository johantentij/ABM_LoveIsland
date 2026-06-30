# ABM LoveIsland: Agent-Based Model of a Dating Market

This repository contains the source code for **ABM LoveIsland**, an agent-based model of a heterosexual dating market where agents search for partners under incomplete information and strategic uncertainty. The simulation evaluates how different behavioral strategies—specifically cautious versus bold approaches to proposing—affect the quality of relationships and overall market stability.

## Model Overview
The simulation is built on a few core mechanisms:
* **Environment & Population:** The default population consists of 200 agents (gender-balanced) distributed across a 40x40 toroidal grid.
* **Latent Compatibility:** Every possible male-female pair has a fixed, latent compatibility score drawn at initialization. Agents cannot observe this directly; instead, they receive noisy signals through interactions and store them in a memory buffer.
* **Behavioral Parameters:** Agents are driven by two main parameters: a **rejection cost** (how aversive an unsuccessful proposal is) and a **rationality parameter** (how deterministically agents act on their expected utilities).
* **Decision Making:** Agents use a one-sided, one-sample t-test to analyze their interaction memory and estimate the probability that a candidate's compatibility exceeds their personal relationship threshold.

## Repository Structure
The repository is split into three main scripts corresponding to the core experiments and extensions detailed in report.pdf:

* **`strategyCompBlocking.py`**
  This script explores the effects of strategic market composition. It systematically varies the fraction of "cautious" agents (who bear a high rejection cost) and "bold" agents (who bear a low rejection cost) to measure mean relationship quality and the fraction of blocking pairs.
  For the temporary relationship scenario, set **`relationship_length=10`** at line 17.
  
* **`sensitivity_5param_parallel.ipynb`**
  This notebooks contains the sensitivity analysis, which identifies how parameters drive outcome variance within a homogeneous population. It executes a global variance decomposition using Saltelli-sampled Sobol indices across five parameters (rejection cost, rationality, interaction radius, interaction noise, and population size) and includes a one-factor-at-a-time (OFAT) sweep for visual inspection.

* **`decay_scan.py`**
  This script contains the additional robustness check that replaces the baseline fixed threshold with a dynamic one. It models a decaying threshold that depends on the duration an agent has remained single, testing if early concessions create a collective behavioral cascade.

## Dependencies:
To run all the code in the current files the following python modules are necessary: numpy, matplotlib, scipy, pandas, numba, joblib, multiprocessing

## Notes:
Because of time-pressure the authors used slightly different versions of the model that are not interchangable. The dependencies are as such:<br><br>
**`strategyCompBlocking.py`** depends on **`dating_market_blocking.py`** <br><br>
**`sensitivity_5param_parallel.ipynb`** depends on **`matching_metrics.py`** depends on **`dating_market_Alex.py`** <br><br>
**`decay_scan.py`** depends on **`dating_market.py`** <br><br>
The **`old_experiments`** folder contains numerous investigations of different setting of the model for the curious reader
## Data & Licensing
* **Data:** This model relies entirely on simulated dynamics; no empirical data has been used.
* **License:** The code in this repository is distributed under the MIT license.
