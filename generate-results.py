from Agents.NPC import SWA, wSWA
from Agents.PC import CTO
from Agents.Naive import Random, RoundRobin
from Environments.RottenBandits import NonParametricRottenBandit, ParametricRottenBandit
from utils import run_experiment
import numpy as np

seed = 0
T = 30000
experiment_repeats = 100

#%% Naive agents < ~1m each experiment
np.random.seed(seed)
run_experiment(NonParametricRottenBandit(), Random, 'NP-Random', T, experiment_repeats)
run_experiment(NonParametricRottenBandit(), RoundRobin, 'NP-RR', T, experiment_repeats)

run_experiment(ParametricRottenBandit(seed = seed), Random, 'P-AV-Random', T, experiment_repeats)
run_experiment(ParametricRottenBandit(seed = seed), RoundRobin, 'P-AV-RR', T, experiment_repeats)

run_experiment(ParametricRottenBandit(ANV = True, seed = seed), Random, 'P-ANV-Random', T, experiment_repeats)
run_experiment(ParametricRottenBandit(ANV = True, seed = seed), RoundRobin, 'P-ANV-RR', T, experiment_repeats)

#%% SWA and wSWA ~ 1-2m each experiment
np.random.seed(seed)
run_experiment(NonParametricRottenBandit(), SWA, 'NP-SWA', T, experiment_repeats)
run_experiment(NonParametricRottenBandit(), wSWA, 'NP-wSWA', T, experiment_repeats)

run_experiment(ParametricRottenBandit(seed = seed), SWA, 'P-AV-SWA', T, experiment_repeats)
run_experiment(ParametricRottenBandit(seed = seed), wSWA, 'P-AV-wSWA', T, experiment_repeats)

run_experiment(ParametricRottenBandit(ANV = True, seed = seed), SWA, 'P-ANV-SWA', T, experiment_repeats)
run_experiment(ParametricRottenBandit(ANV = True, seed = seed), wSWA, 'P-ANV-wSWA', T, experiment_repeats)

#%% CTO - this takes a very long time (~4 hours on macbook with i7)
np.random.seed(seed)
run_experiment(ParametricRottenBandit(), CTO, 'CTO', T, experiment_repeats)


