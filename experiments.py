from Agents.NPC import SWA, wSWA
from Agents.PC import CTO
from Environments.RottenBandits import NonParametricRottenBandit, ParametricRottenBandit
from utils import run_experiment
import numpy as np


np.random.seed(5)
T = 30000
experiment_repeats = 100

experiments = [(ParametricRottenBandit(), wSWA, 'P-AV-wSWA'),
               (ParametricRottenBandit(), SWA, 'P-AV-SWA')]

for experiment in experiments:
    run_experiment(*experiment, T = T, experiment_repeats = experiment_repeats)

