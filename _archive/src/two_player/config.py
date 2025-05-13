
from game_env import AuctionEnv
import numpy as np
import torch
import os


GAME_CONFIG = {
    "action_space_start" : 0,
    "action_space_stop" : 1,
    "forwards" : [150, 120, 100, 90, 85, 80, 75, 70, 62, 61, 60, 45],
    "defensemen" : [90, 80, 75, 70, 68, 66, 50, 45],
    "goalies" : [80, 60, 40, 25],
    "num_players" : 2
}

def gen_env():
    return AuctionEnv(forwards=GAME_CONFIG["forwards"], defensemen=GAME_CONFIG["defensemen"], goalies=GAME_CONFIG["goalies"], num_players=GAME_CONFIG["num_players"])


SIM_CONFIG = {
    "random_games" : 0.05, #probability of the game allowing for opponent randomness
    "prob_random" : 0.1, #in games with random behaviour, probability of the opponent going off the rails
    "greedy" : True,
    "sims_per_iter" : 100,
    "actions_per_step" : 10
}

FUNCTION_CONFIG = {
    "n_discrete" : 102, #split [0,1] action space into n_dsicrete actions
    "obs_dim" : gen_env().OBS_DIM,
    "device" : torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

}

TRAIN_CONFIG = {
    "policy_reservoir_size" : 25_000, #keep a 100000 training samples always
    "regret_reservoir_size" : 1_000_000,  #keep a 100000 training samples always
    "batch_size" : 2048,
    "lr" : 1e-3,
    "iterations" : 10000
}


DISCRETE_ACTION_SPACE = np.linspace(GAME_CONFIG["action_space_start"],GAME_CONFIG["action_space_stop"],FUNCTION_CONFIG["n_discrete"])


EXAMPLES_PATH = 'latest_examples'
POLICY_RESERVOIR_PATH = f'{EXAMPLES_PATH}/policy_reservoir.pkl'
REGRET_RESERVOIR_PATH = f'{EXAMPLES_PATH}/regret_reservoir.pkl'
LOGS_PATH = "game_logs"


if not os.path.exists(EXAMPLES_PATH):
    os.mkdir(EXAMPLES_PATH)
if not os.path.exists(LOGS_PATH):
    os.mkdir(LOGS_PATH)
