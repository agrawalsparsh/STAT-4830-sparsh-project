# Learning to Bid: Deep Counterfactual Regret Minimization in Fantasy Hockey Auctions

**Team Members:** Sparsh Agrawal

## High-Level Summary

This project implements an AI agent for participating in a fantasy sports (Hockey) auction draft. The goal is to draft the best possible team within a fixed budget by strategically bidding on players (forwards, defensemen, goalies) with varying projected values.

The AI is trained using techniques inspired by Deep Counterfactual Regret Minimization (Deep CFR). It utilizes two neural networks:
1.  **Policy Network:** Learns the average strategy over iterations, predicting the probability distribution over possible bids given the current game state.
2.  **Regret Network:** Ill named throughout this code base, this network learns the expected value of taking a specific bid action in a given state.

During training, the agent simulates numerous auction drafts, exploring different bidding strategies and learning from the outcomes to minimize regret and improve its policy. The final agent aims to make near-optimal bidding decisions based on its budget, roster needs, opponent information, and the player currently up for auction.

## Repository Structure Overview

```
.
├── _archive/             # Archived intermediate work (drafts, logs, old code)
├── docs/                 # Final documentation (presentation slides, report)
│   ├── final_presentation.pdf
│   └── report.pdf        # Final report
├── policy_network_weights/ # Saved weights for the trained policy network
├── src/                  # Final, cleaned source code
│   └── final_game/
│       ├── approximators.py    # Neural network definitions (Policy, Regret)
│       ├── config.py           # Configuration for game, training, simulation
│       ├── game_env.py         # Auction draft game environment (AuctionEnv)
│       ├── gen_simulations.py  # Simulation generation (Deep CFR logic)
│       ├── main.py             # Main script for running training iterations
│       ├── play.py             # Script to play against the trained AI
│       ├── reservoir_sampling.py # Reservoir sampling classes for training data
│       ├── train_funcs.py      # Training loop, network training, play logic
│       └── ...                 # Other support files/dirs
├── .gitignore            # Specifies intentionally untracked files
├── README.md             # This file: Project overview and instructions
└── requirements.txt      # Project dependencies
```

The main directories (`docs/`, `src/`, `notebooks/`) contain the final, polished components of the project. All development history, previous drafts, and exploration logs are stored in the `_archive/` directory to keep the main structure clean while preserving the process.

## Setup Instructions

1.  **Clone the Repository:**
    ```bash
    git clone git@github.com:agrawalsparsh/STAT-4830-sparsh-project.git
    cd STAT-4830-sparsh-project
    ```

2.  **Create a Virtual Environment (Recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```

3.  **Install Dependencies:** Ensure you have Python 3.x installed. The specific version used during development was [Specify Python version if known, e.g., 3.10]. Install the required packages:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This project uses PyTorch. The `requirements.txt` should specify the correct version for your system (CPU or GPU - the code attempts to use MPS/GPU if available, otherwise CPU). Refer to the [official PyTorch installation guide](https://pytorch.org/get-started/locally/) if you encounter issues.*
    *The game display uses the `rich` library, which is included in `requirements.txt`.*

4.  **Pre-trained Model:** The repository includes pre-trained weights for the policy network in the `policy_network_weights/` directory (`policy_network.pth`). The `play.py` script will automatically load these.

## Running the Code

There are two main ways to run the code: training the agent or playing against a pre-trained agent.

### 1. Playing Against the AI

To play an interactive game against the trained AI:

```bash
cd src/final_game
python play.py
```

This will load the pre-trained policy network weights (from `policy_network_weights/policy_network.pth`) and start an auction draft. You will be prompted to enter your bids for each player. The AI agent(s) will bid based on their learned policy. The game state and final results will be displayed in the terminal.

### 2. Training the Agent (Optional)

Training requires significant computational resources and time. The main training loop is controlled by `main.py`.

**Prerequisites:**
*   Training generates simulation data stored as pickled reservoir samples (e.g., `policy_reservoir.pkl`, `regret_reservoir.pkl`). These will be saved in the `src/final_game/latest_examples/` directory. This directory will be created automatically by the script if it does not exist.
*   Training logs and intermediate game states might be saved in `src/final_game/game_logs/`. This directory will also be created automatically if it doesn't exist.

**Modifying Game Structure (Optional):**

Before starting training, you can customize the auction parameters by editing the `GAME_CONFIG` dictionary in `src/final_game/config.py`:
*   `forwards`, `defensemen`, `goalies`: Modify these lists to change the pool of available players and their associated values for each position.
*   `num_players`: Adjust the number of agents participating in the auction draft.

*Note: Changing `num_players` will also affect the observation dimension (`obs_dim`), which is calculated dynamically based on `num_players`. The network architectures in `approximators.py` are designed relative to `obs_dim`, so they should adapt automatically. However, significant changes to the game complexity (e.g., vastly different player values or number of players) might require tuning hyperparameters in `SIM_CONFIG`, `FUNCTION_CONFIG`, or `TRAIN_CONFIG`, or potentially adjusting the network architectures themselves.*

**To run training:**

```bash
cd src/final_game
python main.py
```

This script runs training iterations as defined in `train_funcs.py` and `config.py`. It will:
1.  Load existing training data reservoirs (`policy_reservoir.pkl`, `regret_reservoir.pkl`) from `latest_examples/` if they exist.
2.  Train the Policy and Regret networks based on the data in the reservoirs.
3.  Run new simulations using the updated networks (`gen_fully_batched_simulations`) to generate fresh training data.
4.  Add the new data to the reservoirs.
5.  Save the updated reservoirs back to `latest_examples/`.
6.  Periodically save mock game logs to `game_logs/`.

After training (or interrupting it), the latest trained policy network can be used by running `play.py`, which will also save the weights to `policy_network_weights/policy_network.pth` if they weren't already loaded.

## Executable Demo

The primary executable demonstration for this project is playing interactively against the trained AI agent using the `play.py` script.

A Google Colab notebook was not provided as the core demonstration involves interactive terminal input for bidding, which is best suited for local execution.

Please refer to the **Running the Code -> Playing Against the AI** section above for detailed instructions on how to set up the environment and run the interactive demo.






