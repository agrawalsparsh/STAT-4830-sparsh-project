import numpy as np
from config import gen_env, DISCRETE_ACTION_SPACE, FUNCTION_CONFIG, LOGS_PATH
import torch
import os
import sys
import pdb

env = gen_env() #game environment
device = FUNCTION_CONFIG["device"]
def predict_counter_factuals(regret_network, observation):
    """
    Predict counterfactual values for all 40 discrete actions for a single observation.
    Args:
        regret_network: The neural network that takes (obs_dim + 1) input features and outputs a scalar.
        observation: A single observation (1D tensor or convertible to one) of shape (obs_dim,).
        device: The torch device to use (e.g., mps or cpu).
    Returns:
        A NumPy array of predicted counterfactual values for each of the 40 actions.
    """
    import torch
    if not isinstance(observation, torch.Tensor):
        observation = torch.tensor(observation, dtype=torch.float32)

    observation = observation.to(device)
    
    # Ensure observation is a 1D tensor.
    if observation.dim() != 1:
        raise ValueError("Expected a single observation (1D tensor).")
    
    n_actions = len(DISCRETE_ACTION_SPACE)
    
    # Repeat observation for each action: (n_actions, obs_dim)
    observation_batch = observation.unsqueeze(0).expand(n_actions, -1)
    
    # Create a tensor for discrete actions and move it to the correct device
    action_batch = torch.tensor(DISCRETE_ACTION_SPACE, dtype=torch.float32, device=device).unsqueeze(1)  # (40, 1)
    
    # Concatenate along the feature dimension -> (40, obs_dim + 1)
    input_batch = torch.cat((observation_batch, action_batch), dim=1)
    
    # Forward pass through the regret network
    predicted_values = regret_network(input_batch)  # Expected output shape: (40,)
    return predicted_values.detach().cpu().numpy()


def predict_policy(policy_network, observation):
    """
    Predict action probabilities from the policy network for a single observation.
    Args:
        policy_network: The network that takes an observation (obs_dim,) and outputs a probability distribution.
        observation: A single observation (1D tensor or convertible to one) of shape (obs_dim,).
        device: The torch device to use (e.g., mps or cpu).
    Returns:
        A NumPy array of predicted action probabilities.
    """
    import torch
    if not isinstance(observation, torch.Tensor):
        observation = torch.tensor(observation, dtype=torch.float32)

    observation = observation.to(device)
    
    # Ensure observation is a 1D tensor.
    if observation.dim() != 1:
        raise ValueError("Expected a single observation (1D tensor).")
    
    # Add batch dimension -> (1, obs_dim)
    observation = observation.unsqueeze(0)
    
    action_probs = torch.exp(policy_network(observation) ) # -> (1, n_discrete)
    
    return action_probs.squeeze(0).detach().cpu().numpy()

def gen_simulation(policy_network, regret_network, n_per_step, p_random, proportion_random):
    ''' 
    Inputs:
      policy_network -> network used to predict the average strategy (trained offline)
      regret_network -> network used to predict the value/regret of taking an action in a given state: (state, action) -> scalar
      iter_epsilon -> probability parameter for choosing a random action in the main branch
      n_per_step -> number of simulation branches to generate at each decision point for player 0
    Outputs:
      regret_samples -> list of [(State, Action, Value, Reach)] samples to train the regret network
      policy_samples -> list of [(State, Probabilities, Reach)] samples to train the policy network
    '''
    regret_samples = []
    policy_samples = []

    if np.random.random() > p_random:
        proportion_random = 0
    
    env.reset()
    
    while not env.game_over:
        actions = {}
        for p in range(env.num_players):
            obs = env.get_observation_for_agent(p)
            if p == 0:
                # --- Compute instantaneous strategy for player 0 ---
                expected_outcomes = predict_counter_factuals(regret_network, obs)
                cur_policy = predict_policy(policy_network, obs)
                baseline_value = np.dot(expected_outcomes, cur_policy)
                regrets = expected_outcomes - baseline_value
                pos_regrets = np.maximum(regrets, 0)
                sum_pos_regrets = np.sum(pos_regrets)
                if sum_pos_regrets > 1e-8:
                    probabilities = pos_regrets / sum_pos_regrets
                else:
                    probabilities = np.ones(len(DISCRETE_ACTION_SPACE)) / len(DISCRETE_ACTION_SPACE)

                #set all probabilities of invalid bids to 0 
                max_bid = env.get_max_bid(0)
                probabilities[DISCRETE_ACTION_SPACE > max_bid] = 0
                if np.sum(probabilities) == 0:
                    probabilities[0] = 1
                probabilities /= np.sum(probabilities)
                
                # Record the policy sample for training the policy network.
                policy_samples.append([obs, probabilities, 1])
                
                # Determine the main (argmax) action.
                main_action = DISCRETE_ACTION_SPACE[np.argmax(probabilities)]
                
                # --- Branching simulation at the current decision point ---
                branch_actions = []    # actions chosen at this decision point for each branch
                branch_rewards = []    # outcome reward from simulating each branch
                branch_reaches = []    # corresponding reach values (computed from delta_reach)
                
                # For each branch simulate one full rollout.
                for branch in range(n_per_step):
                    # Decide the action for this branch:
                    if branch == 0:
                        # Main branch: use main_action unless a random check perturbs it.
                        #if np.random.random() < iter_epsilon:
                        #    chosen_action = np.random.choice(DISCRETE_ACTION_SPACE)
                        #else:
                        #    chosen_action = main_action
                        chosen_action = main_action
                    else:
                        # Alternative branch: force a deviation by picking a random action (typically not the main one).
                        alt_actions = [a for a in DISCRETE_ACTION_SPACE if a != main_action and a <= max_bid]
                        chosen_action = np.random.choice(alt_actions) if alt_actions else main_action

                    equivalent_actions = []


                    
                    #branch_actions.append(chosen_action)
                    
                    # --- Simulate this branch ---
                    # Make a copy of the current environment state.
                    branch_env = env.copy()
                    
                    # Prepare the actions for the current step in this branch.
                    branch_act = {}
                    branch_act["agent_0"] = chosen_action
                    # For other players, choose actions using the fixed policy.

                    #highest_op = -100
                    for op in range(1, branch_env.num_players):
                        op_obs = branch_env.get_observation_for_agent(op)
                        op_policy = predict_policy(policy_network, op_obs)
                        op_action_idx = np.random.choice(len(DISCRETE_ACTION_SPACE), p=op_policy)
                        branch_act[f"agent_{op}"] = DISCRETE_ACTION_SPACE[op_action_idx]
                        #highest_op = max(DISCRETE_ACTION_SPACE[op_action_idx], highest_op)
                    '''
                    if branch_act["agent_0"] > highest_op and env.get_max_bid(0) > highest_op and False:
                        for i in range(int(np.ceil(highest_op + 1)), 100):
                            equivalent_actions.append(i)
                    else:
                        equivalent_actions = [chosen_action]
                    '''
                        
                        
                    equivalent_actions = [chosen_action]
                    
                    branch_env.step(branch_act)
                    
                    # Roll out the remainder of the game deterministically:
                    # For player 0 we always pick the argmax action from the current regret-based probabilities.
                    while not branch_env.game_over:
                        step_actions = {}
                        for q in range(branch_env.num_players):
                            obs_q = branch_env.get_observation_for_agent(q)
                            if q == 0:
                                exp_outcomes_q = predict_counter_factuals(regret_network, obs_q)
                                pol_q = predict_policy(policy_network, obs_q)
                                baseline_q = np.dot(exp_outcomes_q, pol_q)
                                regrets_q = exp_outcomes_q - baseline_q
                                pos_regrets_q = np.maximum(regrets_q, 0)
                                sum_pos_regrets_q = np.sum(pos_regrets_q)
                                if sum_pos_regrets_q > 1e-8:
                                    probs_q = pos_regrets_q / sum_pos_regrets_q
                                else:
                                    probs_q = np.ones(len(DISCRETE_ACTION_SPACE)) / len(DISCRETE_ACTION_SPACE)
                                step_actions[f"agent_{q}"] = DISCRETE_ACTION_SPACE[np.argmax(probs_q)]
                            else:
                                pol_q = predict_policy(policy_network, obs_q)
                                act_idx_q = np.random.choice(len(DISCRETE_ACTION_SPACE), p=pol_q)
                                step_actions[f"agent_{q}"] = DISCRETE_ACTION_SPACE[act_idx_q]
                        branch_env.step(step_actions)
                    
                    # Get the final reward from this branch.
                    branch_reward = branch_env.get_reward()[0]
                    branch_actions.append(equivalent_actions[0])
                    branch_rewards.append(branch_reward)
                    branch_reaches.append(1)
                    '''
                    for eq in equivalent_actions:
                        branch_actions.append(eq)
                        branch_rewards.append(branch_reward)
                        branch_reaches.append(1/len(equivalent_actions))
                    '''
                    
                    # --- Compute reach probability weight ---
                    # Here we mimic the original logic: if the chosen action is close to the main action,
                    # we assign a higher reach probability.
                    '''if np.abs(chosen_action - main_action) < 0.01:
                        delta_reach = np.log(iter_epsilon / len(DISCRETE_ACTION_SPACE) + 1 - iter_epsilon)
                    else:
                        delta_reach = np.log(iter_epsilon / len(DISCRETE_ACTION_SPACE))
                    branch_reaches.append(np.exp(delta_reach))'''
                    
                
                # --- Record a regret sample for each branch ---
                for a, r, reach in zip(branch_actions, branch_rewards, branch_reaches):
                    regret_samples.append([obs, a, r, reach])
                
                # For the main simulation, continue along branch 0.
                actions["agent_0"] = branch_actions[0]
            else:
                # For players other than player 0, act using the fixed policy.
                cur_policy = predict_policy(policy_network, obs)
                if np.random.random() < proportion_random:
                    act_idx = np.random.choice(len(DISCRETE_ACTION_SPACE))
                else:   
                    act_idx = np.random.choice(len(DISCRETE_ACTION_SPACE), p=cur_policy)
                actions[f"agent_{p}"] = DISCRETE_ACTION_SPACE[act_idx]
        env.step(actions)
    
    return regret_samples, policy_samples

def sim_mock_game(policy_network, iter):
    #### Simulates a game based off a given policy network (assumes all players are playing with it)
    env.reset()

    while not env.game_over:
        actions = {}
        for p in range(env.num_players):
            obs = env.get_observation_for_agent(p)
            cur_policy = predict_policy(policy_network, obs)
            action_idx = np.random.choice(len(DISCRETE_ACTION_SPACE), p=cur_policy)

            action = DISCRETE_ACTION_SPACE[action_idx]
            actions[f"agent_{p}"] = action
        env.step(actions)
    
    log = env.get_game_log()

    with open(os.path.join(LOGS_PATH, f"{iter}.txt"), "w") as f:
        f.write(log)


def clear():
    if sys.platform == "win32":
        os.system("cls")
    else:
        os.system("clear")
    
def play_human(policy_network):
    env.reset()

    while not env.game_over:
        clear()
        # Use the pretty state display instead of the basic one
        env.get_pretty_state()
        actions = {}
        for p in range(env.num_players):
            obs = env.get_observation_for_agent(p)
            if p == 0:
                print(f"Your turn to bid! Current player: {env.nominated_player}")
                #print(f"Player {p}'s policy: {predict_policy(policy_network, obs)}")
                try:
                    action = float(input("Enter your bid: "))/100
                except ValueError:
                    print("Invalid bid. Please enter a number.")
                    action = float(input("Enter your bid: "))/100
            else:
                cur_policy = predict_policy(policy_network, obs)
                action_idx = np.random.choice(len(DISCRETE_ACTION_SPACE), p=cur_policy)
                action = DISCRETE_ACTION_SPACE[action_idx]
            actions[f"agent_{p}"] = action
        env.step(actions)
    clear()
    # Use the pretty game log instead of the basic one
    print(env.get_pretty_game_log())
    input("\nPress Enter to exit...")

# Batched simulation implementation
def batch_predict_counter_factuals(regret_network, observations):
    """
    Predict counterfactual values for all discrete actions for a batch of observations.
    
    Args:
        regret_network: The neural network that takes (obs_dim + 1) input features and outputs a scalar.
        observations: A batch of observations of shape (batch_size, obs_dim).
        
    Returns:
        A NumPy array of predicted counterfactual values for each action for each observation
        with shape (batch_size, n_actions).
    """
    import torch
    if not isinstance(observations, torch.Tensor):
        observations = torch.tensor(observations, dtype=torch.float32)
    
    observations = observations.to(device)
    
    # Get dimensions
    batch_size = observations.shape[0]
    n_actions = len(DISCRETE_ACTION_SPACE)
    
    # Repeat observations for each action: (batch_size, n_actions, obs_dim)
    # First, add a new axis in the middle
    observations_expanded = observations.unsqueeze(1)  # (batch_size, 1, obs_dim)
    # Then repeat along that axis
    observations_batch = observations_expanded.expand(batch_size, n_actions, -1)  # (batch_size, n_actions, obs_dim)
    
    # Reshape to (batch_size * n_actions, obs_dim)
    observations_batch = observations_batch.reshape(-1, observations.shape[1])
    
    # Create a tensor for discrete actions and move it to the correct device
    # We need to repeat each action for each observation in the batch
    action_batch = torch.tensor(DISCRETE_ACTION_SPACE, dtype=torch.float32, device=device)  # (n_actions,)
    action_batch = action_batch.repeat(batch_size)  # (batch_size * n_actions,)
    action_batch = action_batch.unsqueeze(1)  # (batch_size * n_actions, 1)
    
    # Concatenate along the feature dimension
    input_batch = torch.cat((observations_batch, action_batch), dim=1)  # (batch_size * n_actions, obs_dim + 1)
    
    # Forward pass through the regret network
    predicted_values = regret_network(input_batch)  # (batch_size * n_actions,)
    
    # Reshape back to (batch_size, n_actions)
    predicted_values = predicted_values.view(batch_size, n_actions)
    
    return predicted_values.detach().cpu().numpy()


def batch_predict_policy(policy_network, observations):
    """
    Predict action probabilities from the policy network for a batch of observations.
    
    Args:
        policy_network: The network that takes an observation and outputs a probability distribution.
        observations: A batch of observations of shape (batch_size, obs_dim).
        
    Returns:
        A NumPy array of predicted action probabilities with shape (batch_size, n_discrete).
    """
    import torch
    if not isinstance(observations, torch.Tensor):
        observations = torch.tensor(observations, dtype=torch.float32)
    
    observations = observations.to(device)
    
    # Forward pass through the network
    action_probs = torch.exp(policy_network(observations))  # (batch_size, n_discrete)
    
    return action_probs.detach().cpu().numpy()


class BatchedSimulator:
    """
    A class to manage multiple simulation environments in parallel.
    """
    def __init__(self, num_envs, policy_network, regret_network, n_per_step, p_random, proportion_random):
        """
        Initialize a batched simulator with multiple environments.
        
        Args:
            num_envs: Number of environments to simulate in parallel.
            policy_network: Network for predicting policies.
            regret_network: Network for predicting counterfactual values.
            n_per_step: Number of simulation branches at each decision point.
            p_random: Probability parameter for allowing random behavior.
            proportion_random: Proportion of random actions.
        """
        self.num_envs = num_envs
        self.policy_network = policy_network
        self.regret_network = regret_network
        self.n_per_step = n_per_step
        self.p_random = p_random
        
        # Initialize environments
        self.envs = [gen_env() for _ in range(num_envs)]
        
        # Set proportion_random for each environment
        self.proportion_random = [proportion_random if np.random.random() <= p_random else 0 for _ in range(num_envs)]
        
        # Track which environments are still active
        self.active = [True] * num_envs
        
        # Storage for samples
        self.regret_samples = [[] for _ in range(num_envs)]
        self.policy_samples = [[] for _ in range(num_envs)]
        
        # Initialize all environments
        for env in self.envs:
            env.reset()
    
    def step_all_environments(self):
        """
        Advance all active environments by one step.
        Returns True if any environment is still active.
        """
        if not any(self.active):
            return False
        
        # Collect observations for all active environments
        observations = {}
        for player in range(self.envs[0].num_players):
            observations[player] = []
            
        # Collect observations for each player from each active environment
        active_env_indices = []
        for i, (env, is_active) in enumerate(zip(self.envs, self.active)):
            if is_active:
                active_env_indices.append(i)
                for player in range(env.num_players):
                    obs = env.get_observation_for_agent(player)
                    observations[player].append(obs)
        
        if not active_env_indices:
            return False
            
        # Batch predictions for each player
        policy_predictions = {}
        counterfactual_predictions = {}
        
        for player in range(self.envs[0].num_players):
            if observations[player]:  # Only process if there are observations
                # Convert to numpy array
                obs_array = np.array(observations[player])
                
                # Get batch predictions
                policy_predictions[player] = batch_predict_policy(self.policy_network, obs_array)
                
                # For player 0, also get counterfactual predictions
                if player == 0:
                    counterfactual_predictions[player] = batch_predict_counter_factuals(
                        self.regret_network, obs_array
                    )
        
        # Process each active environment
        env_idx = 0
        for i in active_env_indices:
            env = self.envs[i]
            
            actions = {}
            for player in range(env.num_players):
                obs = env.get_observation_for_agent(player)
                
                if player == 0:
                    # Get predictions for this environment
                    expected_outcomes = counterfactual_predictions[player][env_idx]
                    cur_policy = policy_predictions[player][env_idx]
                    
                    # Calculate regrets and probabilities
                    baseline_value = np.dot(expected_outcomes, cur_policy)
                    regrets = expected_outcomes - baseline_value
                    pos_regrets = np.maximum(regrets, 0)
                    sum_pos_regrets = np.sum(pos_regrets)
                    
                    if sum_pos_regrets > 1e-8:
                        probabilities = pos_regrets / sum_pos_regrets
                    else:
                        probabilities = np.ones(len(DISCRETE_ACTION_SPACE)) / len(DISCRETE_ACTION_SPACE)
                    
                    # Handle invalid bids
                    max_bid = env.get_max_bid(0)
                    probabilities[DISCRETE_ACTION_SPACE > max_bid] = 0
                    if np.sum(probabilities) == 0:
                        probabilities[0] = 1
                    probabilities /= np.sum(probabilities)
                    
                    # Record policy sample
                    self.policy_samples[i].append([obs, probabilities, 1])
                    
                    # Determine the main action
                    main_action = DISCRETE_ACTION_SPACE[np.argmax(probabilities)]
                    
                    # Process branches
                    branch_actions = []
                    branch_rewards = []
                    branch_reaches = []
                    branch_envs = []
                    
                    # Setup branches for simulation
                    for branch in range(self.n_per_step):
                        # Choose action for this branch
                        if branch == 0:
                            chosen_action = main_action
                        else:
                            alt_actions = [a for a in DISCRETE_ACTION_SPACE if a != main_action and a <= max_bid]
                            chosen_action = np.random.choice(alt_actions) if alt_actions else main_action
                        
                        # Create the branch environment
                        branch_env = env.copy()
                        branch_actions.append(chosen_action)
                        branch_envs.append(branch_env)
                        branch_reaches.append(1)
                    
                    # First step of all branches - still need to individually process for other players
                    for branch_idx, branch_env in enumerate(branch_envs):
                        branch_act = {}
                        branch_act["agent_0"] = branch_actions[branch_idx]
                        
                        # Choose actions for other players
                        for op in range(1, branch_env.num_players):
                            op_obs = branch_env.get_observation_for_agent(op)
                            op_policy = predict_policy(self.policy_network, op_obs)
                            op_action_idx = np.random.choice(len(DISCRETE_ACTION_SPACE), p=op_policy)
                            branch_act[f"agent_{op}"] = DISCRETE_ACTION_SPACE[op_action_idx]
                        
                        branch_env.step(branch_act)
                    
                    # Now continue with the efficient batch processing of branches until they are all done
                    active_branches = [True] * len(branch_envs)
                    
                    while any(active_branches):
                        # Collect observations from all active branches
                        branch_observations = {}
                        for player in range(branch_envs[0].num_players):
                            branch_observations[player] = []
                        
                        active_branch_indices = []
                        for b_idx, (b_env, is_active) in enumerate(zip(branch_envs, active_branches)):
                            if not is_active or b_env.game_over:
                                active_branches[b_idx] = False
                                continue
                                
                            active_branch_indices.append(b_idx)
                            for player in range(b_env.num_players):
                                obs_b = b_env.get_observation_for_agent(player)
                                branch_observations[player].append(obs_b)
                        
                        if not active_branch_indices:
                            break
                            
                        # Get batched predictions for all branches
                        branch_policy_preds = {}
                        branch_cf_preds = {}
                        
                        for player in range(branch_envs[0].num_players):
                            if branch_observations[player]:
                                obs_array_b = np.array(branch_observations[player])
                                branch_policy_preds[player] = batch_predict_policy(self.policy_network, obs_array_b)
                                
                                if player == 0:
                                    branch_cf_preds[player] = batch_predict_counter_factuals(
                                        self.regret_network, obs_array_b
                                    )
                        
                        # Process each active branch
                        b_env_idx = 0
                        for b_idx in active_branch_indices:
                            b_env = branch_envs[b_idx]
                            
                            step_actions = {}
                            for q in range(b_env.num_players):
                                if q == 0:
                                    # Use regret-minimizing action for player 0
                                    exp_outcomes_q = branch_cf_preds[q][b_env_idx]
                                    pol_q = branch_policy_preds[q][b_env_idx]
                                    baseline_q = np.dot(exp_outcomes_q, pol_q)
                                    regrets_q = exp_outcomes_q - baseline_q
                                    pos_regrets_q = np.maximum(regrets_q, 0)
                                    sum_pos_regrets_q = np.sum(pos_regrets_q)
                                    if sum_pos_regrets_q > 1e-8:
                                        probs_q = pos_regrets_q / sum_pos_regrets_q
                                    else:
                                        probs_q = np.ones(len(DISCRETE_ACTION_SPACE)) / len(DISCRETE_ACTION_SPACE)
                                    step_actions[f"agent_{q}"] = DISCRETE_ACTION_SPACE[np.argmax(probs_q)]
                                else:
                                    # Use policy for other players
                                    pol_q = branch_policy_preds[q][b_env_idx]
                                    act_idx_q = np.random.choice(len(DISCRETE_ACTION_SPACE), p=pol_q)
                                    step_actions[f"agent_{q}"] = DISCRETE_ACTION_SPACE[act_idx_q]
                            
                            b_env.step(step_actions)
                            if b_env.game_over:
                                active_branches[b_idx] = False
                                
                            b_env_idx += 1
                    
                    # Get rewards for all branches
                    for b_idx, b_env in enumerate(branch_envs):
                        branch_rewards.append(b_env.get_reward()[0])
                    
                    # Record regret samples
                    for a, r, reach in zip(branch_actions, branch_rewards, branch_reaches):
                        self.regret_samples[i].append([obs, a, r, reach])
                    
                    # Continue with main action
                    actions["agent_0"] = branch_actions[0]
                
                else:
                    # For other players, use policy
                    cur_policy = policy_predictions[player][env_idx]
                    if np.random.random() < self.proportion_random[i]:
                        act_idx = np.random.choice(len(DISCRETE_ACTION_SPACE))
                    else:
                        act_idx = np.random.choice(len(DISCRETE_ACTION_SPACE), p=cur_policy)
                    actions[f"agent_{player}"] = DISCRETE_ACTION_SPACE[act_idx]
            
            # Step the environment
            env.step(actions)
            
            # Check if this environment is done
            if env.game_over:
                self.active[i] = False
            
            env_idx += 1
        
        return any(self.active)
    
    def get_samples(self):
        """
        Return collected samples from all environments.
        """
        all_regret_samples = []
        all_policy_samples = []
        
        for r_samples, p_samples in zip(self.regret_samples, self.policy_samples):
            all_regret_samples.extend(r_samples)
            all_policy_samples.extend(p_samples)
        
        return all_regret_samples, all_policy_samples


def gen_batch_simulations(policy_network, regret_network, n_per_step, p_random, proportion_random, num_envs=1000):
    """
    Generate simulations in parallel using batched prediction.
    
    Args:
        policy_network: Network for predicting policies.
        regret_network: Network for predicting counterfactual values.
        n_per_step: Number of simulation branches at each decision point.
        p_random: Probability of random behavior.
        proportion_random: Proportion of random actions.
        num_envs: Number of environments to simulate in parallel.
        
    Returns:
        A tuple of (regret_samples, policy_samples) for training.
    """
    # Initialize the batched simulator
    simulator = BatchedSimulator(
        num_envs=num_envs,
        policy_network=policy_network,
        regret_network=regret_network,
        n_per_step=n_per_step,
        p_random=p_random,
        proportion_random=proportion_random
    )
    
    # Run all environments until completion
    while simulator.step_all_environments():
        pass
    
    # Get collected samples
    return simulator.get_samples()

def compare_simulation_approaches(policy_network, regret_network, n_per_step, p_random, proportion_random, num_envs=1000):
    """
    Demonstrates and compares both simulation approaches.
    
    Args:
        policy_network: Network for predicting policies.
        regret_network: Network for predicting counterfactual values.
        n_per_step: Number of simulation branches at each decision point.
        p_random: Probability of random behavior.
        proportion_random: Proportion of random actions.
        num_envs: Number of environments for batched simulation.
    """
    import time
    
    # Test sequential approach with a small number of simulations
    num_seq_sims = 10
    print(f"Running {num_seq_sims} sequential simulations...")
    seq_start = time.time()
    all_regret_samples = []
    all_policy_samples = []
    
    for i in range(num_seq_sims):
        regret_samples, policy_samples = gen_simulation(
            policy_network, regret_network, n_per_step, p_random, proportion_random)
        all_regret_samples.extend(regret_samples)
        all_policy_samples.extend(policy_samples)
    
    seq_end = time.time()
    seq_time = seq_end - seq_start
    print(f"Sequential simulations took {seq_time:.2f} seconds")
    print(f"Generated {len(all_regret_samples)} regret samples and {len(all_policy_samples)} policy samples")
    print(f"Time per simulation: {seq_time / num_seq_sims:.2f} seconds")
    
    # Test batched approach with the same number of simulations
    print(f"\nRunning {num_seq_sims} batched simulations...")
    batch_start = time.time()
    
    batched_regret_samples, batched_policy_samples = gen_batch_simulations(
        policy_network, regret_network, n_per_step, p_random, proportion_random, num_envs=num_seq_sims)
    
    batch_end = time.time()
    batch_time = batch_end - batch_start
    print(f"Batched simulations took {batch_time:.2f} seconds")
    print(f"Generated {len(batched_regret_samples)} regret samples and {len(batched_policy_samples)} policy samples")
    print(f"Time per simulation: {batch_time / num_seq_sims:.2f} seconds")
    
    # Calculate speedup
    speedup = seq_time / batch_time
    print(f"\nSpeedup from batching: {speedup:.2f}x")
    
    # Extrapolate to full-scale simulation
    print(f"\nExtrapolated time for {num_envs} simulations:")
    print(f"Sequential: {seq_time / num_seq_sims * num_envs:.2f} seconds "
          f"({seq_time / num_seq_sims * num_envs / 60:.2f} minutes)")
    print(f"Batched: {batch_time / num_seq_sims * num_envs:.2f} seconds "
          f"({batch_time / num_seq_sims * num_envs / 60:.2f} minutes)")
    
    return all_regret_samples, all_policy_samples, batched_regret_samples, batched_policy_samples

class FullyBatchedSimulator:
    """
    A class to manage multiple simulation environments in parallel
    with full batching of branch simulations across all environments.
    """
    def __init__(self, num_envs, policy_network, regret_network, n_per_step, p_random, proportion_random):
        """
        Initialize a fully batched simulator with multiple environments.
        
        Args:
            num_envs: Number of environments to simulate in parallel.
            policy_network: Network for predicting policies.
            regret_network: Network for predicting counterfactual values.
            n_per_step: Number of simulation branches at each decision point.
            p_random: Probability parameter for allowing random behavior.
            proportion_random: Proportion of random actions.
        """
        self.num_envs = num_envs
        self.policy_network = policy_network
        self.regret_network = regret_network
        self.n_per_step = n_per_step
        self.p_random = p_random
        
        # Initialize environments
        self.envs = [gen_env() for _ in range(num_envs)]
        
        # Set proportion_random for each environment
        self.proportion_random = [proportion_random if np.random.random() <= p_random else 0 for _ in range(num_envs)]
        
        # Track which environments are still active
        self.active = [True] * num_envs
        
        # Storage for samples
        self.regret_samples = [[] for _ in range(num_envs)]
        self.policy_samples = [[] for _ in range(num_envs)]
        
        # Initialize all environments
        for env in self.envs:
            env.reset()
    
    def step_all_environments(self):
        """
        Advance all active environments by one step.
        Returns True if any environment is still active.
        """
        if not any(self.active):
            return False
        
        # Collect observations for all active environments
        observations = {}
        for player in range(self.envs[0].num_players):
            observations[player] = []
            
        # Collect observations for each player from each active environment
        active_env_indices = []
        active_env_map = {}  # Maps active_idx -> original_env_idx
        for i, (env, is_active) in enumerate(zip(self.envs, self.active)):
            if is_active:
                active_idx = len(active_env_indices)
                active_env_indices.append(i)
                active_env_map[active_idx] = i
                for player in range(env.num_players):
                    obs = env.get_observation_for_agent(player)
                    observations[player].append(obs)
        
        if not active_env_indices:
            return False
            
        # Batch predictions for each player
        policy_predictions = {}
        counterfactual_predictions = {}
        
        for player in range(self.envs[0].num_players):
            if observations[player]:  # Only process if there are observations
                # Convert to numpy array
                obs_array = np.array(observations[player])
                
                # Get batch predictions
                policy_predictions[player] = batch_predict_policy(self.policy_network, obs_array)
                
                # For player 0, also get counterfactual predictions
                if player == 0:
                    counterfactual_predictions[player] = batch_predict_counter_factuals(
                        self.regret_network, obs_array
                    )
        
        # Setup for all branches across all environments
        all_branches = []  # List of (env_idx, branch_env, chosen_action, obs, main_action)
        all_orig_env_indices = []  # Original environment indices for each branch
        
        # First, collect info for all branches that need to be simulated
        for active_idx, i in enumerate(active_env_indices):
            env = self.envs[i]
            obs = env.get_observation_for_agent(0)  # Player 0 observation
            
            # Get predictions for this environment
            expected_outcomes = counterfactual_predictions[0][active_idx]
            cur_policy = policy_predictions[0][active_idx]
            
            # Calculate regrets and probabilities
            baseline_value = np.dot(expected_outcomes, cur_policy)
            regrets = expected_outcomes - baseline_value
            pos_regrets = np.maximum(regrets, 0)
            sum_pos_regrets = np.sum(pos_regrets)
            
            if sum_pos_regrets > 1e-8:
                probabilities = pos_regrets / sum_pos_regrets
            else:
                probabilities = np.ones(len(DISCRETE_ACTION_SPACE)) / len(DISCRETE_ACTION_SPACE)
            
            # Handle invalid bids
            max_bid = env.get_max_bid(0)
            probabilities[DISCRETE_ACTION_SPACE > max_bid] = 0
            if np.sum(probabilities) == 0:
                probabilities[0] = 1
            probabilities /= np.sum(probabilities)
            
            # Record policy sample
            self.policy_samples[i].append([obs, probabilities, 1])
            
            # Determine the main action
            main_action = DISCRETE_ACTION_SPACE[np.argmax(probabilities)]
            
            # Create branches for this environment
            for branch in range(self.n_per_step):
                # Choose action for this branch
                if branch == 0:
                    chosen_action = main_action
                else:
                    alt_actions = [a for a in DISCRETE_ACTION_SPACE if a != main_action and a <= max_bid]
                    chosen_action = np.random.choice(alt_actions) if alt_actions else main_action
                
                # Create the branch environment
                branch_env = env.copy()
                
                all_branches.append((i, branch_env, chosen_action, obs, main_action))
                all_orig_env_indices.append(i)
        
        # Setup for storing branch actions for each environment
        env_branch_actions = {i: [] for i in active_env_indices}
        env_branch_rewards = {i: [] for i in active_env_indices}
        
        # Perform initial action for all branches
        branch_envs = []
        for env_idx, branch_env, chosen_action, _, _ in all_branches:
            branch_act = {"agent_0": chosen_action}
            
            # Get actions for other players
            for op in range(1, branch_env.num_players):
                op_obs = branch_env.get_observation_for_agent(op)
                op_idx = active_env_indices.index(env_idx)  # Map to index in active_env_indices
                op_policy = policy_predictions[op][op_idx]
                op_action_idx = np.random.choice(len(DISCRETE_ACTION_SPACE), p=op_policy)
                branch_act[f"agent_{op}"] = DISCRETE_ACTION_SPACE[op_action_idx]
            
            # Apply actions
            branch_env.step(branch_act)
            branch_envs.append(branch_env)
            
            # Store chosen action for this environment
            env_branch_actions[env_idx].append(chosen_action)
        
        # Now continue with fully batched branch simulations
        active_branches = [True] * len(branch_envs)
        
        while any(active_branches):
            # Collect observations from all active branches
            branch_observations = {}
            for player in range(self.envs[0].num_players):
                branch_observations[player] = []
            
            # Map from active_branch_idx -> (branch_idx, branch_env)
            active_branch_indices = []
            active_branch_map = {}
            
            for b_idx, (branch_env, is_active) in enumerate(zip(branch_envs, active_branches)):
                if not is_active or branch_env.game_over:
                    active_branches[b_idx] = False
                    continue
                
                active_idx = len(active_branch_indices)
                active_branch_indices.append(b_idx)
                active_branch_map[active_idx] = b_idx
                
                for player in range(branch_env.num_players):
                    obs_b = branch_env.get_observation_for_agent(player)
                    branch_observations[player].append(obs_b)
            
            if not active_branch_indices:
                break
            
            # Batch predict for all active branches across all environments
            branch_policy_preds = {}
            branch_cf_preds = {}
            
            for player in range(self.envs[0].num_players):
                if branch_observations[player]:
                    obs_array_b = np.array(branch_observations[player])
                    branch_policy_preds[player] = batch_predict_policy(self.policy_network, obs_array_b)
                    
                    if player == 0:
                        branch_cf_preds[player] = batch_predict_counter_factuals(
                            self.regret_network, obs_array_b
                        )
            
            # Process each active branch
            for active_idx, b_idx in active_branch_map.items():
                branch_env = branch_envs[b_idx]
                
                step_actions = {}
                for q in range(branch_env.num_players):
                    if q == 0:
                        # Use regret-minimizing action for player 0
                        exp_outcomes_q = branch_cf_preds[q][active_idx]
                        pol_q = branch_policy_preds[q][active_idx]
                        baseline_q = np.dot(exp_outcomes_q, pol_q)
                        regrets_q = exp_outcomes_q - baseline_q
                        pos_regrets_q = np.maximum(regrets_q, 0)
                        sum_pos_regrets_q = np.sum(pos_regrets_q)
                        if sum_pos_regrets_q > 1e-8:
                            probs_q = pos_regrets_q / sum_pos_regrets_q
                        else:
                            probs_q = np.ones(len(DISCRETE_ACTION_SPACE)) / len(DISCRETE_ACTION_SPACE)
                        step_actions[f"agent_{q}"] = DISCRETE_ACTION_SPACE[np.argmax(probs_q)]
                    else:
                        # Use policy for other players
                        pol_q = branch_policy_preds[q][active_idx]
                        act_idx_q = np.random.choice(len(DISCRETE_ACTION_SPACE), p=pol_q)
                        step_actions[f"agent_{q}"] = DISCRETE_ACTION_SPACE[act_idx_q]
                
                branch_env.step(step_actions)
                if branch_env.game_over:
                    active_branches[b_idx] = False
        
        # Collect rewards from all branches
        for b_idx, branch_env in enumerate(branch_envs):
            env_idx, _, _, _, _ = all_branches[b_idx]
            reward = branch_env.get_reward()[0]
            env_branch_rewards[env_idx].append(reward)
        
        # Record regret samples and perform main actions
        for i in active_env_indices:
            if i in env_branch_actions and env_branch_actions[i]:
                # Record regret samples for this environment
                env_obs = None
                env_main_action = None
                
                # Find the original observation and main action for this environment
                for b_idx, (env_idx, _, _, obs, main_action) in enumerate(all_branches):
                    if env_idx == i:
                        env_obs = obs
                        env_main_action = main_action
                        break
                
                if env_obs is not None:
                    for action, reward in zip(env_branch_actions[i], env_branch_rewards[i]):
                        self.regret_samples[i].append([env_obs, action, reward, 1])
                
                # Perform main action on original environment
                actions = {}
                actions["agent_0"] = env_main_action
                
                # Get actions for other players
                for player in range(1, self.envs[i].num_players):
                    active_idx = active_env_indices.index(i)
                    cur_policy = policy_predictions[player][active_idx]
                    if np.random.random() < self.proportion_random[i]:
                        act_idx = np.random.choice(len(DISCRETE_ACTION_SPACE))
                    else:
                        act_idx = np.random.choice(len(DISCRETE_ACTION_SPACE), p=cur_policy)
                    actions[f"agent_{player}"] = DISCRETE_ACTION_SPACE[act_idx]
                
                # Step the environment
                self.envs[i].step(actions)
                
                # Check if this environment is done
                if self.envs[i].game_over:
                    self.active[i] = False
        
        return any(self.active)
    
    def get_samples(self):
        """
        Return collected samples from all environments.
        """
        all_regret_samples = []
        all_policy_samples = []
        
        for r_samples, p_samples in zip(self.regret_samples, self.policy_samples):
            all_regret_samples.extend(r_samples)
            all_policy_samples.extend(p_samples)
        
        return all_regret_samples, all_policy_samples


def gen_fully_batched_simulations(policy_network, regret_network, n_per_step, p_random, proportion_random, num_envs=1000):
    """
    Generate simulations in parallel using fully batched prediction for branches across all environments.
    
    Args:
        policy_network: Network for predicting policies.
        regret_network: Network for predicting counterfactual values.
        n_per_step: Number of simulation branches at each decision point.
        p_random: Probability of random behavior.
        proportion_random: Proportion of random actions.
        num_envs: Number of environments to simulate in parallel.
        
    Returns:
        A tuple of (regret_samples, policy_samples) for training.
    """
    # Initialize the fully batched simulator
    simulator = FullyBatchedSimulator(
        num_envs=num_envs,
        policy_network=policy_network,
        regret_network=regret_network,
        n_per_step=n_per_step,
        p_random=p_random,
        proportion_random=proportion_random
    )
    
    # Run all environments until completion
    while simulator.step_all_environments():
        pass
    
    # Get collected samples
    return simulator.get_samples()


def compare_all_simulation_approaches(policy_network, regret_network, n_per_step, p_random, proportion_random, num_envs=10):
    """
    Demonstrates and compares all three simulation approaches.
    
    Args:
        policy_network: Network for predicting policies.
        regret_network: Network for predicting counterfactual values.
        n_per_step: Number of simulation branches at each decision point.
        p_random: Probability of random behavior.
        proportion_random: Proportion of random actions.
        num_envs: Number of environments for batched simulation.
    """
    import time
    
    # Test sequential approach with a small number of simulations
    num_seq_sims = 10
    print(f"Running {num_seq_sims} sequential simulations...")
    seq_start = time.time()
    all_regret_samples = []
    all_policy_samples = []
    
    for i in range(num_seq_sims):
        regret_samples, policy_samples = gen_simulation(
            policy_network, regret_network, n_per_step, p_random, proportion_random)
        all_regret_samples.extend(regret_samples)
        all_policy_samples.extend(policy_samples)
    
    seq_end = time.time()
    seq_time = seq_end - seq_start
    print(f"Sequential simulations took {seq_time:.2f} seconds")
    print(f"Generated {len(all_regret_samples)} regret samples and {len(all_policy_samples)} policy samples")
    print(f"Time per simulation: {seq_time / num_seq_sims:.2f} seconds")
    
    # Test partially batched approach with the same number of simulations
    print(f"\nRunning {num_seq_sims} partially batched simulations...")
    batch_start = time.time()
    
    batched_regret_samples, batched_policy_samples = gen_batch_simulations(
        policy_network, regret_network, n_per_step, p_random, proportion_random, num_envs=num_seq_sims)
    
    batch_end = time.time()
    batch_time = batch_end - batch_start
    print(f"Partially batched simulations took {batch_time:.2f} seconds")
    print(f"Generated {len(batched_regret_samples)} regret samples and {len(batched_policy_samples)} policy samples")
    print(f"Time per simulation: {batch_time / num_seq_sims:.2f} seconds")
    
    # Test fully batched approach with the same number of simulations
    print(f"\nRunning {num_seq_sims} fully batched simulations...")
    full_batch_start = time.time()
    
    full_batched_regret_samples, full_batched_policy_samples = gen_fully_batched_simulations(
        policy_network, regret_network, n_per_step, p_random, proportion_random, num_envs=num_seq_sims)
    
    full_batch_end = time.time()
    full_batch_time = full_batch_end - full_batch_start
    print(f"Fully batched simulations took {full_batch_time:.2f} seconds")
    print(f"Generated {len(full_batched_regret_samples)} regret samples and {len(full_batched_policy_samples)} policy samples")
    print(f"Time per simulation: {full_batch_time / num_seq_sims:.2f} seconds")
    
    # Calculate speedups
    speedup1 = seq_time / batch_time
    speedup2 = seq_time / full_batch_time
    speedup3 = batch_time / full_batch_time
    
    print(f"\nSpeedup from sequential to partial batching: {speedup1:.2f}x")
    print(f"Speedup from sequential to full batching: {speedup2:.2f}x")
    print(f"Speedup from partial to full batching: {speedup3:.2f}x")
    
    # Extrapolate to full-scale simulation
    print(f"\nExtrapolated time for {num_envs} simulations:")
    print(f"Sequential: {seq_time / num_seq_sims * num_envs:.2f} seconds "
          f"({seq_time / num_seq_sims * num_envs / 60:.2f} minutes)")
    print(f"Partially batched: {batch_time / num_seq_sims * num_envs:.2f} seconds "
          f"({batch_time / num_seq_sims * num_envs / 60:.2f} minutes)")
    print(f"Fully batched: {full_batch_time / num_seq_sims * num_envs:.2f} seconds "
          f"({full_batch_time / num_seq_sims * num_envs / 60:.2f} minutes)")
    
    return all_regret_samples, all_policy_samples, batched_regret_samples, batched_policy_samples, full_batched_regret_samples, full_batched_policy_samples




