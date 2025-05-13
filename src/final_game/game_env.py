import numpy as np
from enum import Enum
from scipy.stats import gamma
import random
import copy
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import gymnasium as gym
import numpy as np
import json
import pdb
#ATHLETES = sorted([150, 120, 100, 90, 85, 80, 75, 70, 62, 61, 60, 45], reverse=True)
#ATHLETES = sorted([5, 10, 75, 85,  155, 160], reverse=True)

#######################
# RLlib-compatible Auction Environment
#######################
'''

Will try a single position game with CFR

We will fix the list of players - so that we limit the scope of possibilities - and then our function we train will be:

f(B_self, B_opp, Mean_self, Mean_opp, Slots_self, Slots_opp, Cur Player Mean) -> distribution over 1 -> 100 of how much to bid

We will use Deep CFR to solve this


'''
class AuctionEnv(MultiAgentEnv):

	def __init__(self, forwards, defensemen, goalies, num_players=2):
		super().__init__()
		# For simplicity, we assume state=None (i.e. fresh game)
		self.agents = [f"agent_{i}" for i in range(num_players)]
		self.num_players = num_players
		self.GAME_BUDGET = 100

		self.goalies_bkup = copy.deepcopy(goalies)
		self.forwards_bkup = copy.deepcopy(forwards)
		self.defensemen_bkup = copy.deepcopy(defensemen)

		self.MAX_VAL = sum(goalies + forwards + defensemen)

		self.athletes = [(x, "G") for x in goalies] + [(x, "D") for x in defensemen] + [(x, "F") for x in forwards]
		self.athletes = sorted(self.athletes, key = lambda x : x[0], reverse=True)

		self.goalies_needed = np.ones(self.num_players) * len(self.goalies_bkup)/self.num_players
		self.defensemen_needed = np.ones(self.num_players) * len(self.defensemen_bkup)/self.num_players
		self.forwards_needed = np.ones(self.num_players) * len(self.forwards_bkup)/self.num_players
		self.TEAM_SIZE = len(self.athletes)/2

		self.budgets = np.ones(self.num_players) * self.GAME_BUDGET
		# Array to hold bids from each agent (initialized to -1)
		self.bids_placed = -1 * np.ones(self.num_players)
		# Start with a nominated player (pop from list)
		self.nominated_player = self.athletes.pop(0)

		self.members_means = np.zeros(self.num_players)
		self.last_winner = -1
		self.last_price = -1
		self.prev_bids = np.zeros(self.num_players)

		self.game_over = False
		self.history = {}  # Auction history

		self.illegal_bid_penalties = np.zeros(self.num_players)

		# Define the action space for each agent.
		per_agent_action_space = gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32)
		self.action_space = gym.spaces.Dict({
			agent: per_agent_action_space for agent in self.agents
		})

		# Define an observation space. (You'll need to adjust the bounds/shapes to your needs.)
		obs_dim = (
			1 #self budget 
			+ self.num_players - 1 #opp budget
			+ 1 #self mean
			+ self.num_players - 1 #opp mean
			+ 1 #self slots forward
			+ self.num_players - 1 #opp slots forward
			+ 1 #self slots defense
			+ self.num_players - 1 #opp slots defense
			+ 1 #self slots goalie
			+ self.num_players - 1 #opp slots goalie
			+ 1 #current athlete mean
			+ 3 #current athlete position
		)

		self.OBS_DIM = obs_dim

		per_agent_obs_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_dim,), dtype=np.float32)
		self.observation_space = gym.spaces.Dict({
			agent: per_agent_obs_space for agent in self.agents
		})
	
	def get_must_bid(self, agent):
		"""
		Returns 1 if the agent is forced to bid on the current nominated player,
		based on their remaining roster needs.
		"""

		if self.nominated_player[1] == "G" and self.goalies_needed[agent] == 0:
			return 1
		elif self.nominated_player[1] == "D" and self.defensemen_needed[agent] == 0:
			return 1
		elif self.nominated_player[1] == "F" and self.forwards_needed[agent] == 0:
			return 1
		
		return 0

	def get_max_bid(self, player):
		"""
		Returns the maximum bid an agent can make.
		(Here we use a fixed range, e.g., 0 to 1000.)
		"""

		if self.nominated_player[1] == "G" and self.goalies_needed[player] <= 0:
			return -1e10
		elif self.nominated_player[1] == "D" and self.defensemen_needed[player] <= 0:
			return -1e10
		elif self.nominated_player[1] == "F" and self.forwards_needed[player] <= 0:
			return -1e10

		max_val = self.budgets[player] #- self.athletes_needed[player] + 1
		return max_val
	
	# --- RLlib Required Methods: reset and step ---

	def reset(self, seed=None, options=None):
		"""
		Reset the environment to the initial state and return observations for all agents.
		"""
		# For simplicity, we reinitialize a new game.
		self.__init__(self.forwards_bkup, self.defensemen_bkup, self.goalies_bkup, self.num_players)
		obs = {}
		for i in range(self.num_players):
			obs[f"agent_{i}"] = self.get_observation_for_agent(i)
		return obs, {}
	
	def get_game_log(self):
		output = ""
		output += f"Final Means: {self.members_means}\n"
		output += f"Final Budgets: {self.budgets}\n"
		output += "\n\n\nHistory:\n"
		for x in self.history:
			output += f"{str(x)} : {str(self.history[x])}\n"
		return output
	
	def get_pretty_game_log(self):
		"""
		Returns a nicely formatted final game log using rich library
		"""
		try:
			from rich.console import Console
			from rich.table import Table
			from rich.panel import Panel
			from rich.text import Text
			
			console = Console(record=True, width=100)
			
			# Final Results Table
			results = Table(show_header=True, header_style="bold magenta")
			results.add_column("Player")
			results.add_column("Team Value", justify="right")
			results.add_column("Budget Left", justify="right")
			results.add_column("Result", justify="center")
			
			# Calculate winner
			max_score = max(self.members_means)
			winners = np.where(self.members_means == max_score)[0]
			
			# Sort players by team value (descending)
			player_indices = list(range(self.num_players))
			player_indices.sort(key=lambda i: self.members_means[i], reverse=True)
			
			for rank, i in enumerate(player_indices):
				player_id = f"Player {i}"
				if i == 0:
					player_id = f"Player {i} (You)"
				
				result = ""
				if i in winners:
					result = "[bold green]Winner![/bold green]"
					player_id = f"[bold green]{player_id}[/bold green]"
				else:
					rank_str = {0: "1st", 1: "2nd", 2: "3rd"}.get(rank, f"{rank+1}th")
					result = f"{rank_str} place"
				
				results.add_row(
					player_id,
					f"{self.members_means[i]:.0f}",
					f"{self.budgets[i]:.0f}",
					result
				)
			
			# Auction History Table
			auction_history = Table(show_header=True, header_style="bold blue")
			auction_history.add_column("Player")
			auction_history.add_column("Winner", justify="center")
			auction_history.add_column("Price", justify="right")
			auction_history.add_column("Bids", justify="left")
			
			for player, info in self.history.items():
				player_str = f"{player[0]} ({player[1]})"
				winner = f"Player {info['winner']}"
				if info['winner'] == 0:
					winner = "[bold green]You[/bold green]"
				price = f"{info['price']:.0f}"
				bids = ", ".join([f"{i}:{b:.0f}" for i, b in enumerate(info['bids']) if b >= 0])
				
				auction_history.add_row(player_str, winner, price, bids)
			
			# Put it all together
			console.print(Panel(Text("🏆 FINAL RESULTS 🏆", justify="center"), border_style="yellow"))
			console.print(results)
			console.print()
			console.print(Panel(auction_history, title="Complete Auction History", border_style="blue"))
			
			#return console.export_text()
			
		except ImportError:
			# Fallback to regular display if rich is not available
			return self.get_game_log()
	
	def get_current_state(self):
		output = ""
		output += f"Current Means: {self.members_means}\n"
		output += f"Current Budgets: {self.budgets}\n"
		output += f"Current Athlete: {self.nominated_player}\n"
		defense_left = [x for x in self.athletes if x[1] == "D"]
		goalies_left = [x for x in self.athletes if x[1] == "G"]
		forwards_left = [x for x in self.athletes if x[1] == "F"]
		output += f"Forwards Left: {forwards_left}\n"
		output += f"Defensemen Left: {defense_left}\n"
		output += f"Goalies Left: {goalies_left}\n" 
		#output += f"Athletes Left: {self.athletes}\n"
		output += f"Forwards Needed: {self.forwards_needed}\n"
		output += f"Defensemen Needed: {self.defensemen_needed}\n"
		output += f"Goalies Needed: {self.goalies_needed}\n"
		output += "\n\n\nHistory:\n"
		for x in self.history:
			output += f"{str(x)} : {str(self.history[x])}\n"
		return output
	
	def get_pretty_state(self):
		"""
		Returns a nicely formatted representation of the current game state using rich library
		"""
		try:
			from rich.console import Console
			from rich.table import Table
			from rich.panel import Panel
			from rich.columns import Columns
			from rich.text import Text
			
			console = Console(record=True, width=100)
			
			# Current Auction Panel
			current_player = Table.grid(padding=(0, 2))
			player_value, player_pos = self.nominated_player
			position_full = {"G": "Goalie", "D": "Defenseman", "F": "Forward"}[player_pos]
			
			current_player.add_row(
				"[bold yellow]Current Player Auction[/bold yellow]", 
				f"[bold cyan]{position_full} (Value: {player_value})[/bold cyan]"
			)
			
			auction_panel = Panel(
				current_player,
				title="🏒 AUCTION IN PROGRESS", 
				border_style="green"
			)
			
			# Player Stats Table
			player_stats = Table(show_header=True, header_style="bold magenta")
			player_stats.add_column("Player")
			player_stats.add_column("Budget", justify="right")
			player_stats.add_column("Team Value", justify="right")
			player_stats.add_column("F Needed", justify="center")
			player_stats.add_column("D Needed", justify="center")
			player_stats.add_column("G Needed", justify="center")
			
			for i in range(self.num_players):
				player_id = f"Player {i}"
				if i == 0:
					player_id = f"[bold green]{player_id} (You)[/bold green]"
				
				budget_color = "green"
				if self.budgets[i] < 10:
					budget_color = "red"
				elif self.budgets[i] < 30:
					budget_color = "yellow"
				
				player_stats.add_row(
					player_id,
					f"[{budget_color}]{self.budgets[i]:.0f}[/{budget_color}]",
					f"{self.members_means[i]:.0f}",
					f"{self.forwards_needed[i]:.0f}",
					f"{self.defensemen_needed[i]:.0f}",
					f"{self.goalies_needed[i]:.0f}"
				)
			
			# Remaining Players Table
			remaining_players = Table(show_header=True, header_style="bold blue")
			remaining_players.add_column("Position")
			remaining_players.add_column("Players Left", justify="right")
			remaining_players.add_column("Values", justify="left")
			
			def format_player_list(players):
				if not players:
					return "[italic]None[/italic]"
				values = [str(p[0]) for p in players]
				return ", ".join(values)
			
			forwards_left = [x for x in self.athletes if x[1] == "F"]
			defensemen_left = [x for x in self.athletes if x[1] == "D"]
			goalies_left = [x for x in self.athletes if x[1] == "G"]
			
			remaining_players.add_row(
				"Forwards", 
				str(len(forwards_left)), 
				format_player_list(forwards_left)
			)
			remaining_players.add_row(
				"Defensemen", 
				str(len(defensemen_left)), 
				format_player_list(defensemen_left)
			)
			remaining_players.add_row(
				"Goalies", 
				str(len(goalies_left)), 
				format_player_list(goalies_left)
			)
			
			# Auction History Table
			auction_history = Table(show_header=True, header_style="bold red")
			auction_history.add_column("Player")
			auction_history.add_column("Winner", justify="center")
			auction_history.add_column("Price", justify="right")
			auction_history.add_column("Bids", justify="left")
			
			# Show most recent auctions first (up to 5)
			history_items = list(self.history.items())[-5:]
			for player, info in reversed(history_items):
				player_str = f"{player[0]} ({player[1]})"
				winner = f"Player {info['winner']}"
				if info['winner'] == 0:
					winner = "[bold green]You[/bold green]"
				price = f"{info['price']:.0f}"
				bids = ", ".join([f"{i}:{b:.0f}" for i, b in enumerate(info['bids']) if b >= 0])
				
				auction_history.add_row(player_str, winner, price, bids)
			
			# Put it all together
			console.print(auction_panel)
			console.print(player_stats)
			console.print()
			console.print(Panel(remaining_players, title="Remaining Players", border_style="blue"))
			
			if history_items:
				console.print(Panel(auction_history, title="Recent Auction History", border_style="red"))
			
			return console.export_text()
			
		except ImportError:
			# Fallback to regular display if rich is not available
			return self.get_current_state()
	
	def step(self, action_dict):
		"""
		Expects a dictionary of actions from all agents.
		Each agent's action is assumed to be a continuous value in [0, 1] that
		is scaled to the bid range [min_bid, max_bid].
		Processes all bids simultaneously and advances the auction by one nominated player.
		"""
		#pdb.set_trace()
		#print(action_dict)
		# Process bids for each agent.
		for agent_id, action in action_dict.items():
			#action = action_dict["action"]
			# Extract agent index from string (assumes "agent_0", "agent_1", etc.)
			agent_index = int(agent_id.split("_")[1])
			min_bid = self.get_must_bid(agent_index)
			max_bid = self.get_max_bid(agent_index)

			bid = action*self.GAME_BUDGET #(max_bid - min_bid) + min_bid

			self.illegal_bid_penalties[agent_index] = min(0, bid - min_bid, max_bid - bid)

			bid = np.clip(bid, min_bid, max(min_bid, max_bid))
			if max_bid < 0:
				bid = max_bid
			# Scale the action to the bid range.
			#bid = action * (max_bid - min_bid) + min_bid
			bid = np.round(bid)
			self.bids_placed[agent_index] = bid

		# Determine the winner using your Vickrey auction rule.
		scores = self.bids_placed
		max_score = np.max(scores)
		winners = np.where(scores == max_score)[0]
		winner = np.random.choice(winners)
		
		# Fix for second-highest bid determination:
		# If there's only one bid, use that as the price
		if len(self.bids_placed) == 1:
			price_paid = self.bids_placed[0]
		# If there are multiple highest bids, the second price is the same as the highest
		elif len(winners) > 1:
			price_paid = max_score
		# Otherwise use the second highest bid
		else:
			sorted_bids = np.sort(self.bids_placed)
			price_paid = sorted_bids[-2]  # Second highest bid
		
		price_paid = max(0, price_paid)

		self.last_winner = winner
		self.last_price = price_paid
		self.prev_bids = copy.deepcopy(list(self.bids_placed))
		#print("storing old bids:", self.prev_bids)

		# Update winner's budget and history.
		self.budgets[winner] -= price_paid
		self.history[self.nominated_player] = {"price": price_paid, "winner": winner, "bids" : copy.deepcopy(self.prev_bids), "cur_budgets" : copy.deepcopy(list(self.budgets))}
		self.members_means[winner] += self.nominated_player[0]

		if self.nominated_player[1] == "G":
			self.goalies_needed[winner] -= 1
		elif self.nominated_player[1] == "D":
			self.defensemen_needed[winner] -= 1
		else:
			self.forwards_needed[winner] -= 1

		# Reset bids for the next round.
		self.bids_placed = -1 * np.ones(self.num_players)

		# Advance to the next nominated player.
		if len(self.athletes) == 0:
			self.game_over = True
		else:
			self.nominated_player = self.athletes.pop(0)

		# Build observations for all agents.
		obs = {}
		for i in range(self.num_players):
			obs[f"agent_{i}"] = self.get_observation_for_agent(i)

		# Get rewards—here we use your get_reward method.
		rewards_arr = self.get_reward()
		rewards = {}
		for i in range(self.num_players):
			rewards[f"agent_{i}"] = rewards_arr[i]

		# The environment is done if the game is over.
		dones = {"__all__": self.game_over}
		truncated = {"__all__" : False}
		info = {}
		return obs, rewards, dones, truncated, info
	
	def get_observation_for_agent(self, agent):
		"""
		Returns a normalized, flattened observation vector for the given agent.
		"""

		def position_encode(position):

			#return np.array([position == "F", position == "D", position == "G"]).astype(np.float32)
			if position == "F":
				return [1, 0, 0]
			elif position == "D":
				return [0, 1, 0]
			else:
				return [0, 0, 1]

		obs = np.concatenate([
			[self.budgets[agent] / self.GAME_BUDGET], #our remaining budget
			[x/self.GAME_BUDGET for i,x in enumerate(self.budgets) if i != agent], #opponent remaining budget
			[self.members_means[agent] / self.MAX_VAL], #our current mean
			[x/self.MAX_VAL for i,x in enumerate(self.members_means) if i != agent], #opponent current mean
			[self.forwards_needed[agent]/self.TEAM_SIZE], #our forwards needed
			[x/self.TEAM_SIZE for i,x in enumerate(self.forwards_needed) if i != agent], #opponent forwards needed
			[self.defensemen_needed[agent]/self.TEAM_SIZE], #our defensemen needed
			[x/self.TEAM_SIZE for i,x in enumerate(self.defensemen_needed) if i != agent], #opponent defensemen needed
			[self.goalies_needed[agent]/self.TEAM_SIZE], #our goalies needed
			[x/self.TEAM_SIZE for i,x in enumerate(self.goalies_needed) if i != agent], #opponent goalies needed
			[self.nominated_player[0]/self.MAX_VAL], #current athlete mean
			position_encode(self.nominated_player[1]) #current athlete position
		])

		
		obs = obs.astype(np.float32)

		return obs


	def get_reward(self):
		rewards = np.zeros(self.num_players)

		if self.game_over:
			# Final score components


			# Budget utilization penalty (lose points for leftover budget)
			#budget_left_ratios = self.budgets / self.GAME_BUDGET
			#budget_penalty = budget_left_ratios * 0.5  # Max 0.5 penalty for full budget

			final_scores = (self.members_means) # - budget_penalty
			mean_score = np.mean(final_scores)

			# Ranking reward [-1 to 1]
			#rankings = np.argsort(final_scores)[::-1]
			#ranking_reward = np.linspace(1.0, -1.0, num=self.num_players)
			#rewards += ranking_reward[rankings]

			# Relative performance bonus
			rewards += (final_scores - mean_score)/5000
			win_rewards = (rewards == np.max(rewards)).astype(np.float32)
			win_rewards /= np.sum(win_rewards)
			rewards += win_rewards
			rewards -= 1/self.num_players #make sure its 0-sum
			rewards = np.clip(rewards, -1, 1)

		return rewards
	
	def copy(self):
		return copy.deepcopy(self)