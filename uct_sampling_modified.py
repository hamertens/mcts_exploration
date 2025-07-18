import numpy as np
import pandas as pd
import random
import math
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.metrics import mean_squared_error
from sklearn.exceptions import ConvergenceWarning
import argparse

warnings.filterwarnings("ignore", category=ConvergenceWarning)

parser = argparse.ArgumentParser(description='UCT-based GP sampling script')
parser.add_argument('--uct_horizon', type=int, default=10, help='Rollout horizon for UCT')
parser.add_argument('--uct_simulations', type=int, default=2000, help='Number of simulations for UCT')
args = parser.parse_args()

uct_horizon = args.uct_horizon
uct_simulations = args.uct_simulations

# -------------------------------
# 1. Load the data
# -------------------------------
data = np.load("dem_box_full.npz")
slope_degrees = data["slope_degrees"]            # True slope values, shape (41, 41)
slip_coefficients = data["slip_coefficients"]      # Slip coefficients, shape (41, 41)

grid_size = slope_degrees.shape[0]  # assume square grid (e.g., 41)

# Pre-compute the highest and lowest slope values and their flattened indices.
highest_slope_value = slope_degrees.max()
highest_idx = int(np.argmax(slope_degrees))  # flattened index of highest slope

lowest_slope_value = slope_degrees.min()
lowest_idx = int(np.argmin(slope_degrees))    # flattened index of lowest slope

# -------------------------------
# 2. Initialize the slope array using full knowledge
# -------------------------------
# With full knowledge, we simply use the true slope values.
current_slope = slope_degrees.copy()

# -------------------------------
# 3. Helper functions for grid moves
# -------------------------------
# Define eight discrete actions as (delta_row, delta_col)
actions = [(-1, 0),    # up
           (-1, 1),    # up-right
           (0, 1),     # right
           (1, 1),     # right-down
           (1, 0),     # down
           (1, -1),    # left-down
           (0, -1),    # left
           (-1, -1)]   # left-up

def get_allowed_actions(state, grid_size, visited=None):
    """
    Given a state (row, col), return a list of all valid moves (from the eight directions)
    that keep the resulting cell within the grid.
    """
    allowed = []
    r, c = state
    for a in actions:
        new_r = r + a[0]
        new_c = c + a[1]
        if 0 <= new_r < grid_size and 0 <= new_c < grid_size:
            allowed.append(a)
    return allowed

def get_four_neighbors(idx, grid_size):
    """Return the 4-connected neighbor indices for a given flattened index."""
    neighbors = []
    row = idx // grid_size
    col = idx % grid_size
    if row > 0:
        neighbors.append((row - 1) * grid_size + col)
    if row < grid_size - 1:
        neighbors.append((row + 1) * grid_size + col)
    if col > 0:
        neighbors.append(row * grid_size + (col - 1))
    if col < grid_size - 1:
        neighbors.append(row * grid_size + (col + 1))
    return neighbors

# -------------------------------
# 4. UCT (Upper Confidence Trees) Functions & Node Class
# -------------------------------
class Node:
    def __init__(self, state, visited, parent=None, action_taken=None):
        """
        state: (row, col) tuple.
        visited: set of flattened indices already visited (global training set).
        parent: parent node.
        action_taken: action from parent that led here.
        """
        self.state = state
        self.visited = set(visited)
        self.parent = parent
        self.action_taken = action_taken
        self.children = {}    # action -> Node
        self.visits = 0
        self.total_reward = 0.0
        # Untried actions available from this state.
        self.untried_actions = get_allowed_actions(state, grid_size, self.visited).copy()

def best_child(node, c_param=100):
    """Select a child node using the UCT formula."""
    best_value = -float('inf')
    best_node = None
    for child in node.children.values():
        avg_reward = child.total_reward / child.visits if child.visits > 0 else 0
        exploration = c_param * math.sqrt(math.log(node.visits) / child.visits) if child.visits > 0 else float('inf')
        uct_value = avg_reward + exploration
        if uct_value > best_value:
            best_value = uct_value
            best_node = child
    return best_node

def expand(node):
    """Expand a node by selecting one untried action."""
    action_index = random.randrange(len(node.untried_actions))
    action = node.untried_actions.pop(action_index)
    new_state = (node.state[0] + action[0], node.state[1] + action[1])
    new_idx = new_state[0] * grid_size + new_state[1]
    new_visited = set(node.visited)
    new_visited.add(new_idx)
    child = Node(new_state, new_visited, parent=node, action_taken=action)
    node.children[action] = child
    return child

def tree_policy(node, depth, horizon, gamma, GP_uncertainty, u_min, u_max, global_known_set):
    """
    Traverse the tree: if the node is not fully expanded, expand it;
    otherwise choose the best child using UCT.
    Stop when the planning horizon is reached.
    """
    while depth < horizon:
        if node.untried_actions:
            return expand(node)
        else:
            if not node.children:
                return node
            node = best_child(node)
            depth += 1
    return node

def rollout_policy(state, visited, depth, horizon, gamma, GP_uncertainty, u_min, u_max, global_known_set):
    """
    Perform a random rollout (simulation) from the given state until horizon.
    At each step the immediate reward is the sum of:
      - Uncertainty-based reward.
      - A bonus of +1000 for visiting the highest slope cell (if not sampled globally yet).
      - A bonus of +1000 for visiting the lowest slope cell (if not sampled globally yet).
      - A penalty of -0.01 for visiting a cell that has already been sampled.
    """
    if depth >= horizon:
        return 0.0
    allowed = get_allowed_actions(state, grid_size, visited)
    if not allowed:
        return 0.0
    action = random.choice(allowed)
    new_state = (state[0] + action[0], state[1] + action[1])
    new_idx = new_state[0] * grid_size + new_state[1]
    
    # Apply penalty if this location was already sampled (duplicate visit).
    if new_idx in visited:
        reward_duplicate = -0.01
    else:
        reward_duplicate = 0.0
    
    # Apply bonus for the highest slope location if it hasn't been visited globally.
    if (new_idx == highest_idx) and (highest_idx not in global_known_set):
        reward_highest = 100.0
    else:
        reward_highest = 0.0

    # Apply bonus for the lowest slope location if it hasn't been visited globally.
    if (new_idx == lowest_idx) and (lowest_idx not in global_known_set):
        reward_lowest = 100.0
    else:
        reward_lowest = 0.0

    # Uncertainty-based reward (as before).
    reward_uncertainty = (GP_uncertainty[new_idx] - u_min) / (u_max - u_min) if (u_max > u_min) else 0.0
    
    reward_total = reward_uncertainty + reward_highest + reward_lowest + reward_duplicate

    new_visited = set(visited)
    new_visited.add(new_idx)
    
    return reward_total + gamma * rollout_policy(new_state, new_visited, depth + 1, horizon,
                                                   gamma, GP_uncertainty, u_min, u_max, global_known_set)

def default_policy(node, depth, horizon, gamma, GP_uncertainty, u_min, u_max, global_known_set):
    """A rollout from the node's state using the rollout_policy."""
    return rollout_policy(node.state, node.visited, depth, horizon, gamma,
                          GP_uncertainty, u_min, u_max, global_known_set)

def backup(node, reward):
    """Backpropagate the simulation reward up the tree."""
    while node is not None:
        node.visits += 1
        node.total_reward += reward
        node = node.parent

def uct_search_planning(current_state, known_set, GP_uncertainty, horizon, n_simulations, gamma, global_known_set):
    """
    Use UCT to plan from the current state (a (row, col) tuple) given known_set (sampled indices).
    GP_uncertainty is a 1D array over grid cells.
    Returns the best initial action (one of the eight moves).
    """
    visited_set = set(known_set)  # sampled indices (global training set)
    root = Node(current_state, visited_set, parent=None, action_taken=None)
    
    # Compute min and max uncertainty over unsampled cells.
    all_indices = set(range(grid_size * grid_size))
    unsampled_indices = np.array(list(all_indices - visited_set))
    if len(unsampled_indices) > 0:
        u_min_val = GP_uncertainty[unsampled_indices].min()
        u_max_val = GP_uncertainty[unsampled_indices].max()
    else:
        u_min_val, u_max_val = 0.0, 1.0

    for _ in range(n_simulations):
        node_sim = root
        depth = 0
        node_sim = tree_policy(node_sim, depth, horizon, gamma, GP_uncertainty, u_min_val, u_max_val, global_known_set)
        sim_reward = default_policy(node_sim, depth, horizon, gamma, GP_uncertainty, u_min_val, u_max_val, global_known_set)
        backup(node_sim, sim_reward)
    
    # Select best first move from root.
    best_action = None
    best_value = -float('inf')
    for action, child in root.children.items():
        if child.visits > 0:
            avg_reward = child.total_reward / child.visits
            if avg_reward > best_value:
                best_value = avg_reward
                best_action = action
    return best_action

# -------------------------------
# 5. Initialize known set and trajectory
# -------------------------------
# Choose a starting location (center of grid)
start_row, start_col = 20, 20
start_index = start_row * grid_size + start_col
# Set starting cell to true slope (full knowledge)
current_slope[start_row, start_col] = slope_degrees[start_row, start_col]
known_indices = [start_index]  # Global training set (unique indices)

# Also add one neighbor (using 4-connected neighbors) for initialization.
neighbors_of_start = get_four_neighbors(start_index, grid_size)
initial_neighbor = neighbors_of_start[0]
nbr_row = initial_neighbor // grid_size
nbr_col = initial_neighbor % grid_size
current_slope[nbr_row, nbr_col] = slope_degrees[nbr_row, nbr_col]
if initial_neighbor not in known_indices:
    known_indices.append(initial_neighbor)

# Record the trajectory as (x, y) coordinates (x = column, y = row).
trajectory = []
trajectory.append((start_col, start_row))
trajectory.append((nbr_col, nbr_row))

# -------------------------------
# 6. Error tracking lists
# -------------------------------
total_errors = []       # RMSE over entire grid per iteration
test_errors = []        # RMSE over unsampled cells per iteration
retroactive_errors = [] # Retroactive error for new sample (before sampling)

# -------------------------------
# 7. Main loop: GP training & UCT-based sampling (full knowledge of slopes)
# -------------------------------
n_iterations = 400
gamma_discount = 0.9  # Discount factor for UCT rollouts

for iteration in range(n_iterations):
    # Prepare training data using true slope values at sampled locations (unique samples).
    X_train = np.array([slope_degrees[idx // grid_size, idx % grid_size] for idx in known_indices]).reshape(-1, 1)
    y_train = np.array([slip_coefficients[idx // grid_size, idx % grid_size] for idx in known_indices])
    
    # Use the raw slope values as training inputs.
    X_all = slope_degrees.flatten().reshape(-1, 1)
    
    # Train the Gaussian Process with only one restart.
    kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))
    gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=1, normalize_y=True)
    gp.fit(X_train, y_train)
    
    # Predict on all points.
    y_pred, y_std = gp.predict(X_all, return_std=True)
    
    # Compute total RMSE over the entire grid.
    total_rmse = np.sqrt(mean_squared_error(slip_coefficients.flatten(), y_pred))
    total_errors.append(total_rmse)
    
    # Compute test RMSE over unsampled cells.
    unsampled_mask = np.ones(grid_size * grid_size, dtype=bool)
    unsampled_mask[list(known_indices)] = False
    if np.sum(unsampled_mask) > 0:
        test_rmse = np.sqrt(mean_squared_error(slip_coefficients.flatten()[unsampled_mask],
                                               y_pred[unsampled_mask]))
    else:
        test_rmse = 0.0
    test_errors.append(test_rmse)
    
    # Define GP uncertainty for UCT.
    GP_uncertainty = y_std
    
    # Define current state as the last visited cell in the trajectory.
    # (Note: trajectory stores coordinates as (x, y), so we convert to (row, col).)
    last_traj = trajectory[-1]
    current_state = (last_traj[1], last_traj[0])
    global_known_set = set(known_indices)  # Global training set
    
    # Use UCT-based planning (with our modified rewards) to choose the best initial action.
    best_action = uct_search_planning(current_state, global_known_set, GP_uncertainty,
                                      uct_horizon, uct_simulations, gamma_discount, global_known_set)
    
    if best_action is not None:
        new_state = (current_state[0] + best_action[0], current_state[1] + best_action[1])
        new_sample = new_state[0] * grid_size + new_state[1]
    else:
        # Fallback: use the first allowed move.
        allowed = get_allowed_actions(current_state, grid_size, global_known_set)
        best_action = allowed[0]
        new_state = (current_state[0] + best_action[0], current_state[1] + best_action[1])
        new_sample = new_state[0] * grid_size + new_state[1]
    
    # Compute retroactive error: prediction error on new_sample (using true slope) before sampling.
    X_new = np.array([[slope_degrees[new_state[0], new_state[1]]]])
    y_new_pred = gp.predict(X_new)
    retro_err = np.sqrt((slip_coefficients[new_state[0], new_state[1]] - y_new_pred)**2)[0]
    retroactive_errors.append(retro_err)
    
    # Add the new sample to the global training set only if it hasn't been sampled before.
    if new_sample not in global_known_set:
        known_indices.append(new_sample)
    
    # Record the trajectory (x = column, y = row).
    traj_coord = (new_state[1], new_state[0])
    trajectory.append(traj_coord)
    
    # (Optional) Print progress:
    # print(f"Iteration {iteration+1}: Total RMSE = {total_rmse:.3f}, Test RMSE = {test_rmse:.3f}, Retro Err = {retro_err:.3f}")

# -------------------------------
# 8. Save outputs to CSV files
# -------------------------------
df_errors = pd.DataFrame({
    "Iteration": range(1, n_iterations + 1),
    "Total Error": total_errors,
    "Test Error": test_errors,
    "Retroactive Error": retroactive_errors
})
df_errors.to_csv(f"output_data/uct_sampling_errors_h{uct_horizon}_s{uct_simulations}.csv", index=False)

df_traj = pd.DataFrame(trajectory, columns=["X", "Y"])
df_traj["Step"] = range(1, len(trajectory) + 1)
df_traj.to_csv(f"output_data/trajectory_h{uct_horizon}_s{uct_simulations}.csv", index=False)