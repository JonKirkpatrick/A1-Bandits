# bandit_module.py
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import matplotlib.pyplot as plt
from non_stationary import *

class NormalBandit:
    """
    A class to represent an individual bandit with a normal reward distribution.

    ...
    
    Attributes
    ----------
    mean : float
        the average reward to be expected from this bandit
    variance : float
        the variance associated with the normal reward distribution
    verbose : boolean
        optional parameter used to provide greater detail to stdout

    Methods
    -------
    pull():
        outputs a randomly generated reward based on the bandit's distribution
    """

    def __init__(self, means, variance=1, verbose=True):
        """
        Constructs all the necessary attributes for the NormalBandit object.

        Parameters
        ----------
            mean : float
                the average reward to be expected from this bandit
            variance : float
                the variance associated with the normal reward distribution
            verbose : boolean
                optional parameter used to provide greater detail to stdout
        """
        self.means = means
        self.variance = variance
        self.verbose = verbose

    def pull(self, t):
        """
        Operates this instance of NormalBandit.

        A random reward is generated from its defined normal reward distribution,
        which may vary depending on the time step t.

        Parameters
        ----------
        t : int
            The current time step.

        Returns
        -------
        float
            A float value representing the reward.
        """

        # Get the mean for the current time step t
        mean = self.means[t]

        # Set random seed to None to ignore any previous seed values
        np.random.seed(None)

        return np.random.normal(mean, self.variance)

class BanditsGame:
    """
    Creates an instance of the bandits testbed.

    Generates K NormalBandit objects to populate the BanditsGame

    ...
    Attributes
    ----------
    K : int
        the number of NormalBandits to populate
    verbose : boolean
        optional parameter used to provide greater detail to stdout

    Methods
    -------
    run_greedy(T):
        runs an instance of the BanditsGame using the greedy algorithm

    run_epsilon_greedy(T, epsilon):
        runs an instance of the BanditsGame using the epsilon-greedy algorithm

    run_optimistic_initial_values(T, initial_value):
        runs an instance of the BanditsGame using the optimistic-greedy algorithm

    run_gradient_bandit(T, alpha):
        runs an instance of the BanditsGame using the gradient-bandit algorithm
    """

    def __init__(self, K, T, seed=None, non_stationary_type="stationary", verbose=True):
        """
        Constructs all the necessary attributes for the BanditsGame object.

        Parameters
        ----------
            K : int
                the number of NormalBandit objects to include in the game
            seed : int or None
                optional parameter used as seed for random number generation
            verbose : boolean
                optional parameter used to provide greater detail to stdout
        """

        self.K = K
        self.T = T
        self.seed = seed
        self.non_stationary_type = non_stationary_type
        self.verbose = verbose

        if seed is not None:
            np.random.seed(seed)

        self.bandits = [
            NormalBandit(means=generate_means(np.random.randn(), T, non_stationary_type), verbose=verbose) 
            for _ in range(K)
        ]
        
        if self.non_stationary_type == "abrupt":
            bandits = abrupt_change(self.bandits)

        self.optimal_action = [np.argmax([bandit.means[t] for bandit in self.bandits]) for t in range(T)]
        
    def run_greedy(self, T):
        """
        Runs an instance of the BanditsGame using the greedy algorithm.

        Parameters
        ----------
        T : int
            The number of steps to run the algorithm.

        Returns
        -------
        numpy.ndarray
            An array of rewards obtained at each step.
        numpy.ndarray
            An array indicating whether the optimal action was chosen at each step.
        """

        Q = np.zeros(self.K)  # Initial action-value estimates
        N = np.zeros(self.K)  # Action counts
        rewards = np.zeros(T)
        optimal_action_counts = np.zeros(T)
        
        for t in range(T):
            action = np.argmax(Q)
            reward = self.bandits[action].pull(t)
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
            rewards[t] = reward
            optimal_action_counts[t] = 1 if action == self.optimal_action[t] else 0
            
            if self.verbose:
                print(f"t={t}, Action={action}, Reward={reward:.2f}, Q={Q}, N={N}")
        
        return rewards, optimal_action_counts

    def run_epsilon_greedy(self, T, epsilon):
        """
        Runs an instance of the BanditsGame using the epsilon-greedy algorithm.

        Parameters
        ----------
        T : int
            The number of steps to run the algorithm.
        epsilon : float
            The probability of exploring a random action.

        Returns
        -------
        numpy.ndarray
            An array of rewards obtained at each step.
        numpy.ndarray
            An array indicating whether the optimal action was chosen at each step.
        """

        Q = np.zeros(self.K)  # Initial action-value estimates
        N = np.zeros(self.K)  # Action counts
        rewards = np.zeros(T)
        optimal_action_counts = np.zeros(T)
        np.random.seed(None)
        
        for t in range(T):
            if np.random.rand() < epsilon:
                action = np.random.choice(self.K)
            else:
                action = np.argmax(Q)
            reward = self.bandits[action].pull(t)
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
            rewards[t] = reward
            optimal_action_counts[t] = 1 if action == self.optimal_action[t] else 0

            if self.verbose:
                print(f"t={t}, Action={action}, Reward={reward:.2f}, Q={Q}, N={N}")

        return rewards, optimal_action_counts

    def run_optimistic_initial_values(self, T, initial_value):
        """
        Runs an instance of the BanditsGame using the optimistic-greedy algorithm.

        Parameters
        ----------
        T : int
            The number of steps to run the algorithm.
        initial_value : float
            The initial optimistic value for all actions.

        Returns
        -------
        numpy.ndarray
            An array of rewards obtained at each step.
        numpy.ndarray
            An array indicating whether the optimal action was chosen at each step.
        """

        Q = np.full(self.K, float(initial_value))  # Optimistic initial values
        N = np.zeros(self.K)  # Action counts
        rewards = np.zeros(T)
        optimal_action_counts = np.zeros(T)
        
        for t in range(T):
            action = np.argmax(Q)
            reward = self.bandits[action].pull(t)
            N[action] += 1
            Q[action] += (reward - Q[action]) / N[action]
            rewards[t] = reward
            optimal_action_counts[t] = 1 if action == self.optimal_action[t] else 0
            
            if self.verbose:
                print(f"t={t}, Action={action}, Reward={reward:.2f}, Q={Q}, N={N}")
        
        return rewards, optimal_action_counts

    def run_gradient_bandit(self, T, alpha):
        """
        Runs an instance of the BanditsGame using the gradient-bandit algorithm.

        Parameters
        ----------
        T : int
            The number of steps to run the algorithm.
        alpha : float
            The learning rate for updating preferences.

        Returns
        -------
        numpy.ndarray
            An array of rewards obtained at each step.
        numpy.ndarray
            An array indicating whether the optimal action was chosen at each step.
        """

        H = np.zeros(self.K)  # Preferences
        pi = np.ones(self.K) / self.K  # Action probabilities
        rewards = np.zeros(T)
        optimal_action_counts = np.zeros(T)
        np.random.seed(None)
        
        for t in range(T):
            action = np.random.choice(self.K, p=pi)
            reward = self.bandits[action].pull(t)
            average_reward = rewards[:t].mean() if t > 0 else 0
            
            for a in range(self.K):
                if a == action:
                    H[a] += alpha * (reward - average_reward) * (1 - pi[a])
                else:
                    H[a] -= alpha * (reward - average_reward) * pi[a]
            
            pi = np.exp(H) / np.sum(np.exp(H))
            rewards[t] = reward
            optimal_action_counts[t] = 1 if action == self.optimal_action[t] else 0
            
            if self.verbose:
                print(f"t={t}, Action={action}, Reward={reward:.2f}, H={H}, pi={pi}")
        
        return rewards, optimal_action_counts

def run_simulation(K, T, n_problems, algorithm, seed=None, non_stationary_type="stationary", **kwargs):
    """
    Runs multiple instances of the BanditsGame and computes average rewards and optimal action proportions.

    Parameters
    ----------
    K : int
        The number of bandits.
    T : int
        The number of steps to run each game.
    n_problems : int
        The number of independent problems to simulate.
    algorithm : str
        The algorithm to use ('greedy', 'epsilon_greedy', 'optimistic_initial_values', 'gradient_bandit').
    seed : int, optional
        Random seed for reproducibility (default is None).
    non_stationary_type : str, optional
        Type of non-stationary behavior ('stationary' or specified in non_stationary_types).
    **kwargs : dict
        Additional parameters specific to the algorithm.

    Returns
    -------
    numpy.ndarray
        Average rewards across all problems at each step.
    numpy.ndarray
        Proportion of times the optimal action was chosen across all problems at each step.
    """

    if seed == None:
        seed = random.randint(0, 2**32 - 1)

    rewards_all = np.zeros((n_problems, T))
    optimal_action_counts_all = np.zeros((n_problems, T))
    
    for i in range(n_problems):
        game = BanditsGame(K, T, seed=seed, non_stationary_type=non_stationary_type, verbose=False)
        
        if algorithm == "greedy":
            rewards, optimal_action_counts = game.run_greedy(T)
        elif algorithm == "epsilon_greedy":
            rewards, optimal_action_counts = game.run_epsilon_greedy(T, kwargs["epsilon"])
        elif algorithm == "optimistic_initial_values":
            rewards, optimal_action_counts = game.run_optimistic_initial_values(T, kwargs["initial_value"])
        elif algorithm == "gradient_bandit":
            rewards, optimal_action_counts = game.run_gradient_bandit(T, kwargs["alpha"])
        
        rewards_all[i, :] = rewards
        optimal_action_counts_all[i, :] = optimal_action_counts
    
    average_rewards = rewards_all.mean(axis=0)
    rewards_per_trial = np.sum(rewards_all, axis=1)
    optimal_action_proportions = optimal_action_counts_all.mean(axis=0)
    
    return average_rewards, optimal_action_proportions, rewards_per_trial

def plot_results(avg_rewards_greedy, avg_rewards_epsilon_greedy, avg_rewards_optimistic, avg_rewards_gradient,
                 opt_action_props_greedy, opt_action_props_epsilon_greedy, opt_action_props_optimistic, opt_action_props_gradient,
                 best_epsilon, best_alpha, avg_rewards_per_trial_greedy, avg_rewards_per_trial_optimistic, 
                 avg_rewards_per_trial_epsilon_greedy, avg_rewards_per_trial_gradient):
    """
    Plots the results of different algorithms.

    Parameters
    ----------
    avg_rewards_greedy : numpy.ndarray
        Average rewards obtained using the greedy algorithm.
    avg_rewards_epsilon_greedy : numpy.ndarray
        Average rewards obtained using the epsilon-greedy algorithm.
    avg_rewards_optimistic : numpy.ndarray
        Average rewards obtained using the optimistic initial values algorithm.
    avg_rewards_gradient : numpy.ndarray
        Average rewards obtained using the gradient bandit algorithm.
    opt_action_props_greedy : numpy.ndarray
        Proportion of times the optimal action was chosen using the greedy algorithm.
    opt_action_props_epsilon_greedy : numpy.ndarray
        Proportion of times the optimal action was chosen using the epsilon-greedy algorithm.
    opt_action_props_optimistic : numpy.ndarray
        Proportion of times the optimal action was chosen using the optimistic initial values algorithm.
    opt_action_props_gradient : numpy.ndarray
        Proportion of times the optimal action was chosen using the gradient bandit algorithm.
    best_epsilon : float
        The best epsilon value found during parameter tuning.
    best_alpha : float
        The best alpha value found during parameter tuning.
    """

    # Plot average rewards
    plt.figure(figsize=(12, 6))
    plt.plot(avg_rewards_greedy, label='Greedy')
    plt.plot(avg_rewards_epsilon_greedy, label=f'Epsilon-Greedy (ε={best_epsilon:.2f})')
    plt.plot(avg_rewards_optimistic, label='Optimistic Initial Values')
    plt.plot(avg_rewards_gradient, label=f'Gradient Bandit (α={best_alpha:.2f})')
    plt.xlabel('Steps')
    plt.ylabel('Average Reward')
    plt.legend()
    plt.title('Average Reward Comparison')
    plt.show()

    # Plot % optimal action
    plt.figure(figsize=(12, 6))
    plt.plot(opt_action_props_greedy, label='Greedy')
    plt.plot(opt_action_props_epsilon_greedy, label=f'Epsilon-Greedy (ε={best_epsilon:.2f})')
    plt.plot(opt_action_props_optimistic, label='Optimistic Initial Values')
    plt.plot(opt_action_props_gradient, label=f'Gradient Bandit (α={best_alpha:.2f})')
    plt.xlabel('Steps')
    plt.ylabel('% Optimal Action')
    plt.legend()
    plt.title('% Optimal Action Comparison')
    plt.show()

    # Boxplot of total rewards
    
    rewards_data = [avg_rewards_per_trial_greedy, avg_rewards_per_trial_optimistic, 
                 avg_rewards_per_trial_epsilon_greedy, avg_rewards_per_trial_gradient]
    labels = ['Greedy', f'Epsilon-Greedy (ε={best_epsilon:.2f})', 'Optimistic Initial Values', 
            f'Gradient Bandit (α={best_alpha:.2f})']
    
    plt.figure(figsize=(12, 6))
    plt.boxplot(rewards_data, labels=labels)
    plt.ylabel('Total Rewards')
    plt.title('Boxplot of Total Rewards for Different Algorithms')
    plt.show()
