# bandit_interface.py
from k_armed_bandits import *

def main():
    K = 10
    T = 1000
    n_problems = 1000
    seed = 57
    non_stationary_type = "stationary"
    epsilon = 0.07
    alpha = 0.1174
    initial_value = 5

    # Greedy
    avg_rewards_greedy, opt_action_props_greedy, avg_rewards_per_trial_greedy = run_simulation(K, T, n_problems, "greedy", seed, non_stationary_type=non_stationary_type)

    # Optimistic Initial Values
    avg_rewards_optimistic, opt_action_props_optimistic, avg_rewards_per_trial_optimistic = run_simulation(K, T, n_problems, "optimistic_initial_values", seed, non_stationary_type=non_stationary_type, initial_value=initial_value)

    #Epsilon Greedy
    avg_rewards_epsilon_greedy, opt_action_props_epsilon_greedy, avg_rewards_per_trial_epsilon_greedy = run_simulation(K, T, n_problems, "epsilon_greedy", seed, non_stationary_type=non_stationary_type, epsilon=epsilon)

    #Gradient
    avg_rewards_gradient, opt_action_props_gradient, avg_rewards_per_trial_gradient = run_simulation(K, T, n_problems, "gradient_bandit", seed, non_stationary_type=non_stationary_type, alpha=alpha)

    plot_results(avg_rewards_greedy, avg_rewards_epsilon_greedy, avg_rewards_optimistic, avg_rewards_gradient,
                 opt_action_props_greedy, opt_action_props_epsilon_greedy, opt_action_props_optimistic, opt_action_props_gradient,
                 epsilon, alpha, avg_rewards_per_trial_greedy, avg_rewards_per_trial_optimistic, avg_rewards_per_trial_epsilon_greedy, avg_rewards_per_trial_gradient)

if __name__ == "__main__":
    main()
