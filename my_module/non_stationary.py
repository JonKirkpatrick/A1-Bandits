# non_stationary.py
import numpy as np

def drift_change(previous_mean, initial_mean):
    """
    Applies a small random drift to the mean.

    Parameters
    ----------
    previous_mean : float
        The mean at the previous time step.
    initial_mean : float
        The initial mean of the bandit.

    Returns
    -------
    float
        The updated mean after applying the drift.
    """
    np.random.seed(None)
    return previous_mean + np.random.normal(0, 0.001)

def mean_reverting_change(previous_mean, initial_mean):
    """
    Applies a mean-reverting change to the mean.

    Parameters
    ----------
    previous_mean : float
        The mean at the previous time step.
    initial_mean : float
        The initial mean of the bandit.

    Returns
    -------
    float
        The updated mean after applying the mean-reverting change.
    """
    np.random.seed(None)
    return previous_mean + 0.5 * (initial_mean - previous_mean) + np.random.normal(0, 0.01)

def abrupt_change(bandits):
    """
    Permutes the means corresponding to each bandit.

    At each time step, with probability 0.005, permutes the means corresponding to each of the reward distributions.

    Parameters
    ----------
    bandits : list of NormalBandit
        List of bandits whose means need to be permuted.
    """

    np.random.seed(None)    
    num_bandits = len(bandits)
    num_means = len(bandits[0].means)
    means_matrix = np.array([bandit.means for bandit in bandits])

    for t in range(num_means):
        if np.random.rand() < 0.005:
            np.random.shuffle(means_matrix[:, t:])

    # Update the means for each bandit using the shuffled means_matrix
    for i, bandit in enumerate(bandits):
        bandit.means = means_matrix[i].tolist()
    return bandits

def generate_means(initial_mean, T, non_stationary_type):
    """
    Generates a sequence of means for the given non-stationary type.

    Parameters
    ----------
    initial_mean : float
        The initial mean of the bandit.
    T : int
        The number of time steps.
    non_stationary_type : str
        The type of non-stationary behavior ('drift', 'mean_reverting', 'abrupt').

    Returns
    -------
    numpy.ndarray
        An array of means for each time step.
    """
    np.random.seed(None)
    means = np.zeros(T)
    means[0] = initial_mean
    for t in range(1, T):
        if non_stationary_type in ["drift", "mean_reverting"]:
            means[t] = non_stationary_types[non_stationary_type](means[t-1], initial_mean)
        elif non_stationary_type == "abrupt":
            means[t] = means[0]
        else:
            means[t] = means[0]  # No change for unrecognized types
    return means

# Dictionary to map non-stationary types to their functions
non_stationary_types = {
    "drift": drift_change,
    "mean_reverting": mean_reverting_change,
    "abrupt": abrupt_change
}
