import numpy as np
import pandas as pd
import pymc as pm
import matplotlib.pyplot as plt
from hmmlearn.hmm import CategoricalHMM

# loading the data into pandas dataFrame
df = pd.read_csv(pm.get_data("deaths_and_temps_england_wales.csv"))

# use for discretise the death and temp
def discretise(series, n_bins):
    # use panda qcut to quantise the numerical value to equally-sized bins
    categories, bins = pd.qcut(series, q=n_bins, labels=False, retbins=True, duplicates="drop")
    return categories, bins

# Supervised learning since latent state is given (i.e. tempeature states)
# I searched it up, smoothing is the Laplace smoothing, 
# which enforce no probabilty = 0 in the starting state 
def hmm_1(temp_states, death_observations, n_states, n_death_levels, smoothing=1.0):
    
    # convert into numpy arrays
    temp_states = np.asarray(temp_states, dtype=int)
    death_observations = np.asarray(death_observations, dtype=int)

    # initialise the initial state of probabilities
    pi_counts = np.zeros(n_states, dtype=float)
    pi_counts[temp_states[0]] += 1.0 
    
    # transition counts A[i, j], count of state i to state j
    A_counts = np.zeros((n_states, n_states), dtype=float)
    
    for t in range(len(temp_states) - 1):
        i = temp_states[t]
        j = temp_states[t+1]
        A_counts[i, j] += 1.0
        
    # emisson counts E[i, k], count of observation k in state i
    E_counts = np.zeros((n_states, n_death_levels), dtype=float)
    
    for t in range(len(temp_states)):
        i = temp_states[t]
        k = death_observations[t]
        E_counts[i, k] += 1.0
        
    # use all the counts from pi, A, E to construct the probability distribution
    # for each row they sum up to the probability of 1
    pi = (pi_counts + smoothing) / (pi_counts.sum() + smoothing * n_states)
    A = (A_counts + smoothing) / (A_counts.sum(axis=1, keepdims=True) + smoothing * n_states)
    E = (E_counts + smoothing) / (E_counts.sum(axis=1, keepdims=True) + smoothing * n_death_levels)
    
    # finally build an HMM model using hmmlearn
    model = CategoricalHMM(
        n_components=n_states,
        n_features=n_death_levels,
        random_state=42,
    )
    
    # set model params
    model.startprob_ = pi
    model.transmat_ = A
    model.emissionprob_ = E
    
    return model

# Unsupervised learning, unknown latent state
def hmm_2(death_observations, n_component, n_death_levels, n_iter=100):
    model = CategoricalHMM(
        n_components=n_component,
        n_features=n_death_levels,
        n_iter=n_iter,
        random_state=42
    )
    
    # hmm wants the shape of (n_sample, 1), so need to flatten it 
    observations = np.asarray(death_observations, dtype=int).reshape(-1, 1)
    
    model.fit(observations)
    return model

# bins for temperature can be customised
temp_states, temp_bins = discretise(df["temp"], n_bins=3)
# bins for deaths observation "low", "medium", "high"
death_observations, deaths_bins = discretise(df["deaths"], n_bins=3)

# find the number of temp states and number level of death
n_temp_states = len(np.unique(temp_states))
n_death_levels = len(np.unique(death_observations))

# use HMM1 for supervised learning (temp as the latent variable)
hmm1 = hmm_1(temp_states, death_observations, n_temp_states, n_death_levels)

print("=============== Supervised HMM (HMM1) Parameters ===============")
print(f"Start Probability:\n {hmm1.startprob_}")
print(f"Transition Matrix:\n {hmm1.transmat_}")
print(f"Emission Matrix:\n {hmm1.emissionprob_}")

# use HMM2 for unsupervised learning (unknown latent variable)
hmm2 = hmm_2(death_observations, n_temp_states, n_death_levels)
print("=============== Unsupervised HMM (HMM2) Parameters ===============")

print(f"Start Probability:\n {hmm2.startprob_}")
print(f"Transition Matrix:\n {hmm2.transmat_}")
print(f"Emission Matrix:\n {hmm2.emissionprob_}")

# Sample sequences of deaths values from each HMM
hmm1_observations, hmm1_states = hmm1.sample(len(death_observations))
hmm2_observations, hmm2_states = hmm2.sample(len(death_observations))

# flatten into 1D array for normalise counts
hmm1_observations = hmm1_observations.flatten()
hmm2_observations = hmm2_observations.flatten()

# Normalised frequencies for 0,1,2
def normalised_counts(obs, n_levels):
    # count for each bin, how many occurence in observations
    counts = np.bincount(obs, minlength=n_levels)
    return counts / counts.sum()

# normalise the counts for frequency plot
true_freq = normalised_counts(death_observations, n_death_levels)
hmm1_freq = normalised_counts(hmm1_observations, n_death_levels)
hmm2_freq = normalised_counts(hmm2_observations, n_death_levels)

# Plot function for the Marginal death distribution
def plot_freq_bar(true_freq, hmm1_freq, hmm2_freq):
    # 3 categories for death level (i.e. low, medium, high)
    labels = ['Low', 'Medium', 'High']
    x = np.arange(n_death_levels)
    width = 0.25
    
    # create a frequency bar plot
    plt.figure()
    # add offset to each bar plot for 3 bars (true, HMM1, HMM2)
    plt.bar(x - width, true_freq,  width, label='True')
    plt.bar(x,         hmm1_freq, width, label='HMM1')
    plt.bar(x + width, hmm2_freq, width, label='HMM2')
    
    plt.xticks(x, labels)
    plt.ylabel('Frequency')
    
    plt.title('Death-Level Distributions: Real vs HMM1 vs HMM2')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig("sequential_data_figures/hmm_samples_comparison.png", dpi=300, bbox_inches='tight')
    plt.show()
    
plot_freq_bar(true_freq, hmm1_freq, hmm2_freq)

# Use for the heatmap for the transitions of the death levels
def empircal_transition_matrix(observations, n_death_levels):
    # count how many transitions from death level i to death level j
    counts = np.zeros((n_death_levels, n_death_levels), dtype=float)
    
    for t in range(len(observations) - 1):
        i = observations[t]
        j = observations[t+1]
        counts[i, j] += 1.0
        
    # Row-normalise to get probabilities
    row_sums = counts.sum(axis=1, keepdims=True)
    transition_probs = counts / row_sums
    return transition_probs
    
# plot a 3x3 transition matrix as a heatmap
def plot_transition_heatmap(matrix, title, model_name):
    plt.figure()
    plt.imshow(matrix, aspect='equal')
    plt.colorbar(label='Transition probability')
    
    # Create a 3x3 grid 
    plt.xticks(range(matrix.shape[1]), ['Low', 'Med', 'High'])
    plt.yticks(range(matrix.shape[0]), ['Low', 'Med', 'High'])
    
    # label the probabilities
    plt.xlabel('Next death level')
    plt.ylabel('Current death level')
    plt.title(title)
    
    plt.tight_layout()
    plt.savefig(f"sequential_data_figures/{model_name}_transitions.png", dpi=300, bbox_inches='tight')
    plt.show()

# find the empircal transition matrix first
true_matrix = empircal_transition_matrix(death_observations, n_death_levels)
hmm1_matrix = empircal_transition_matrix(hmm1_observations, n_death_levels)
hmm2_matrix = empircal_transition_matrix(hmm2_observations, n_death_levels)

# use the empircal transition matrix to plot the heatmap of each model to
# visualise how the order differs from each model to the true one
plot_transition_heatmap(true_matrix, title="True Observations Transitions", model_name="true")
plot_transition_heatmap(hmm1_matrix, title="HMM1 Sample Obvservations Transitions", model_name="hmm1")
plot_transition_heatmap(hmm2_matrix, title="HMM2 Sample Obvservations Transitions", model_name="hmm2")