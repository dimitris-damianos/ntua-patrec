import numpy as np
from pomegranate.distributions import Normal
from pomegranate.gmm import GeneralMixtureModel
from pomegranate.hmm import DenseHMM

from parser import parser
from parser import make_scale_fn
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix
from plot_confusion_matrix import plot_confusion_matrix

from sklearn.metrics import accuracy_score


# TODO: YOUR CODE HERE
# Play with diffrent variations of parameters in your experiments
n_states = 2  # the number of HMM states
n_mixtures = 3  # the number of Gaussians
gmm = True  # whether to use GMM or plain Gaussian
covariance_type = "diag"  # Use diagonal covariange


# Gather data separately for each digit
def gather_in_dic(X, labels, spk):
    dic = {}
    for dig in set(labels):
        x = [X[i] for i in range(len(labels)) if labels[i] == dig]
        lengths = [len(i) for i in x]
        y = [dig for _ in range(len(x))]
        s = [spk[i] for i in range(len(labels)) if labels[i] == dig]
        dic[dig] = (x, lengths, y, s)
    return dic


def create_data():
    X, X_test, y, y_test, spk, spk_test = parser("./recordings", n_mfcc=13)

    # TODO: YOUR CODE HERE
    # By default stratified
    (
        X_train,
        X_val,
        y_train,
        y_val,
        spk_train,
        spk_val
    ) = train_test_split(X, y, spk, test_size=0.2) # split X into a 80/20 train validation split
    
    # Scale: 0 mean and unit var
    print("using all data to calculate normalization statistics")
    scale_fn = make_scale_fn(X_train + X_val + X_test)
 
    X_train = scale_fn(X_train)
    X_val = scale_fn(X_val)
    X_test = scale_fn(X_test)
    
    train_dic = gather_in_dic(X_train, y_train, spk_train)
    val_dic = gather_in_dic(X_val, y_val, spk_val)
    test_dic = gather_in_dic(X_test, y_test, spk_test)
    labels = list(set(y_train))

    return train_dic, y_train, val_dic, y_val, test_dic, y_test, labels


def initialize_and_fit_gmm_distributions(X, n_states, n_mixtures):
    # TODO: YOUR CODE HERE
    dists = []
    for _ in range(n_states):
        distributions = [Normal(covariance_type = "diag") for _ in range(n_mixtures)]  # n_mixtures gaussian distributions
        a = GeneralMixtureModel(distributions, verbose=True).fit(
            np.concatenate(X)
        )  # Concatenate all frames from all samples into a large matrix
        dists.append(a)
    return dists


def initialize_and_fit_normal_distributions(X, n_states):
    dists = []
    for _ in range(n_states):
        # TODO: YOUR CODE HERE
        d = Normal(covariance_type = "diag").fit(
            np.concatenate(X)
        )# Fit a normal distribution on X
        dists.append(d)
    return dists


def initialize_transition_matrix(n_states):
    # TODO: YOUR CODE HERE
    A = np.zeros((n_states, n_states), dtype = np.float32)
    
    # According to instructions
    for i in range(n_states):
        for j in range(n_states):
            if i == j :
                A[i][j] = 0.3
            if i == (j - 1) :
                A[i][j] = 0.7
    
    A[n_states - 1][n_states - 1] = 1
    
    return A


def initialize_starting_probabilities(n_states):
    # TODO: YOUR CODE HERE
    start_probs = np.zeros(n_states, dtype = np.float32)
    start_probs[0] = 1 # Can only start from first state
    return start_probs


def initialize_end_probabilities(n_states):
    # TODO: YOUR CODE HERE
    end_probs = np.zeros(n_states, dtype = np.float32)
    end_probs[n_states - 1] = 1 # Can only end in end state 
    return end_probs


def train_single_hmm(X, emission_model, digit, n_states):
    A = initialize_transition_matrix(n_states)
    start_probs = initialize_starting_probabilities(n_states)
    end_probs = initialize_end_probabilities(n_states)
    data = [x.astype(np.float32) for x in X]
    
    print(A)
    print(start_probs)
    print(end_probs)
    
    model = DenseHMM(
        distributions=emission_model,
        edges=A,
        starts=start_probs,
        ends=end_probs,
        verbose=True,
    ).fit(data)
    return model


def train_hmms(train_dic, labels):
    hmms = {}  
    
    # Train one hmm for each digit
    for dig in labels:
        X, _, _, _ = train_dic[dig]
        # TODO: YOUR CODE HERE
        print(dig)
        print(len(X))
        print(X[0].shape)
        
        emission_model = initialize_and_fit_gmm_distributions(X, n_states, n_mixtures)
        hmms[dig] = train_single_hmm(X, emission_model, dig, n_states)
    return hmms


def evaluate(hmms, dic, labels):
    pred, true = [], []
    for dig in labels:
        X, _, _, _ = dic[dig]
        for sample in X:
            ev = [0] * len(labels)
            sample = np.expand_dims(sample, 0)
            for digit, hmm in hmms.items():
                # TODO: YOUR CODE HERE
                # Calculate the logprob output for each hmm when given the test sample
                logp = hmm.log_probability(sample) 
                ev[digit] = logp

            # TODO: YOUR CODE HERE
            predicted_digit = ev.index(max(ev)) # Calculate the most probable digit
            pred.append(predicted_digit)
            true.append(dig)
    return pred, true

# Create data
train_dic, y_train, val_dic, y_val, test_dic, y_test, labels = create_data()
# Train
hmms = train_hmms(train_dic, labels)
labels = list(set(y_train))

# TODO: YOUR CODE HERE
# Calculate and print the accuracy score on the validation and the test sets
# Plot the confusion matrix for the validation and the test set

# Validation
pred_val, true_val = evaluate(hmms, val_dic, labels)
accuracy = accuracy_score(true_val, pred_val)
print(f"Model's accuracy in dev: {accuracy}")

plot_confusion_matrix(confusion_matrix(true_val, pred_val) , labels,
                          normalize=False,
                          title='Confusion matrix for dev',
                          cmap=plt.cm.Blues)

# Test
pred_test, true_test = evaluate(hmms, test_dic, labels)
accuracy = accuracy_score(true_test, pred_test)
print(f"Model's accuracy in test: {accuracy}")

plot_confusion_matrix(confusion_matrix(true_test, pred_test) , labels,
                          normalize=False,
                          title='Confusion matrix for test',
                          cmap=plt.cm.Blues)


