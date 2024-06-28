import numpy as np

### CONSTANTS ###
LEAKY_RELU_ALPHA = 0.1


### FUNCTIONS ###
def relu(linear_activation: float, all_linear_activations_sum: float, all_exp_activations_sum: float) -> float:
    return max(0, linear_activation)

def sigmoid(linear_activation: float, all_linear_activations_sum: float, all_exp_activations_sum: float) -> float:
    return 1 / (1 + np.exp(-linear_activation))

def softmax(linear_activation: float, all_linear_activations_sum: float, all_exp_activations_sum: float) -> float:
    return np.exp(linear_activation) / all_exp_activations_sum

def tanh(linear_activation: float, all_linear_activations_sum: float, all_exp_activations_sum: float) -> float:
    return np.tanh(linear_activation)

def linear(linear_activation: float, all_linear_activations_sum: float, all_exp_activations_sum: float) -> float:
    return linear_activation

def leaky_relu(linear_activation: float, all_linear_activations_sum: float, all_exp_activations_sum: float) -> float:
    return max(LEAKY_RELU_ALPHA * linear_activation, linear_activation)