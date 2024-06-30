import numpy as np
class Neuron: 
    activation = 0
    def __init__(self, bias: float, weights: dict[int, float] = {}): self.bias, self.weights = bias, weights
    def activate(self, inputs: dict = None) -> None:
        if self.bias is None:
            self.activation = 0
            return
        self.activation = self.bias
        if inputs is None: return
        for key, value in inputs.items():
            self.activation += (self.weights[key] if key in self.weights else 0) * value
    
class Layer:
    def __init__(self, neurons: list[Neuron], activation_function, names: list[str] = []):
        self.neurons = neurons
        self.activation_function = activation_function
        self.names = names
    
    def activate(self, previous_layer = None):
        previous_values = previous_layer.get_neuron_activations() if previous_layer is not None else None
        for neuron in self.neurons: neuron.activate(previous_values)

    def get_neuron_activations(self) -> dict[int, float]:
        neurons_data = {}
        all_linear_activations_sum = sum([neuron.activation for neuron in self.neurons])
        all_exp_activations_sum = sum([np.exp(neuron.activation) for neuron in self.neurons])

        for index in range(len(self.neurons)):
            neurons_data[index] = self.activation_function(
                linear_activation           = self.neurons[index].activation,
                all_linear_activations_sum  = all_linear_activations_sum,
                all_exp_activations_sum     = all_exp_activations_sum
            )
        return neurons_data

    def output(self, names: list[str]) -> dict[str, float]: # Match the names with the neurons
        neurons = self.get_neuron_activations() # to apply the activation function
        return {(names[i] if i < len(names) else str(i)): neurons[i] for i in range(len(neurons))} # sphagetti... sorry...
    
    def __str__(self):
        string = ""
        for key, value in self.output(self.names).items():
            string += f"{key}: {value:.4f} | "
        return string[:-3] + "\n"
    
class Network:
    def __init__(self, input_layer: Layer, hidden_layers: list[Layer], output_layer: Layer):
        self.input_layer = input_layer
        self.hidden_layers = hidden_layers
        self.output_layer = output_layer

    def activate(self):
        self.input_layer.activate()
        previous_layer = self.input_layer
        for layer in self.hidden_layers:
            layer.activate(previous_layer)
            previous_layer = layer
        self.output_layer.activate(previous_layer)

    def output(self) -> dict[str, float]:
        return self.output_layer.__str__()
    
    def __str__(self):
        string = self.input_layer.__str__()
        for layer in self.hidden_layers: string += layer.__str__()
        string += self.output_layer.__str__()
        return string