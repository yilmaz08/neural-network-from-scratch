from models import Network, Layer, Neuron
import activation_methods
import numpy as np
import json
import zlib

### EXPORTING ###
def export_network_file(network: Network, filename: str, with_weigths: bool = True, compress: bool = False) -> None:
    content = export_network(network, with_weigths, compress)
    if compress: content = zlib.compress(content)
    with open(filename, "wb") as file:
        file.write(content)

def export_network(network: Network, with_weigths: bool = True) -> dict:
    new_network = {
        "input": export_layer(network.input_layer, False, network.input_layer.names), 
        "output": export_layer(network.output_layer, with_weigths, network.output_layer.names), 
        "hidden": []
    }
    for hidden_layer in network.hidden_layers:
        new_network["hidden"].append(export_layer(hidden_layer, with_weigths, None))
    return json.dumps(new_network)

def export_layer(layer: Layer, with_weights: bool = True, names: list[str] = None) -> dict:
    new_layer = {"activation": layer.activation_function.__name__}
    if not with_weights and names is None: new_layer["neurons"] = len(layer.neurons) # Only the number of neurons
    else:
        new_neurons = []
        for index in range(len(layer.neurons)):
            # add bias and weights
            neuron = {"bias": layer.neurons[index].bias, "weights": layer.neurons[index].weights} if with_weights else {}
            # add name
            if names is not None:
                neuron["name"] = names[index] if index < len(names) else str(index)
                
            new_neurons.append(neuron)
        new_layer["neurons"] = new_neurons    
    return new_layer

""" EXAMPLE EXPORTING

my_neural_network = Network(...) # Create a network

### FILE
# Non-compressed
export_network_file(my_neural_network, "network.json", False, False) # Exports a non-compressed json file - no weights
export_network_file(my_neural_network, "networkw.json", True, False) # Exports a non-compressed json file - with weights
# Compressed
export_network_file(my_neural_network, "network.nn", False, True) # Exports a compressed json file - no weights
export_network_file(my_neural_network, "network.nnw", True, True) # Exports a compressed json file - with weights

### DICT
export_network(my_neural_network, False) # Return a dictionary - no weights
export_network(my_neural_network, True) # Return a dictionary - with weights

"""

### IMPORTING ###
def import_network_file(filename: str, compressed: bool = False) -> Network:
    try:
        with open(filename, "rb") as file:
            if compressed: data = json.loads(zlib.decompress(file.read()))
            else: data = json.loads(file.read())
        return import_network(data)
    except:
        raise Exception("Error while reading the network")

def import_network(data: dict) -> Network:
    try:
        input_layer = data["input"]
        output_layer = data["output"]
        hidden_layers = data["hidden"]

        new_hidden_layers = []
        for hidden_layer in hidden_layers:
            new_hidden_layers.append(import_layer(hidden_layer))

        return Network(
            input_layer=import_layer(input_layer),
            output_layer=import_layer(output_layer),
            hidden_layers=new_hidden_layers
        )
    except:
        raise Exception("Error while importing the network")

def import_layer(data: dict) -> Layer:
    names = []
    new_neurons = []
    if type(data["neurons"]) == int:
        names = None
        for i in range(data["neurons"]):
            new_neurons.append(Neuron(bias=None, weights=None)) # default
    else:
        for neuron in data["neurons"]:
            if "weights" in neuron and "bias" in neuron:
                new_neurons.append(Neuron(bias=neuron["bias"], weights=neuron["weights"]))
            if "name" in neuron and names is not None:
                names.append(neuron["name"])
            else:
                names = None

    new_layer = Layer(
        activation_function=activation_methods.__dict__[data["activation"]], # Get the function from the module
        neurons=new_neurons,
        names=names
    )
    return new_layer

""" EXAMPLE IMPORTING

import_network_file("network.json", False) # Imports a non-compressed json file
import_network_file("network.nn", True) # Imports a compressed json file

import_network(data) # Imports a dictionary

"""

### GENERATING ###
def neural_network(*layers: list[Layer]) -> Network:
    if len(layers) < 3: raise Exception("Network must have at least 3 layers")
    return Network(input_layer=layers[0], output_layer=layers[-1], hidden_layers=layers[1:-1])

def layer(size: int, activation_function, names: list[str] = []) -> Layer:
    return Layer(neurons=[Neuron(None, None) for i in range(size)], activation_function=activation_function, names=names)

""" EXAMPLE GENERATION

new_neural_network = neural_network(
    layer(2, activation_methods.relu, ["Input A", "Input B"]),
    layer(3, activation_methods.relu),
    layer(6, activation_methods.relu),
    layer(12, activation_methods.relu),
    layer(1, activation_methods.softmax, ["Output"])
)

# First layer is input, last layer is output and others are hidden layers in the same order as entered.
# Names can be put to identify the neurons in the layers easier.

new_neural_network.activate()
print(new_neural_network)
"""