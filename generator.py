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

def export_network(network: Network, with_weigths: bool = True) -> None:
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

### IMPORTING ###
def import_network_file(filename: str, compressed: bool = False) -> Network:
    try:
        with open(filename, "rb") as file:
            if compressed:
                content = zlib.decompress(file.read())
                data = json.loads(content)
            else:
                data = json.load(file)
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