from models import Network, Layer, Neuron
import activation_methods
import numpy as np
import json
import zlib

### EXPORTING ###
def export_network(network: Network, filename: str, with_weigths: bool = True) -> None:
    new_network = {
        "input": export_layer(network.input_layer, False, network.input_layer.names),
        "output": export_layer(network.output_layer, with_weigths, network.output_layer.names),
        "hidden": []
    }
    for hidden_layer in network.hidden_layers:
        new_network["hidden"].append(export_layer(hidden_layer, with_weigths, None))

    with open(filename, "wb") as file:
        content = json.dumps(new_network)
        compressed_content = zlib.compress(content.encode())
        file.write(compressed_content)

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


# example_network = Network(
#     input_layer=Layer(
#         neurons=[
#             Neuron(bias=0.1, weights={0: 0.1, 1: 0.2}),
#             Neuron(bias=0.2, weights={0: 0.2, 1: 0.3}),
#             Neuron(bias=0.3, weights={0: 0.3, 1: 0.4})
#         ],
#         activation_function=activation_methods.relu
#     ),
#     hidden_layers=[
#         Layer(
#             neurons=[
#                 Neuron(bias=0.1, weights={0: 0.1, 1: 0.2}),
#                 Neuron(bias=0.2, weights={0: 0.2, 1: 0.3}),
#                 Neuron(bias=0.3, weights={0: 0.3, 1: 0.4})
#             ],
#             activation_function=activation_methods.relu
#         ),
#         Layer(
#             neurons=[
#                 Neuron(bias=0.1, weights={0: 0.1, 1: 0.2}),
#                 Neuron(bias=0.2, weights={0: 0.2, 1: 0.3}),
#                 Neuron(bias=0.3, weights={0: 0.3, 1: 0.4})
#             ],
#             activation_function=activation_methods.relu
#         )
#     ],
#     output_layer=Layer(
#         neurons=[
#             Neuron(bias=0.1, weights={0: 0.1, 1: 0.2}),
#             Neuron(bias=0.2, weights={0: 0.2, 1: 0.3}),
#             Neuron(bias=0.3, weights={0: 0.3, 1: 0.4})
#         ],
#         activation_function=activation_methods.relu,
#         names=["output1", "output2", "output3"]
#     )
# )

# print(export_network(example_network, "models/example_network.nn", False))
# print(export_network(example_network, "models/example_network.nnw", True))