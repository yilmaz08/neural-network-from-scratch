import mnist

# Load the MNIST dataset
mnist = mnist.MNIST(
    "data/train-images.idx3-ubyte",
    "data/train-labels.idx1-ubyte",
    "data/t10k-images.idx3-ubyte",
    "data/t10k-labels.idx1-ubyte"
)
(train_images, train_labels), (test_images, test_labels) = mnist.load_dataset()