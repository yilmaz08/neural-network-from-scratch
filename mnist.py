import struct
import numpy as np
from array import array

class MNIST:
    def __init__(
            self,
            trainging_images_path: str,
            training_labels_path: str,
            test_images_path: str,
            test_labels_path: str
    ) -> None:
        self.training_images_path = trainging_images_path
        self.training_labels_path = training_labels_path
        self.test_images_path = test_images_path
        self.test_labels_path = test_labels_path

    def read_dataset(
            self,
            images_path: str,
            labels_path: str
    ) -> tuple:
        labels = []
        with open(labels_path, "rb") as file:
            magic, size = struct.unpack(">II", file.read(8))
            if magic != 2049:
                raise ValueError("Invalid magic number in labels file")
            labels = array("B", file.read())
        
        with open(images_path, "rb") as file:
            magic, size, rows, cols = struct.unpack(">IIII", file.read(16))
            if magic != 2051:
                raise ValueError("Invalid magic number in images file")
            image_data = array("B", file.read())

        images = []
        for i in range(size):
            img = np.array(image_data[i * rows * cols:(i + 1) * rows * cols])
            img = img.reshape(rows, cols)
            images.append(img)
        
        return images, labels

    def load_dataset(self) -> tuple:
        self.training_images, self.training_labels = self.read_dataset(self.training_images_path, self.training_labels_path)
        self.test_images, self.test_labels = self.read_dataset(self.test_images_path, self.test_labels_path)
        return (self.training_images, self.training_labels), (self.test_images, self.test_labels)