from tensorize import Model, Convolution2D, RectifiedLinearUnit, MaxPooling2D
from tensorize import Flatten, FullyConnected, CategoricalPredictionOutput
from tensorize import Dropout
from tensorize import Tanh
from tensorize import CategoricalCrossEntropy, CategoricalAccuracy
from tensorize import GradientDescentOptimizer, AdamOptimizer

from tensorize import BatchImageInput


class Lenet(Model):

    def build_inference_graph(self, input_dim, output_dim):
        BatchImageInput(input_dim)
        Convolution2D(kernel=[5, 5], filters=20)
        Tanh()
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        Convolution2D(kernel=[5, 5], filters=50)
        Tanh()
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        Flatten()
        FullyConnected(hidden_dim=500)
        Tanh()

        CategoricalPredictionOutput(output_dim)

    def build_train_graph(self, output_dim):
        CategoricalCrossEntropy(output_dim)
        CategoricalAccuracy()

        GradientDescentOptimizer()

class LenetOptimized(Model):

    def build_inference_graph(self, input_dim, output_dim):
        BatchImageInput(input_dim)
        Convolution2D(kernel=[5, 5], filters=20)
        RectifiedLinearUnit()
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        Convolution2D(kernel=[5, 5], filters=50)
        RectifiedLinearUnit()
        MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        Flatten()
        FullyConnected(hidden_dim=512)
        RectifiedLinearUnit()

        FullyConnected(hidden_dim=512)
        RectifiedLinearUnit()

        Dropout(0.8)

        CategoricalPredictionOutput(output_dim)

    def build_train_graph(self, output_dim):
        CategoricalCrossEntropy(output_dim)
        CategoricalAccuracy()

        AdamOptimizer()
