from tensorize import Model, Convolution2D, RectifiedLinearUnit, MaxPooling2D
from tensorize import Flatten, FullyConnected, CategoricalPredictionOutput
from tensorize import Dropout
from tensorize import CategoricalCrossEntropy, CategoricalAccuracy
from tensorize import GradientDescentOptimizer

from tensorize import BatchImageInput
from tensorize import ZeroPadding2D


class VGG16(Model):

    def build_inference_graph(self, input_dim, output_dim):
        BatchImageInput(input_dim)

        ZeroPadding2D()
        Convolution2D(kernel=[3, 3], filters=64)
        RectifiedLinearUnit()

        ZeroPadding2D()
        Convolution2D(kernel=[3, 3], filters=64)
        RectifiedLinearUnit()

        MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        ZeroPadding2D()
        Convolution2D(kernel=[3, 3], filters=128)
        RectifiedLinearUnit()

        ZeroPadding2D()
        Convolution2D(kernel=[3, 3], filters=128)
        RectifiedLinearUnit()

        MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        ZeroPadding2D()
        Convolution2D(kernel=[3, 3], filters=256)
        RectifiedLinearUnit()

        ZeroPadding2D()
        Convolution2D(kernel=[3, 3], filters=256)
        RectifiedLinearUnit()

        ZeroPadding2D()
        Convolution2D(kernel=[3, 3], filters=256)
        RectifiedLinearUnit()

        MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        ZeroPadding2D()
        Convolution2D(kernel=[3, 3], filters=512)
        RectifiedLinearUnit()

        ZeroPadding2D()
        Convolution2D(kernel=[3, 3], filters=512)
        RectifiedLinearUnit()

        ZeroPadding2D()
        Convolution2D(kernel=[3, 3], filters=512)
        RectifiedLinearUnit()

        MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        ZeroPadding2D()
        Convolution2D(kernel=[3, 3], filters=512)
        RectifiedLinearUnit()

        ZeroPadding2D()
        Convolution2D(kernel=[3, 3], filters=512)
        RectifiedLinearUnit()

        ZeroPadding2D()
        Convolution2D(kernel=[3, 3], filters=512)
        RectifiedLinearUnit()

        MaxPooling2D(pool_size=(2, 2), strides=(2, 2))

        Flatten()

        FullyConnected(hidden_dim=4096)
        RectifiedLinearUnit()
        Dropout(1.0)

        FullyConnected(hidden_dim=4096)
        RectifiedLinearUnit()
        Dropout(1.0)

        CategoricalPredictionOutput(output_dim)

    def build_train_graph(self, output_dim):
        CategoricalCrossEntropy(output_dim)
        CategoricalAccuracy()

        GradientDescentOptimizer()
