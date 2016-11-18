from deepwater import *
import deepwater

class InceptionV4(Model):

    def build_inference_graph(self, inputs, outputs):
        stem(inputs, outputs)

        for x in xrange(4):
            inceptionA()
        # from Table.1 of the 1602.07261.pdf
        reduceA(192, 224, 256, 384)

        for x in xrange(7):
            inceptionB()

        reduceB()

        for x in xrange(3):
            inceptionC()

        AveragePooling2x2()
        Dropout(0.8)
        CategoricalPredictionOutput(outputs)

    def build_train_graph(self, outputs):
        CategoricalCrossEntropy(outputs)
        CategoricalAccuracy()
        GradientDescentOptimizer()


def stem(inputs, outputs):

    BatchImageInput(inputs)
    Convolution3x3(filters=32)
    Convolution3x3(filters=32)
    Convolution3x3(filters=64)

    with ParallelBlock() as parallel:
        with parallel:
            MaxPooling2D()
        with parallel:
            Convolution3x3(filters=64)

    print("layer:", deepwater._default_model._layers)

    FilterConcat()

    with ParallelBlock() as parallel:
        with parallel:
            Convolution1x1(filters=64)
            Convolution3x3(filters=96)

        with parallel:
            Convolution1x1(filters=64)
            Convolution2D([7, 1], filters=64)
            Convolution2D([1, 7], filters=64)
            Convolution3x3(filters=96)
        FilterConcat()

    with ParallelBlock() as block:
        with block:
            MaxPooling2D()
        with block:
            Convolution3x3(filters=64)
        FilterConcat()

def inceptionA():
    with ParallelBlock() as parallel:
        with parallel:
            AveragePooling2x2()
            Convolution1x1(filters=96)

        with parallel:
            Convolution1x1(filters=96)

        with parallel:
            Convolution1x1(filters=64)
            Convolution3x3(filters=96)

        with parallel:
            Convolution1x1(filters=64)
            Convolution3x3(filters=96)
            Convolution3x3(filters=96)

        FilterConcat()

def inceptionB():

    with ParallelBlock() as parallel:
        with parallel:
            AveragePooling2x2()
            Convolution1x1(filters=128)

        with parallel:
            Convolution1x1(filters=384)

        with parallel:
            Convolution1x1(filters=192)
            Convolution2D([1, 7], filters=224)
            Convolution2D([1, 7], filters=256)

        with parallel:
            Convolution1x1(filters=192)
            Convolution2D([1, 7], filters=192)
            Convolution2D([7, 1], filters=224)
            Convolution2D([1, 7], filters=224)
            Convolution2D([7, 1], filters=256)

        FilterConcat()

def inceptionC():
    with ParallelBlock() as parallel:
        with parallel:
            AveragePooling2x2()
            Convolution1x1(filters=256)

        with parallel:
            Convolution1x1(filters=256)

        with parallel:
            Convolution1x1(filters=384)

            with ParallelBlock() as parallel_inner:
                with parallel_inner:
                    Convolution2D([1, 3], filters=256)

                with parallel_inner:
                    Convolution2D([3, 1], filters=256)

        with parallel:

            Convolution1x1(filters=384)
            Convolution2D([1, 3], filters=384)
            Convolution2D([3, 1], filters=512)

        FilterConcat()

def reduceA(n, l, k, m):
    with ParallelBlock() as parallel:

        with parallel:
            MaxPooling2D([1, 3, 3, 1])

        with parallel:
            Convolution3x3(filters=n)

        with parallel:
            Convolution1x1(filters=k)
            Convolution3x3(filters=l)
            Convolution3x3(filters=m)

        FilterConcat()

def reduceB():
    with ParallelBlock() as parallel:

        with parallel:
            MaxPooling2D([1, 3, 3, 1], strides=[1, 2, 2, 1])

        with parallel:
            Convolution1x1(filters=192)
            Convolution3x3(filters=192)

        with parallel:
            Convolution1x1(filters=256)
            Convolution2D([1, 7], filters=256)
            Convolution2D([7, 1], filters=320)
            Convolution3x3(filters=320, stride=2)

        FilterConcat()


