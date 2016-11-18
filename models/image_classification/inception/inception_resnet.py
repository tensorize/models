from tensorize import *

class InceptionResnetV1(Model):

    def inference(self, inputs, output):
        stem(inputs, outputs)

        for x in xrange(4):
            inceptionA()

        reductionA()

        for x in xrange(7):
            inceptionB()

        reductionB()

        for x in xrange(3):
            inceptionC()

        AveragePooling()
        Dropout(0.8)
        CategoricalPredictionOutput(output)

    def train(self, outputs):
        CategoricalCrossEntropy()
        CategoricalAccuracy(outputs)
        GradientDescentOptimizer()


class InceptionResnetV2(Model):

    def inference(self, inputs, output):
        stem(inputs, outputs)

        for x in xrange(4):
            inceptionA()

        reductionA()

        for x in xrange(7):
            inceptionB()

        reductionB()

        for x in xrange(3):
            inceptionC()

        AveragePooling()
        Dropout(0.8)
        CategoricalPredictionOutput(output)

    def train(self, outputs):
        CategoricalCrossEntropy()
        CategoricalAccuracy(outputs)
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
            AveragePooling()
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
            AveragePooling()
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
            AveragePooling()
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
            MaxPooling2D([3, 3])

        with parallel:
            Convolution3x3(n)

        with parallel:
            Convolution1x1(filters=k)
            Convolution3x3(filters=l)
            Convolution3x3(filters=m)

        FilterConcat()

def reduceB():
    with ParallelBlock() as parallel:

        with parallel:
            MaxPooling2D([3, 3], stride=2)

        with parallel:
            Convolution1x1(192)
            Convolution3x3(192)

        with parallel:
            Convolution1x1(filters=256)
            Convolution2D([1, 7], filters=256)
            Convolution2D([7, 1], filters=320)
            Convolution3x3(filters=320, stride=2)

        FilterConcat()


def inceptionResnetA():
    RectifiedLinearUnit()

    with ParallelBlock() as parallel:

        with parallel:
            with ParallelBlock() as parallel_inner:
                with parallel_inner:
                    Convolution1x1(32)

                with parallel_inner:
                    Convolution1x1(32)
                    Convolution3x3(32)

                with parallel_inner:
                    Convolution1x1(32)
                    Convolution3x3(32)
                    Convolution3x3(32)

            Convolution1x1(filters=256)

    Sum()


def inceptionResnetB():
    RectifiedLinearUnit()

    with ParallelBlock() as parallel:

        with parallel:
            with ParallelBlock() as parallel_inner:

                with parallel_inner:
                    Convolution1x1(128)

                with parallel_inner:
                    Convolution1x1(128)
                    Convolution2D([1, 7], filters=128)
                    Convolution2D([7, 1], filters=128)

            Convolution1x1(filters=896)
    Sum()

