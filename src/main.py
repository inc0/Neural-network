from network import NeuralNet
from learning import PATTERNS
from tests import TESTS
import math


def main():
    nn = NeuralNet(64,20,2)
    nn.train(PATTERNS, iterations=1000)
    print nn.show(TESTS)


if __name__ == '__main__':
    main()
