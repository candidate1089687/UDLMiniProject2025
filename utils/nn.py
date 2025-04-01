from torch import nn

class NN(nn.Module):
    """
    A class used in the MLE initialization step of VCL for permuted MNIST.
    Adapted from Practical 3: https://colab.research.google.com/drive/1VXJgDP8wGm_ixsD8PYTI6A0kNj-fOFmN
    """
    def __init__(self, n_inputs: int, n_outputs: int):
        super(NN, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(n_inputs, 100),
            nn.ReLU(),
            nn.Linear(100, 100),
            nn.ReLU(),
            nn.Linear(100, n_outputs),
            nn.Softmax(dim=1)
        )
    def forward(self, x):
        return self.layers(x)
