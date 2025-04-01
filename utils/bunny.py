from torch.nn.functional import relu, softmax, softplus
import torch
import pyro
from pyro.distributions import Normal, Categorical

class Base():
    """
    A base class from which two types of BNN will subclass.
    """

    def __init__(self, device):
        self.device = device
        self.locs = {}
        self.scales = {}
    
    def set_prior_from_state_dict(self, path):
        """
        Set the prior to match the state dict saved at the given path,
        with all self.scales set to 0.001.
        """
        sdict = torch.load(path, weights_only=False)
        self.locs["W1"] = sdict["layers.0.weight"].transpose(0, 1).to(self.device)
        self.locs["b1"] = sdict["layers.0.bias"].unsqueeze(0).to(self.device)
        self.locs["W2"] = sdict["layers.2.weight"].transpose(0, 1).to(self.device)
        self.locs["b2"] = sdict["layers.2.bias"].unsqueeze(0).to(self.device)
        self.locs["W3"] = sdict["layers.4.weight"].transpose(0, 1).to(self.device)
        self.locs["b3"] = sdict["layers.4.bias"].unsqueeze(0).to(self.device)
        self.scales["W1"] = 1e-3*torch.ones(784, 100).to(self.device)
        self.scales["b1"] = 1e-3*torch.ones(1, 100).to(self.device)
        self.scales["W2"] = 1e-3*torch.ones(100, 100).to(self.device)
        self.scales["b2"] = 1e-3*torch.ones(1, 100).to(self.device)
        self.scales["W3"] = 1e-3*torch.ones(100, 10).to(self.device)
        self.scales["b3"] = 1e-3*torch.ones(1, 10).to(self.device)
    
    def set_prior_from_param_store(self):
        """
        Set the prior to match the current param store.
        """
        params = pyro.get_param_store()
        self.locs["W1"] = params["AutoNormal.locs.W1"].detach().to(self.device)
        self.locs["b1"] = params["AutoNormal.locs.b1"].detach().to(self.device)
        self.locs["W2"] = params["AutoNormal.locs.W2"].detach().to(self.device)
        self.locs["b2"] = params["AutoNormal.locs.b2"].detach().to(self.device)
        self.locs["W3"] = params["AutoNormal.locs.W3"].detach().to(self.device)
        self.locs["b3"] = params["AutoNormal.locs.b3"].detach().to(self.device)
        self.scales["W1"] = softplus(params["AutoNormal.scales.W1"]).detach().to(self.device)  # Positive params are stored as their preimage before softplus
        self.scales["b1"] = softplus(params["AutoNormal.scales.b1"]).detach().to(self.device)
        self.scales["W2"] = softplus(params["AutoNormal.scales.W2"]).detach().to(self.device)
        self.scales["b2"] = softplus(params["AutoNormal.scales.b2"]).detach().to(self.device)
        self.scales["W3"] = softplus(params["AutoNormal.scales.W3"]).detach().to(self.device)
        self.scales["b3"] = softplus(params["AutoNormal.scales.b3"]).detach().to(self.device)
    
    def _run(self, x):
        """
        Run the BNN forward and return the output of dimension 10.
        """
        # First layer: 28*28 -> 100 with ReLU
        W1 = pyro.sample("W1", Normal(
            loc=self.locs["W1"],
            scale=self.scales["W1"]
        ).to_event(2))  # we move the two rightmost dims to the event shape (in this case its all dims)
        b1 = pyro.sample("b1", Normal(
            loc=self.locs["b1"],
            scale=self.scales["b1"]
        ).to_event(2))
        output1 = relu(x @ W1 + b1)  # W1 and b1 are broadcasted
        # Second layer: 100 -> 100 with ReLU
        W2 = pyro.sample("W2", Normal(
            loc=self.locs["W2"],
            scale=self.scales["W2"]
        ).to_event(2))
        b2 = pyro.sample("b2", Normal(
            loc=self.locs["b2"],
            scale=self.scales["b2"]
        ).to_event(2))
        output2 = relu(output1 @ W2 + b2)  # W2 and b2 are broadcasted
        # Third layer: 100 -> 10 with softmax
        W3 = pyro.sample("W3", Normal(
            loc=self.locs["W3"],
            scale=self.scales["W3"]
        ).to_event(2))
        b3 = pyro.sample("b3", Normal(
            loc=self.locs["b3"],
            scale=self.scales["b3"]
        ).to_event(2))
        return softmax(output2 @ W3 + b3, dim=1)  # W3 and b3 are broadcasted

class CategoricalBNN(Base):
    """
    A class which implements a BNN for permuted MNIST that samples from a latent
    categorical distribution.
    """

    def __call__(self, x, y=None):
        probs = self._run(x)
        # Make observations with noise
        with pyro.plate("data", x.shape[0]):
            return pyro.sample("obs", Categorical(
                probs=probs
            ), obs=y)

class GaussianBNN(Base):
    """
    A class which implements a BNN for permuted MNIST that samples from a latent
    Gaussian distribution with fixed standard deviation 0.001.
    """

    def __call__(self, x, y=None):
        means = self._run(x)
        with pyro.plate("data", x.shape[0]):
            return pyro.sample("obs", Normal(
                loc=means,
                scale=1e-3*torch.ones(1, 10).to(self.device)  # broadcasted
            ).to_event(1), obs=y)
