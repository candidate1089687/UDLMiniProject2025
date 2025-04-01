import torch
from torch.nn.functional import one_hot
from torchvision import datasets, transforms

class PermutedMNIST():
    """
    Produce permuted MNIST data and labels.
    """

    # Load and process the data and labels exactly once up front
    _transform = transforms.Compose([transforms.ToTensor()])
    _train_dataset = datasets.MNIST('./data', train=True, download=True, transform=_transform)
    _train_data = _train_dataset.data.reshape(60000, 784).to(dtype=torch.float32)
    _train_labels = _train_dataset.targets
    _one_hot_train_labels = one_hot(_train_dataset.targets).to(dtype=torch.float32)
    _test_dataset = datasets.MNIST('./data', train=False, download=True, transform=_transform)
    _test_data = _test_dataset.data.reshape(10000, 784).to(dtype=torch.float32)
    _test_labels = _test_dataset.targets
    _one_hot_test_labels = one_hot(_test_dataset.targets).to(dtype=torch.float32)

    @classmethod
    def to(cls, device):
        """
        Copy data and labels to the given device.
        """
        cls._train_data = cls._train_data.to(device)
        cls._train_labels = cls._train_labels.to(device)
        cls._one_hot_train_labels = cls._one_hot_train_labels.to(device)
        cls._test_data = cls._test_data.to(device)
        cls._test_labels = cls._test_labels.to(device)
        cls._one_hot_test_labels = cls._one_hot_test_labels.to(device)

    @classmethod
    def metagenerator(cls, perm, batch_size, one_hot=False):
        """
        Return a function which, when called, yields permuted MNIST training data and labels
        in random batches of the given size. Each call creates new random batches.
        """
        num_batches = 60000 // batch_size
        remainder = 60000 % batch_size
        if one_hot:
            labels = cls._one_hot_train_labels
        else:
            labels = cls._train_labels
        def generator():
            for indices in torch.randperm(60000)[remainder:].reshape(num_batches, batch_size):
                yield cls._train_data[:,perm][indices], labels[indices]
        return generator
    
    @classmethod
    def test_data(cls, perm):
        """
        Return permuted MNIST test data.
        """
        return cls._test_data[:,perm]
    
    @classmethod
    def test_labels(cls):
        """
        Return MNIST test labels.
        """
        return cls._test_labels
    
    @classmethod
    def one_hot_test_labels(cls):
        """
        Return MNIST test labels.
        """
        return cls._one_hot_test_labels
