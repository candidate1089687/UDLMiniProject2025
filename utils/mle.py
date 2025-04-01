from tqdm import trange
import torch

def mle(model, metagenerator, loss_fn, lr=0.001, epochs=100):
    """
    Trains the given NN on the data batches from the given metagenerator to obtain the MLE
    with respect to the given loss function.
    Adapted from Practical 3: https://colab.research.google.com/drive/1VXJgDP8wGm_ixsD8PYTI6A0kNj-fOFmN
    """
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    for epoch in trange(epochs):
        train_loss = 0
        for batch_x, batch_y in metagenerator():
            optimiser.zero_grad()
            output = model(batch_x)
            loss = loss_fn(output, batch_y)
            loss.backward()
            optimiser.step()
            train_loss += loss.item() / len(batch_x)
        yield train_loss
