from tqdm import trange
import pyro
from pyro.optim import Adam
from pyro.infer import SVI, TraceMeanField_ELBO

def vcl(model, guide, metagenerator, lr=0.001, betas=(0.9, 0.99), epochs=100):
    """
    Trains the given BNN on the data batches from the given metagenerator using SVI
    and yields each epoch's average loss.
    """
    pyro.clear_param_store()
    optimizer = Adam({"lr": lr, "betas": betas})
    svi = SVI(model, guide, optimizer, loss=TraceMeanField_ELBO())
    for _ in trange(epochs):
        avg_loss = 0
        num_examples = 0
        for batch_x, batch_y in metagenerator():
            avg_loss += svi.step(batch_x, batch_y)
            num_examples += batch_x.shape[0]
        avg_loss /= num_examples
        yield avg_loss
