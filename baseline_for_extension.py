from itertools import chain
import wandb
import torch
from torch.nn import MSELoss
from torcheval.metrics.functional import mean_squared_error
import pii
from utils.minnie import PermutedMNIST
from utils.nn import NN
from utils.mle import mle

def helper(metagen, coreset):
    def f():
        return chain(metagen(), coreset)
    return f

# Start a new wandb run to track this script
run = wandb.init(
    entity=pii.ENTITY,
    project="Candidate 1089687's UDL Mini Project 2025",
    config={
        "name": "Baseline Method (Coreset Only, Extension)",
        "learning_rate": 0.0001,
        "epochs": 100,
    },
)

# Find device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
# Initialize things
model = NN(784, 10)
model.to(device)
PermutedMNIST.to(device)
perms = torch.load("perms.pt", weights_only=True)
coreset = []
# Training
for i in range(4):
    # Train on current dataset plus coreset
    metagen = PermutedMNIST.metagenerator(perms[i], 256, one_hot=True)
    metagen_with_coreset = helper(metagen, coreset)
    for avg_loss in mle(model, metagen_with_coreset, MSELoss(), lr=0.0001, epochs=100):
        run.log({"avg_loss": avg_loss})
    # Save weights from first task for reuse by VCL later
    if i == 0:
        torch.save(model.state_dict(), "mle_extension.pt")
    # Log average accuracy
    data = torch.cat(tuple(PermutedMNIST.test_data(perms[j]) for j in range(i+1)))
    y = torch.cat(tuple(PermutedMNIST.one_hot_test_labels() for j in range(i+1)))
    preds = model(data)
    run.log({"mse": mean_squared_error(preds, y)})
    # Add a random batch of the current dataset to the coreset
    coreset.append(next(metagen()))
run.finish()
