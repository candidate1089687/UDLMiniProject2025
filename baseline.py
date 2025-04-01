from itertools import chain
import wandb
import torch
from torch.nn import KLDivLoss
import pii
from utils.minnie import PermutedMNIST
from utils.nn import NN
from utils.mle import mle

def acc(preds, labels):
    return torch.sum(torch.eq(torch.argmax(preds, dim=1), labels), dim=0) / len(preds)

def helper(metagen, coreset):
    def f():
        return chain(metagen(), coreset)
    return f

# Start a new wandb run to track this script
run = wandb.init(
    entity=pii.ENTITY,
    project="Candidate 1089687's UDL Mini Project 2025",
    config={
        "name": "Baseline Method (Coreset Only)",
        "learning_rate": 0.0001,
        "epochs": 100,
    },
)

# Find device
if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
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
    for avg_loss in mle(model, metagen_with_coreset, KLDivLoss(reduction="batchmean"), lr=0.0001, epochs=100):
        run.log({"avg_loss": avg_loss})
    # Save weights from first task for reuse by VCL later
    if i == 0:
        torch.save(model.state_dict(), "mle.pt")
    # Log average accuracy
    data = torch.cat(tuple(PermutedMNIST.test_data(perms[j]) for j in range(i+1)))
    labels = torch.cat(tuple(PermutedMNIST.test_labels() for j in range(i+1)))
    preds = model(data)
    run.log({"acc": acc(preds, labels)})
    # Add a random batch of the current dataset to the coreset
    coreset.append(next(metagen()))
run.finish()
