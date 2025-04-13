import pii
import wandb
import torch
from torcheval.metrics.functional import mean_squared_error
from pyro.infer import Predictive
from pyro.infer.autoguide import AutoNormal
from pyro.infer.autoguide.initialization import init_to_mean
from utils.minnie import PermutedMNIST
from utils.bunny import GaussianBNN
from utils.vcl import vcl

def acc(scores, labels):
    return torch.sum(torch.eq(scores, labels))/len(scores)

init_loc_fn, init_scale, lr, epochs = init_to_mean, 0.001, 0.0001, 100

# Start a new wandb run to track this script
run = wandb.init(
    entity=pii.ENTITY,
    project="Candidate 1089687's UDL Mini Project 2025",
    config={
        "name": "Coreset-Free VCL Extension 11 April",
        "init_loc_fn": "init_to_mean",
        "init_scale": init_scale,
        "learning_rate": lr,
        "epochs": epochs
    },
)

# Find device
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

# Send the training data to the right device
PermutedMNIST.to(device)

# Construct one model and one guide for all tasks
bnn = GaussianBNN(device)
bnn.set_prior_from_state_dict("mle_extension.pt")
guide = AutoNormal(bnn, init_loc_fn=init_loc_fn, init_scale=init_scale)

# Train on four tasks
perms = torch.load("perms.pt", weights_only=True)
for i in range(4):
    # Train
    gen = PermutedMNIST.metagenerator(perms[i], 256, one_hot=True)
    # Send losses to W&B
    for avg_loss in vcl(bnn, guide, gen, lr=lr, epochs=epochs):
        run.log({"avg_loss": avg_loss})
    # Send accuracy to W&B
    predictive = Predictive(bnn, guide=guide, num_samples=300, return_sites=("obs",))
    data = torch.cat(tuple(PermutedMNIST.test_data(perms[j]) for j in range(i+1)))
    preds = torch.mode(predictive(data)["obs"], dim=0).values
    y = torch.cat(tuple(PermutedMNIST.one_hot_test_labels() for j in range(i+1)))
    run.log({"Mean Squared Error": mean_squared_error(preds, y)})
    # Posterior becomes next prior
    bnn.set_prior_from_param_store()
run.finish()
