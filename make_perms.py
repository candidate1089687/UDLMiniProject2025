import torch

perms = torch.stack(
    (
        torch.randperm(784),
        torch.randperm(784),
        torch.randperm(784),
        torch.randperm(784)
    )
)
torch.save(perms, "perms.pt")
