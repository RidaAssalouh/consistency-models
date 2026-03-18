"""Simple MNIST loader for bootstrapping experiments."""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from ..config import ConsistencyConfig


def build_mnist_dataloader(config: ConsistencyConfig) -> DataLoader:
    """Return a DataLoader over MNIST train split."""

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )
    dataset = datasets.MNIST(
        root="data",
        train=True,
        download=True,
        transform=transform,
    )
    loader = DataLoader(
        dataset,
        batch_size=config.data.batch_size,
        shuffle=True,
        num_workers=config.data.num_workers,
        drop_last=True,
    )
    return loader
