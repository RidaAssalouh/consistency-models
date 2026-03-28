import os
import math
import random
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from typing import Optional


@dataclass
class Config:
    data_root: str = "consistency_models/data"
    ckpt_dir: str = "consistency_models/consistency_models_ckpts"
    ckpt_name: str = "fashionmnist_classifier_best.pt"

    seed: int = 42
    image_size: int = 28
    num_classes: int = 10

    batch_size: int = 256
    num_workers: int = 4
    epochs: int = 30

    lr: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05

    train_val_split: float = 0.9
    embedding_dim: int = 128

    use_amp: bool = True


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


class ConvBNAct(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel_size: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = ConvBNAct(channels, channels, 3, 1, 1)
        self.conv2 = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(channels),
        )
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv1(x)
        out = self.conv2(out)
        out = out + x
        return self.act(out)


class DownBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvBNAct(in_ch, out_ch, 3, 1, 1),
            ResidualBlock(out_ch),
            nn.MaxPool2d(2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class FashionClassifier(nn.Module):
    def __init__(self, num_classes: int = 10, embedding_dim: int = 128):
        super().__init__()

        self.stem = nn.Sequential(
            ConvBNAct(1, 64, 3, 1, 1),
            ResidualBlock(64),
        )

        self.stage1 = DownBlock(64, 128)   # 28 -> 14
        self.stage2 = DownBlock(128, 256)  # 14 -> 7

        self.stage3 = nn.Sequential(
            ConvBNAct(256, 256, 3, 1, 1),
            ResidualBlock(256),
            ConvBNAct(256, 256, 3, 1, 1),
        )

        self.pool = nn.AdaptiveAvgPool2d((1, 1))

        self.embedding = nn.Sequential(
            nn.Linear(256, 256),
            nn.BatchNorm1d(256),
            nn.SiLU(inplace=True),
            nn.Dropout(p=0.15),
            nn.Linear(256, embedding_dim),
        )

        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x).flatten(1)
        z = self.embedding(x)
        return z

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.forward_features(x)
        logits = self.classifier(z)
        return logits



def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler,
    criterion,
    device: torch.device,
    scaler: Optional[torch.cuda.amp.GradScaler],
    use_amp: bool,
) -> tuple[float, float]:
    model.train()

    running_loss = 0.0
    running_acc = 0.0
    count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()

        scheduler.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy_from_logits(logits.detach(), targets) * batch_size
        count += batch_size

    return running_loss / count, running_acc / count


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion,
    device: torch.device,
    use_amp: bool,
) -> tuple[float, float]:
    model.eval()

    running_loss = 0.0
    running_acc = 0.0
    count = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        if use_amp and device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, targets)
        else:
            logits = model(images)
            loss = criterion(logits, targets)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        running_acc += accuracy_from_logits(logits, targets) * batch_size
        count += batch_size

    return running_loss / count, running_acc / count


def main() -> None:
    cfg = Config()
    set_seed(cfg.seed)

    os.makedirs(cfg.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(cfg.ckpt_dir, cfg.ckpt_name)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    mean, std = 0.2860, 0.3530

    train_transform = transforms.Compose([
        transforms.RandomAffine(
            degrees=10,
            translate=(0.08, 0.08),
            scale=(0.95, 1.05),
        ),
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((mean,), (std,)),
    ])

    full_train_dataset = datasets.FashionMNIST(
        root=cfg.data_root,
        train=True,
        transform=train_transform,
        download=False,
    )

    full_train_dataset_eval = datasets.FashionMNIST(
        root=cfg.data_root,
        train=True,
        transform=eval_transform,
        download=False,
    )

    train_size = int(cfg.train_val_split * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size

    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset_aug = random_split(full_train_dataset, [train_size, val_size], generator=generator)

    val_indices = val_dataset_aug.indices
    train_indices = train_dataset.indices

    train_dataset = torch.utils.data.Subset(full_train_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_train_dataset_eval, val_indices)

    test_dataset = datasets.FashionMNIST(
        root=cfg.data_root,
        train=False,
        transform=eval_transform,
        download=False,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        persistent_workers=cfg.num_workers > 0,
    )

    model = FashionClassifier(
        num_classes=cfg.num_classes,
        embedding_dim=cfg.embedding_dim,
    ).to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )

    total_steps = cfg.epochs * len(train_loader)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_steps,
        eta_min=1e-6,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    best_val_acc = -1.0
    best_epoch = -1

    for epoch in range(1, cfg.epochs + 1):
        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            scheduler=scheduler,
            criterion=criterion,
            device=device,
            scaler=scaler,
            use_amp=cfg.use_amp,
        )

        val_loss, val_acc = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            use_amp=cfg.use_amp,
        )

        print(
            f"Epoch {epoch:03d}/{cfg.epochs:03d} | "
            f"train_loss={train_loss:.4f} | train_acc={100*train_acc:.2f}% | "
            f"val_loss={val_loss:.4f} | val_acc={100*val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch

            checkpoint = {
                "config": asdict(cfg),
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "best_val_acc": best_val_acc,
                "num_classes": cfg.num_classes,
                "embedding_dim": cfg.embedding_dim,
                "class_names": [
                    "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
                    "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"
                ],
                "normalization": {
                    "mean": mean,
                    "std": std,
                },
            }
            torch.save(checkpoint, ckpt_path)
            print(f"Saved best checkpoint to: {ckpt_path}")

    print(f"\nBest validation accuracy: {100*best_val_acc:.2f}% at epoch {best_epoch}")

    best_ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(best_ckpt["model_state_dict"])

    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        use_amp=cfg.use_amp,
    )

    print(f"Test loss: {test_loss:.4f}")
    print(f"Test accuracy: {100*test_acc:.2f}%")
    print(f"Final checkpoint saved at: {ckpt_path}")


if __name__ == "__main__":
    main()