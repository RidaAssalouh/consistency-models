import math
import os
import random
import copy
from dataclasses import dataclass, asdict
from typing import Tuple, Optional, List
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, utils

import matplotlib.pyplot as plt

from models.cd_model_utils import UNet as UNet_CD
from models.ct_model_utils import UNet as UNet_CT
from models.teacher_utils import UNet as UNet_Teacher

@dataclass
class InpaintConfig:
    data_root: str = "./data"
    image_size: int = 32
    in_channels: int = 1

    batch_size: int = 16
    num_workers: int = 2

    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    time_emb_dim: int = 256
    dropout: float = 0.0

    sigma_data: float = 0.5
    eps: float = 1e-3
    sigma_max: float = 50.0
    rho: float = 7.0
    num_steps: int = 40

    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42

    num_examples: int = 10
    class_filter: Optional[int] = None 
    save_dir: str = "./inverse_problems_results/inpainting_results"

    mask_type: str = "random_pixels" # options: "center", "random_square", "half", "random_pixels"
    random_pixel_prob: float = 0.7
    square_size: int = 20



def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denorm_to_01(x: Tensor) -> Tensor:
    return ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)


class ConsistencyModel_CD(nn.Module):
    def __init__(self, backbone: nn.Module, sigma_data: float = 0.5, eps: float = 1e-3):
        super().__init__()
        self.backbone = backbone
        self.sigma_data = sigma_data
        self.eps = eps

    def c_skip(self, sigma: Tensor) -> Tensor:
        return self.sigma_data ** 2 / (sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: Tensor) -> Tensor:
        return sigma * self.sigma_data / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        sigma  = sigma.float().clamp(min=self.eps)
        c_skip = self.c_skip(sigma).view(-1, 1, 1, 1)
        c_out  = self.c_out(sigma).view(-1, 1, 1, 1)
        F_out  = self.backbone(x, sigma)
        return c_skip * x + c_out * F_out




class ModelEMA_CD:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        msd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            src = msd[k]
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src.detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(src)

    @torch.no_grad()
    def __call__(self, x: Tensor, sigma: Tensor) -> Tensor:
        return self.ema_model(x, sigma)
    



class ConsistencyModel_CT(nn.Module):
    def __init__(self, backbone: nn.Module, sigma_data: float = 0.5, eps: float = 1e-3):
        super().__init__()
        self.backbone = backbone
        self.sigma_data = sigma_data
        self.eps = eps

    def c_skip(self, sigma: Tensor) -> Tensor:
        return (self.sigma_data ** 2) / (((sigma - self.eps) ** 2) + self.sigma_data ** 2)

    def c_out(self, sigma: Tensor) -> Tensor:
        return self.sigma_data * (sigma - self.eps) / torch.sqrt(self.sigma_data ** 2 + sigma ** 2)

    def c_in(self, sigma: Tensor) -> Tensor:
        return 1.0 / torch.sqrt(self.sigma_data ** 2 + sigma ** 2)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        sigma = sigma.float().clamp(min=self.eps)

        cskip = self.c_skip(sigma).view(-1, 1, 1, 1)
        cout = self.c_out(sigma).view(-1, 1, 1, 1)
        cin = self.c_in(sigma).view(-1, 1, 1, 1)

        raw = self.backbone(cin * x, sigma)
        return cskip * x + cout * raw



def build_karras_sigmas(
    sigma_min: float,
    sigma_max: float,
    num_scales: int,
    rho: float,
    device: torch.device,
) -> Tensor:
    ramp = torch.linspace(0, 1, num_scales, device=device)
    inv_rho_min = sigma_min ** (1.0 / rho)
    inv_rho_max = sigma_max ** (1.0 / rho)
    sigmas = (inv_rho_max + ramp * (inv_rho_min - inv_rho_max)) ** rho
    return sigmas


def build_dataset_loader(cfg: InpaintConfig) -> DataLoader:
    transform = transforms.Compose([
        transforms.Resize((cfg.image_size, cfg.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    ds = datasets.FashionMNIST(
        root=cfg.data_root,
        train=False,
        download=False,
        transform=transform,
    )

    if cfg.class_filter is None:
        subset = ds
    else:
        idxs = [i for i, (_, y) in enumerate(ds) if int(y) == int(cfg.class_filter)]
        subset = torch.utils.data.Subset(ds, idxs)

    return DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )


def make_inpainting_mask(
    batch: Tensor,
    mask_type: str = "center",
    square_size: int = 14,
    random_pixel_prob: float = 0.5,
) -> Tensor:
    """
    Returns Omega with shape (B,1,H,W), where:
      Omega = 1 on missing pixels,
      Omega = 0 on known pixels.
    """
    b, c, h, w = batch.shape
    device = batch.device
    omega = torch.zeros((b, c, h, w), device=device)

    if mask_type == "center":
        s = square_size
        top = (h - s) // 2
        left = (w - s) // 2
        omega[:, :, top:top+s, left:left+s] = 1.0

    elif mask_type == "random_square":
        s = square_size
        for i in range(b):
            top = random.randint(0, h - s)
            left = random.randint(0, w - s)
            omega[i, :, top:top+s, left:left+s] = 1.0

    elif mask_type == "half":
        omega[:, :, :, w // 2:] = 1.0

    elif mask_type == "random_pixels":
        omega = (torch.rand((b, c, h, w), device=device) < random_pixel_prob).float()

    else:
        raise ValueError(f"Unknown mask_type: {mask_type}")

    return omega



def build_ct_model_from_checkpoint(ckpt_path: str, device: torch.device) -> ConsistencyModel_CT:
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg_dict = ckpt.get("config", {})
    in_channels = cfg_dict.get("in_channels", 1)
    base_channels = cfg_dict.get("base_channels", 64)
    channel_mults = tuple(cfg_dict.get("channel_mults", [1, 2, 4]))
    num_res_blocks = cfg_dict.get("num_res_blocks", 2)
    time_emb_dim = cfg_dict.get("time_emb_dim", 256)
    dropout = cfg_dict.get("dropout", 0.0)
    sigma_data = cfg_dict.get("sigma_data", 0.5)
    eps = cfg_dict.get("eps", 1e-3)

    backbone = UNet_CT(
        in_channels=in_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        time_emb_dim=time_emb_dim,
        dropout=dropout,
    ).to(device)

    model = ConsistencyModel_CT(
        backbone=backbone,
        sigma_data=sigma_data,
        eps=eps,
    ).to(device)

    state_dict = ckpt.get("ema_model", None)
    if state_dict is None:
        state_dict = ckpt.get("model", None)
    if state_dict is None:
        raise KeyError(
            f"Could not find 'ema_model' or 'model' in checkpoint: {ckpt_path}"
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model

def build_cd_model_from_checkpoint(ckpt_path: str, device: torch.device) -> ConsistencyModel_CD:
    ckpt = torch.load(ckpt_path, map_location=device)

    cfg_dict = ckpt.get("config", {})
    in_channels = cfg_dict.get("in_channels", 1)
    base_channels = cfg_dict.get("base_channels", 64)
    channel_mults = tuple(cfg_dict.get("channel_mults", [1, 2, 4]))
    num_res_blocks = cfg_dict.get("num_res_blocks", 2)
    time_emb_dim = cfg_dict.get("time_emb_dim", 256)
    dropout = cfg_dict.get("dropout", 0.0)
    sigma_data = cfg_dict.get("sigma_data", 0.5)
    eps = cfg_dict.get("eps", 1e-3)

    backbone = UNet_CD(
        in_channels=in_channels,
        base_channels=base_channels,
        channel_mults=channel_mults,
        num_res_blocks=num_res_blocks,
        time_emb_dim=time_emb_dim,
        dropout=dropout,
    ).to(device)

    model = ConsistencyModel_CD(
        backbone=backbone,
        sigma_data=sigma_data,
        eps=eps,
    ).to(device)

    state_dict = ckpt.get("ema_model", None)
    if state_dict is None:
        state_dict = ckpt.get("model", None)
    if state_dict is None:
        raise KeyError(
            f"Could not find 'ema_model' or 'model' in checkpoint: {ckpt_path}"
        )

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# algorithm 4 for inpainting
@torch.no_grad()
def algorithm4_inpaint(
    model,
    y_masked: Tensor,
    omega: Tensor,
    sigmas: Tensor,
    eps: float,
) -> Tensor:

    device = y_masked.device
    b = y_masked.shape[0]
    y = y_masked * (1.0 - omega)
    t1 = sigmas[0]
    x = y + t1 * torch.randn_like(y)
    x = model(x, torch.full((b,), t1.item(), device=device))
    x = y * (1.0 - omega) + x * omega
    for tn in sigmas[1:]:
        noise_std = torch.sqrt(torch.clamp(tn**2 - eps**2, min=0.0))
        x = x + noise_std * torch.randn_like(x)
        x = model(x, torch.full((b,), tn.item(), device=device))
        x = y * (1.0 - omega) + x * omega

    return x.clamp(-1.0, 1.0)


def show_results(
    originals: Tensor,
    masked: Tensor,
    masks: Tensor,
    recon: Tensor,
    title: str,
    save_path: Optional[str] = None,
) -> None:
    originals = denorm_to_01(originals).cpu()
    masked = denorm_to_01(masked).cpu()
    recon = denorm_to_01(recon).cpu()
    masks = masks.cpu()

    b = originals.shape[0]
    fig, axes = plt.subplots(b, 4, figsize=(10, 2.5 * b))
    if b == 1:
        axes = axes[None, :]

    col_titles = ["Original", "Masked input y", "Mask Ω", "Inpainted"]
    for j in range(4):
        axes[0, j].set_title(col_titles[j], fontsize=11)

    for i in range(b):
        axes[i, 0].imshow(originals[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(masked[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(masks[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 3].imshow(recon[i, 0], cmap="gray", vmin=0, vmax=1)

        for j in range(4):
            axes[i, j].axis("off")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()

    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")

    plt.show()


@dataclass
class TeacherConfig:
    data_root: str = "./data"
    output_dir: str = "consistency_models_ckpts/teacher_epoch_060.pt"

    image_size: int = 32
    in_channels: int = 1
    batch_size: int = 128
    num_workers: int = 4

    base_channels: int = 64
    channel_mults: Tuple[int, ...] = (1, 2, 4)
    num_res_blocks: int = 2
    time_emb_dim: int = 256
    dropout: float = 0.0

    sigma_data: float = 0.5
    eps: float = 1e-3
    sigma_max: float = 50.0
    rho: float = 7.0

    epochs: int = 100
    lr: float = 2e-4
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0

    amp: bool = True
    ema_decay: float = 0.99
    seed: int = 42

    log_every: int = 100
    save_every_epochs: int = 10
    sample_every_epochs: int = 5
    num_sample_images: int = 16
    sample_steps: int = 32

    sigma_sample_mode: str = "karras"
    num_karras_scales: int = 40

    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@torch.no_grad()
def make_inpainting_observation(x0: Tensor, mask: Tensor) -> Tensor:

    return mask * x0
def build_karras_sigmas(
    sigma_min: float,
    sigma_max: float,
    num_scales: int,
    rho: float,
    device: torch.device,
) -> Tensor:
    ramp = torch.linspace(0, 1, num_scales, device=device)
    inv_rho_min = sigma_min ** (1.0 / rho)
    inv_rho_max = sigma_max ** (1.0 / rho)
    sigmas = (inv_rho_max + ramp * (inv_rho_min - inv_rho_max)) ** rho
    return sigmas



def dps_inpaint_sample(
    model,
    y: Tensor,
    mask: Tensor,
    sigma_max: float,
    sigma_min: float,
    steps: int,
    rho: float,
    device: torch.device,
    guidance_scale: float = 1e-3,
    num_guidance_steps: int = 1,
    hard_project_x0: bool = True,
):
    model.eval()
    B, C, H, W = y.shape

    sigmas = build_karras_sigmas(
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_scales=steps,
        rho=rho,
        device=device,
    )

    x = sigma_max * torch.randn((B, C, H, W), device=device)
    traj = []

    num_obs = mask.sum(dim=(1, 2, 3), keepdim=True).clamp_min(1.0)

    for i in range(len(sigmas) - 1):
        sigma_hi = sigmas[i]
        sigma_lo = sigmas[i + 1]
        sigma_batch_hi = torch.full((B,), sigma_hi.item(), device=device)

        for _ in range(num_guidance_steps):
            x = x.detach().requires_grad_(True)

            eps_hat = model(x, sigma_batch_hi)
            x0_hat = x - sigma_hi * eps_hat

            loss_per_sample = 0.5 * ((mask * (x0_hat - y)) ** 2).sum(dim=(1, 2, 3), keepdim=True) / num_obs
            loss = loss_per_sample.mean()

            grad = torch.autograd.grad(loss, x)[0]

            x = (x - guidance_scale * grad).detach()

        eps_hat = model(x, sigma_batch_hi)
        x0_hat = x - sigma_hi * eps_hat

        if hard_project_x0:
            x0_hat = mask * y + (1.0 - mask) * x0_hat

        traj.append(x0_hat.detach().cpu())

        x = x0_hat + sigma_lo * eps_hat

    sigma_batch_last = torch.full((B,), sigmas[-1].item(), device=device)
    x0_hat = model.predict_x0(x, sigma_batch_last)

    if hard_project_x0:
        x0_hat = mask * y + (1.0 - mask) * x0_hat

    x0_hat = x0_hat.clamp(-1.0, 1.0)
    return x0_hat, traj





class ModelEMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.ema_model = copy.deepcopy(model).eval()
        for p in self.ema_model.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        model_sd = model.state_dict()
        for k, v in self.ema_model.state_dict().items():
            src = model_sd[k]
            if v.dtype.is_floating_point:
                v.mul_(self.decay).add_(src.detach(), alpha=1.0 - self.decay)
            else:
                v.copy_(src)



class DiffusionTeacher(nn.Module):
    def __init__(self, backbone: nn.Module, sigma_data: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.sigma_data = sigma_data

    def c_in(self, sigma: Tensor) -> Tensor:
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: Tensor) -> Tensor:
        return sigma / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        sigma = sigma.float()
        cin = self.c_in(sigma).view(-1, 1, 1, 1)
        cout = self.c_out(sigma).view(-1, 1, 1, 1)
        raw = self.backbone(cin * x, sigma)
        return cout * raw

    @torch.no_grad()
    def predict_x0(self, x_sigma: Tensor, sigma: Tensor) -> Tensor:
        eps_hat = self(x_sigma, sigma)
        return x_sigma - sigma.view(-1, 1, 1, 1) * eps_hat


def run_one_checkpoint(
    model_type: str,
    ckpt_path: str,
    cfg: InpaintConfig,
    y_masked, mask
) -> None:
    device = torch.device(cfg.device)
    if model_type == "cd":
        model = build_cd_model_from_checkpoint(ckpt_path, device)
    else:
        model = build_ct_model_from_checkpoint(ckpt_path, device)

    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg_dict = ckpt.get("config", {})
    eps = float(cfg_dict.get("eps", cfg.eps))
    sigma_max = float(cfg_dict.get("sigma_max", cfg.sigma_max))
    rho = float(cfg_dict.get("rho", cfg.rho))
    num_steps = int(cfg_dict.get("num_scales", cfg.num_steps))

    sigmas = build_karras_sigmas(
        sigma_min=eps,
        sigma_max=sigma_max,
        num_scales=num_steps,
        rho=rho,
        device=device,
    )


    omega = 1 - mask

    x_hat = algorithm4_inpaint(
        model=model,
        y_masked=y_masked,
        omega=omega,
        sigmas=sigmas,
        eps=eps,
    )

    stem = Path(ckpt_path).stem
    save_path = str(Path(cfg.save_dir) / f"{stem}_{cfg.mask_type}_inpainting.png")

    return x_hat

def show_method_comparison(
    gt:        Tensor,
    y:         Tensor,
    mask:      Tensor,
    recons:    List[Tensor],
    labels:    List[str],
    save_path: Optional[str] = None
) -> None:
    all_tensors = [gt, y] + recons
    all_labels  = ["Ground truth", "Masked input"] + labels
    all_cmaps   = ["gray", "gray"] + ["gray"] * len(recons)

    all_tensors = [denorm_to_01(t).cpu() for t in all_tensors]
    mask_cpu    = mask.cpu()

    B     = gt.shape[0]
    nrows = len(all_tensors)
    fig, axes = plt.subplots(nrows, B, figsize=(2.8 * B, 2.8 * nrows),
                              gridspec_kw={"wspace": 0.06, "hspace": 0.25})
    if B == 1:
        axes = axes[:, None]

    for i, lbl in enumerate(all_labels):
        color = "#1D9E75" if i >= 2 else "black"
        axes[i, 0].text(
            -0.05, 0.5, lbl,
            transform=axes[i, 0].transAxes,
            fontsize=30, 
            ha="right", va="center",
            color=color,
        )

    for i, (tensor, cmap) in enumerate(zip(all_tensors, all_cmaps)):
        for j in range(B):
            axes[i, j].imshow(tensor[j, 0], cmap=cmap, vmin=0, vmax=1,
                               interpolation="nearest")
            axes[i, j].axis("off")

    for j in range(B):
        axes[1, j].contour(mask_cpu[j, 0].numpy(), levels=[0.5],
                           colors=["red"], linewidths=[0.8], alpha=0.7)




    plt.tight_layout()
    if save_path is not None:
        save_path = str(save_path) 
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def load_teacher(cfg: TeacherConfig, device: torch.device) -> DiffusionTeacher:
    backbone = UNet_Teacher(
        in_channels=cfg.in_channels,
        base_channels=cfg.base_channels,
        channel_mults=cfg.channel_mults,
        num_res_blocks=cfg.num_res_blocks,
        time_emb_dim=cfg.time_emb_dim,
        dropout=cfg.dropout,
    ).to(device)

    teacher = DiffusionTeacher(backbone=backbone, sigma_data=cfg.sigma_data).to(device)

    ckpt = torch.load(cfg.output_dir, map_location=device)

    if "ema_model" not in ckpt:
        raise KeyError("Teacher checkpoint does not contain 'ema_model'.")
    teacher.load_state_dict(ckpt["ema_model"])
    print(f"Loaded EMA teacher from: {cfg.output_dir}")


    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher



def sample_fashionmnist_indices(
    root="./data",
    train=False,
    per_class_counts=None,
):
    """
    Returns indices for selected classes.

    per_class_counts example:
        {9:2, 6:2, 1:2, 7:2, 8:2}  -> total = 10
    """
    if per_class_counts is None:
        per_class_counts = {
            5: 1,
            6: 2, 
            9 : 1,
            1: 1, 
            7: 2,
            8: 1,
            3 :1
        }


    ds = datasets.FashionMNIST(
        root=root,
        train=train,
        download=True,
        transform=transforms.ToTensor(),
    )

    class_to_indices = {c: [] for c in per_class_counts}

    for idx in range(len(ds)):
        _, y = ds[idx]
        if y in class_to_indices:
            class_to_indices[y].append(idx)

    selected = []

    for c, k in per_class_counts.items():
        idxs = torch.tensor(class_to_indices[c])
        perm = idxs[torch.randperm(len(idxs))]
        selected.extend(perm[:k].tolist())

    return selected


if __name__ == "__main__":
    teacher_cfg = TeacherConfig()
    cfg = InpaintConfig()
    loader = build_dataset_loader(cfg)

    device = torch.device(teacher_cfg.device)
    dataset = loader.dataset

    indices = sample_fashionmnist_indices()

    x0 = torch.stack([dataset[i][0] for i in indices]).to(device)
    labels = torch.tensor([dataset[i][1] for i in indices])

    omega = make_inpainting_mask(
        x0,
        mask_type=cfg.mask_type,
        square_size=cfg.square_size,
        random_pixel_prob=cfg.random_pixel_prob,
    )
    y_masked = x0 * (1.0 - omega)
    mask = 1 - omega

    teacher = load_teacher(teacher_cfg, device)

    edm_recon, traj = dps_inpaint_sample(
    model=teacher,
    y=y_masked,
    mask=1-omega,
    sigma_max=teacher_cfg.sigma_max,
    sigma_min=teacher_cfg.eps,
    steps=teacher_cfg.sample_steps,
    rho=teacher_cfg.rho,
    device=device,
    guidance_scale=1e-3,
    num_guidance_steps=1,
    hard_project_x0=True,
)
    

    checkpoints = [
        ("./consistency_models_ckpts/ct_model_final_l1.pt", r"CT ($\ell_1$)"),
        ("./consistency_models_ckpts/ct_model_final_l2.pt", r"CT ($\ell_2$)"),
        ("./consistency_models_ckpts/cd_model_final_l2.pt", r"CD ($\ell_2$)"),
        ("./consistency_models_ckpts/cd_model_final_l1.pt", r"CD ($\ell_1$)")
    ]
    ct_recons = []
    ct_labels = []
    for ckpt_path, label in checkpoints:
        print(f"\nRunning inpainting with checkpoint: {ckpt_path}")
        x_hat = run_one_checkpoint(
            model_type=label.split()[0].lower(),
            ckpt_path=ckpt_path,
            cfg=cfg,
            y_masked=y_masked,
            mask=mask
        )
        ct_recons.append(x_hat)
        ct_labels.append(label)
    show_method_comparison(
        gt        = x0,
        y         = y_masked,
        mask      = mask,
        recons    = [edm_recon] + ct_recons,
        labels    = ["EDM teacher (DPS)"] + ct_labels,
        save_path = f"inverse_problems_results/inpainting_results/inpainting_{cfg.mask_type}_comparison.png",

    )
