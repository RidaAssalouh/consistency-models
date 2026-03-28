import copy
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, Optional, List

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.cd_model_utils import UNet as UNet_CD
from models.ct_model_utils import UNet as UNet_CT
from models.teacher_utils import UNet as UNet_Teacher


# Config

@dataclass
class SuperResConfig:
    data_root: str = "./data"
    save_dir: str = "./superres_results"

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

    num_examples: int = 8
    class_filter: Optional[int] = None

    patch_size: int = 4


# Utils

def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def denorm_to_01(x: Tensor) -> Tensor:
    return ((x.clamp(-1.0, 1.0) + 1.0) / 2.0).clamp(0.0, 1.0)


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


# Consistency models

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
        sigma = sigma.float().clamp(min=self.eps)
        c_skip = self.c_skip(sigma).view(-1, 1, 1, 1)
        c_out = self.c_out(sigma).view(-1, 1, 1, 1)
        F_out = self.backbone(x, sigma)
        return c_skip * x + c_out * F_out


# Checkpoint loaders

def build_cd_model_from_checkpoint(ckpt_path: str, device: torch.device) -> ConsistencyModel_CD:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("config", {})

    backbone = UNet_CD(
        in_channels=cfg_dict.get("in_channels", 1),
        base_channels=cfg_dict.get("base_channels", 64),
        channel_mults=tuple(cfg_dict.get("channel_mults", [1, 2, 4])),
        num_res_blocks=cfg_dict.get("num_res_blocks", 2),
        time_emb_dim=cfg_dict.get("time_emb_dim", 256),
        dropout=cfg_dict.get("dropout", 0.0),
    ).to(device)

    model = ConsistencyModel_CD(
        backbone=backbone,
        sigma_data=cfg_dict.get("sigma_data", 0.5),
        eps=cfg_dict.get("eps", 1e-3),
    ).to(device)

    state_dict = ckpt.get("ema_model") or ckpt.get("model")
    if state_dict is None:
        raise KeyError(f"Could not find 'ema_model' or 'model' in checkpoint: {ckpt_path}")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def build_ct_model_from_checkpoint(ckpt_path: str, device: torch.device) -> ConsistencyModel_CT:
    ckpt = torch.load(ckpt_path, map_location=device)
    cfg_dict = ckpt.get("config", {})

    backbone = UNet_CT(
        in_channels=cfg_dict.get("in_channels", 1),
        base_channels=cfg_dict.get("base_channels", 64),
        channel_mults=tuple(cfg_dict.get("channel_mults", [1, 2, 4])),
        num_res_blocks=cfg_dict.get("num_res_blocks", 2),
        time_emb_dim=cfg_dict.get("time_emb_dim", 256),
        dropout=cfg_dict.get("dropout", 0.0),
    ).to(device)

    model = ConsistencyModel_CT(
        backbone=backbone,
        sigma_data=cfg_dict.get("sigma_data", 0.5),
        eps=cfg_dict.get("eps", 1e-3),
    ).to(device)

    state_dict = ckpt.get("ema_model") or ckpt.get("model")
    if state_dict is None:
        raise KeyError(f"Could not find 'ema_model' or 'model' in checkpoint: {ckpt_path}")

    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# Teacher model

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
    """
    Preconditioned noise predictor: given x_t = x0 + sigma*eps, predicts eps.

    Forward output:  eps_hat  (noise prediction)
    Tweedie estimate: x0_hat = x_t - sigma * eps_hat
    """
    def __init__(self, backbone: nn.Module, sigma_data: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.sigma_data = sigma_data

    def c_in(self, sigma: Tensor) -> Tensor:
        return 1.0 / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def c_out(self, sigma: Tensor) -> Tensor:
        return sigma / torch.sqrt(sigma ** 2 + self.sigma_data ** 2)

    def forward(self, x: Tensor, sigma: Tensor) -> Tensor:
        """Returns eps_hat (noise prediction)."""
        sigma = sigma.float()
        cin = self.c_in(sigma).view(-1, 1, 1, 1)
        cout = self.c_out(sigma).view(-1, 1, 1, 1)
        raw = self.backbone(cin * x, sigma)
        return cout * raw

    def predict_x0(self, x_t: Tensor, sigma: Tensor) -> Tensor:
        """Tweedie denoising estimate: x0_hat = x_t - sigma * eps_hat."""
        eps_hat = self(x_t, sigma)
        return x_t - sigma.view(-1, 1, 1, 1) * eps_hat


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


# Patch orthogonal transform  A  and  A^{-1}

def build_patch_orthogonal_matrix(patch_size: int, device: torch.device) -> Tensor:
    """
    Build Q in R^{p^2 x p^2} whose first *column* is the normalised all-ones
    vector (norm = 1 when p^2 entries are each 1/p, because sum of squares = p^2/p^2 = 1).
    The first coordinate of A(x) is then proportional to the patch average.
    """
    d = patch_size * patch_size
    first_col = torch.ones(d, device=device) / patch_size
    M = torch.randn(d, d, device=device)
    M[:, 0] = first_col
    Q, _ = torch.linalg.qr(M, mode="reduced")
    if torch.dot(Q[:, 0], first_col) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q


def A_transform(x: Tensor, Q: Tensor, patch_size: int) -> Tensor:
    """
    x : (B, 1, H, W)
    Returns latent z : (B, H/p, W/p, p^2)
    Each p×p patch is flattened and projected by Q.
    """
    B, C, H, W = x.shape
    assert C == 1
    p = patch_size
    assert H % p == 0 and W % p == 0
    hp, wp = H // p, W // p

    patches = x.unfold(2, p, p).unfold(3, p, p)        # (B,1,hp,wp,p,p)
    patches = patches.contiguous().view(B, hp, wp, p * p)
    return torch.einsum("bhwd,dk->bhwk", patches, Q)


def A_inverse(z: Tensor, Q: Tensor, patch_size: int) -> Tensor:
    """
    z : (B, H/p, W/p, p^2)
    Returns x : (B, 1, H, W)
    Q is orthogonal, so A^{-1} = Q^T applied patch-wise, then fold.
    """
    B, hp, wp, d = z.shape
    p = patch_size
    assert d == p * p

    # Q^T via the einsum transpose
    patches = torch.einsum("bhwk,dk->bhwd", z, Q)      # multiply by Q^T
    patches = patches.view(B, hp, wp, p, p)

    # Fold patches back into image — vectorised version
    x = patches.permute(0, 1, 3, 2, 4).contiguous()    # (B, hp, p, wp, p)
    x = x.view(B, 1, hp * p, wp * p)
    return x


def make_lowres_reference(x: Tensor, Q: Tensor, patch_size: int) -> Tuple[Tensor, Tensor]:
    """
    Build the constrained reference image and the mask Omega.

    Latent-space convention:
      coord 0           → known   (patch average, observed from low-res image)
      coords 1..p^2-1   → unknown (high-frequency detail)

    Returns:
      y_ref_fullres : (B,1,H,W)  image built from known coords only (zeros elsewhere)
      omega         : (B,hp,wp,p^2)  1 = unknown, 0 = known
    """
    latent = A_transform(x, Q, patch_size)
    omega = torch.ones_like(latent)
    omega[..., 0] = 0.0                         

    latent_ref = latent * (1.0 - omega)             # zero out unknown coords
    y_ref_fullres = A_inverse(latent_ref, Q, patch_size)
    return y_ref_fullres, omega


# DPS for super-resolution using the teacher
def dps_superres_patch_sample(
    model: DiffusionTeacher,
    y_latent_known: Tensor,     
    omega: Tensor,             
    Q: Tensor,
    patch_size: int,
    sigma_max: float,
    sigma_min: float,
    steps: int,
    rho: float,
    device: torch.device,
    guidance_scale: float = 1.0,
) -> Tensor:
    """
    Diffusion Posterior Sampling (DPS) for the patch super-resolution inverse problem.

    Measurement model:
        A(x0)[..., 0] = y_latent_known[..., 0]      (patch-average constraint)

    At each Karras step i  (sigma_hi → sigma_lo):
        1. Compute Tweedie estimate:
               eps_hat = model(x_t, sigma_hi)
               x0_hat  = x_t - sigma_hi * eps_hat
        2. Compute measurement loss in latent space (known coords only):
               loss = 0.5 * || A(x0_hat)[..., 0] - y_known[..., 0] ||^2
        3. DPS gradient w.r.t. x_t:
               grad = ∇_{x_t} loss
        4. DDIM update step + DPS correction:
               x_{t-1} = x0_hat + sigma_lo * eps_hat - guidance_scale * grad

    The gradient flows through the Tweedie estimate.
    """
    model.eval()

    B, hp, wp, d = y_latent_known.shape
    H = hp * patch_size
    W = wp * patch_size


    known_coord_mask = (1.0 - omega)              

    sigmas = build_karras_sigmas(sigma_min, sigma_max, steps, rho, device)


    x = sigma_max * torch.randn((B, 1, H, W), device=device)

    for i in range(len(sigmas) - 1):
        sigma_hi = sigmas[i]
        sigma_lo = sigmas[i + 1]
        sigma_batch = torch.full((B,), sigma_hi.item(), device=device)



        x = x.detach().requires_grad_(True)

        # Noise prediction (teacher forward pass)
        eps_hat = model(x, sigma_batch)                    
        # Tweedie denoising estimate (differentiable w.r.t. x)
        x0_hat = x - sigma_hi * eps_hat                    

        # Project Tweedie estimate into latent space
        latent_hat = A_transform(x0_hat, Q, patch_size)  

        # Measurement residual
        residual = (latent_hat - y_latent_known) * known_coord_mask
        loss = 0.5 * (residual ** 2).sum() / B


        # Step 3: DPS gradient  ∇_{x_t} loss
        grad = torch.autograd.grad(loss, x)[0]             


        # Step 4: DDIM deterministic update + DPS correction
        # x_{t-1} = x0_hat + sigma_lo * eps_hat - zeta * grad
        with torch.no_grad():
            x_next = x0_hat + sigma_lo * eps_hat - guidance_scale * grad

        x = x_next

    # Final denoising step at sigma_min
    with torch.no_grad():
        sigma_batch = torch.full((B,), sigmas[-1].item(), device=device)
        x_final = model.predict_x0(x, sigma_batch)

    return x_final.clamp(-1.0, 1.0)



@torch.no_grad()
def algorithm4_superres_grayscale(
    model,
    y_ref_fullres: Tensor,
    omega: Tensor,
    Q: Tensor,
    patch_size: int,
    sigmas: Tensor,
    eps: float,
) -> Tensor:

    device = y_ref_fullres.device
    B = y_ref_fullres.shape[0]

    y_latent = A_transform(y_ref_fullres, Q, patch_size)
    y_latent = y_latent * (1.0 - omega)
    y = A_inverse(y_latent, Q, patch_size)

    t1 = sigmas[0]
    x = y + t1 * torch.randn_like(y)

    x = model(x, torch.full((B,), t1.item(), device=device))

    x_latent = A_transform(x, Q, patch_size)
    x_latent = y_latent * (1.0 - omega) + x_latent * omega
    x = A_inverse(x_latent, Q, patch_size)

    for tn in sigmas[1:]:
        noise_std = torch.sqrt(torch.clamp(tn ** 2 - eps ** 2, min=0.0))
        x = x + noise_std * torch.randn_like(x)
        x = model(x, torch.full((B,), tn.item(), device=device))

        x_latent = A_transform(x, Q, patch_size)
        x_latent = y_latent * (1.0 - omega) + x_latent * omega
        x = A_inverse(x_latent, Q, patch_size)

    return x.clamp(-1.0, 1.0)


# Visualisation
def show_method_comparison(
    gt: Tensor,
    y: Tensor,
    recons: List[Tensor],
    labels: List[str],
    save_path: Optional[str] = None,
) -> None:
    all_tensors = [gt, y] + recons
    all_labels = ["Ground truth", "Low-resolution input"] + labels

    all_tensors = [denorm_to_01(t).cpu() for t in all_tensors]

    B = gt.shape[0]
    nrows = len(all_tensors)
    fig, axes = plt.subplots(
        nrows, B,
        figsize=(2.8 * B, 2.8 * nrows),
        gridspec_kw={"wspace": 0.06, "hspace": 0.25},
    )
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

    for i, tensor in enumerate(all_tensors):
        for j in range(B):
            axes[i, j].imshow(tensor[j, 0], cmap="gray", vmin=0, vmax=1,
                              interpolation="nearest")
            axes[i, j].axis("off")

    plt.tight_layout()
    if save_path is not None:
        save_path = str(save_path)
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
        print(f"Saved: {save_path}")
    plt.show()


def show_results(
    originals: Tensor,
    lowres_ref: Tensor,
    recon: Tensor,
    title: str,
    save_path: Optional[str] = None,
) -> None:
    originals = denorm_to_01(originals).cpu()
    lowres_ref = denorm_to_01(lowres_ref).cpu()
    recon = denorm_to_01(recon).cpu()

    b = originals.shape[0]
    fig, axes = plt.subplots(b, 3, figsize=(8, 2.5 * b))
    if b == 1:
        axes = axes[None, :]

    for j, h in enumerate(["Original", "Low-res reference", "Super-resolved"]):
        axes[0, j].set_title(h, fontsize=11)

    for i in range(b):
        axes[i, 0].imshow(originals[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 1].imshow(lowres_ref[i, 0], cmap="gray", vmin=0, vmax=1)
        axes[i, 2].imshow(recon[i, 0], cmap="gray", vmin=0, vmax=1)
        for j in range(3):
            axes[i, j].axis("off")

    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()


# Dataset
def build_dataset_loader(cfg: SuperResConfig) -> DataLoader:
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
        idxs = [i for i, (_, label) in enumerate(ds) if int(label) == int(cfg.class_filter)]
        subset = torch.utils.data.Subset(ds, idxs)

    return DataLoader(
        subset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )



def run_one_checkpoint(
    model_type: str,
    ckpt_path: str,
    cfg: SuperResConfig,
    lowres_ref: Tensor,
    Q: Tensor,
    omega: Tensor,
) -> Tensor:
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

    return algorithm4_superres_grayscale(
        model=model,
        y_ref_fullres=lowres_ref,
        omega=omega,
        Q=Q,
        patch_size=cfg.patch_size,
        sigmas=sigmas,
        eps=eps,
    )


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
            5: 1,  # ankle boot
            6: 2,  # shirt
            9 : 1,
            1: 1,  # trouser
            7: 2,  # sneaker
            8: 1,  # bag
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


# Main

def main():
    teacher_cfg = TeacherConfig()
    cfg = SuperResConfig()
    loader = build_dataset_loader(cfg)

    device = torch.device(teacher_cfg.device)

    dataset = loader.dataset

    indices = sample_fashionmnist_indices()

    x0 = torch.stack([dataset[i][0] for i in indices]).to(device)
    labels = torch.tensor([dataset[i][1] for i in indices])

    p = cfg.patch_size
    if cfg.image_size % p != 0:
        raise ValueError(f"patch_size={p} must divide image_size={cfg.image_size}")


    Q = build_patch_orthogonal_matrix(p, device)


    lowres_ref, omega = make_lowres_reference(x0, Q, p)


    y_latent_known = A_transform(x0, Q, p) * (1.0 - omega)


    teacher = load_teacher(teacher_cfg, device)
    edm_recon = dps_superres_patch_sample(
        model=teacher,
        y_latent_known=y_latent_known,
        omega=omega,
        Q=Q,
        patch_size=p,
        sigma_max=teacher_cfg.sigma_max,
        sigma_min=teacher_cfg.eps,
        steps=teacher_cfg.sample_steps,
        rho=teacher_cfg.rho,
        device=device,
        guidance_scale=10,        
    )

    checkpoints = [
        ("./consistency_models_ckpts/ct_model_final_l1.pt", r"CT ($\ell_1$)"),
        ("./consistency_models_ckpts/ct_model_final_l2.pt", r"CT ($\ell_2$)"),
        ("./consistency_models_ckpts/cd_model_final_l2.pt", r"CD ($\ell_2$)"),
        ("./consistency_models_ckpts/cd_model_final_l1.pt", r"CD ($\ell_1$)"),
    ]

    ct_recons, ct_labels = [], []
    for ckpt_path, label in checkpoints:
        print(f"\nRunning super-resolution with checkpoint: {ckpt_path}")
        x_hat = run_one_checkpoint(
            model_type=label.split()[0].lower(),    # "ct" or "cd"
            ckpt_path=ckpt_path,
            cfg=cfg,
            lowres_ref=lowres_ref,
            Q=Q,
            omega=omega,
        )
        ct_recons.append(x_hat)
        ct_labels.append(label)

    # Visualise
    show_method_comparison(
        gt=x0,
        y=lowres_ref,
        recons=[edm_recon] + ct_recons,
        labels=["EDM (DPS)"] + ct_labels,
        save_path="superres_results/superres_comparison.png",
    )


if __name__ == "__main__":
    main()