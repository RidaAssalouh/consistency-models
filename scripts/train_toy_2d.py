from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from src.datasets.toy_2d import Toy2DDataConfig, build_toy_dataloaders, denormalize_points
from src.sampling.toy_2d_sampler import (
    save_2d_comparison_grid,
    sample_consistency_multi_step,
    sample_consistency_one_step,
    sample_diffusion_teacher,
)
from src.training.train_ct_2d import (
    Consistency2DTrainConfig,
    train_consistency_distillation_2d,
    train_consistency_training_2d,
)
from src.training.train_diffusion_2d import Diffusion2DTrainConfig, train_diffusion_2d


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train diffusion + consistency models on 2D make_moons.")
    parser.add_argument("--output-dir", type=str, default="outputs/toy_2d")
    parser.add_argument("--n-samples", type=int, default=20000)
    parser.add_argument("--data-noise", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--diffusion-epochs", type=int, default=200)
    parser.add_argument("--consistency-epochs", type=int, default=200)

    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--sigma-min", type=float, default=0.002)
    parser.add_argument("--sigma-max", type=float, default=1.0)
    parser.add_argument("--rho", type=float, default=7.0)
    parser.add_argument("--num-scales", type=int, default=40)
    parser.add_argument("--ema-decay", type=float, default=0.999)

    parser.add_argument("--num-plot-samples", type=int, default=3000)
    parser.add_argument("--diffusion-sampling-steps", type=int, default=100)
    parser.add_argument("--consistency-multistep-steps", type=int, default=4)

    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def save_history(history: dict, path: Path) -> None:
    path.write_text(json.dumps(history, indent=2))


def main() -> None:
    args = parse_args()

    torch.manual_seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = Toy2DDataConfig(
        n_samples=args.n_samples,
        noise=args.data_noise,
        batch_size=args.batch_size,
        num_workers=0,
        val_fraction=0.1,
        seed=args.seed,
        normalize=True,
    )

    print("Building toy make_moons dataloaders...")
    train_loader, val_loader, stats = build_toy_dataloaders(data_config)

    diffusion_config = Diffusion2DTrainConfig(
        epochs=args.diffusion_epochs,
        lr=args.lr,
        device=args.device,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        checkpoint_dir=str(output_dir / "checkpoints_diffusion"),
        checkpoint_every=50,
    )

    consistency_cd_config = Consistency2DTrainConfig(
        epochs=args.consistency_epochs,
        lr=args.lr,
        device=args.device,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        num_scales=args.num_scales,
        ema_decay=args.ema_decay,
        loss_type="l2",
        checkpoint_dir=str(output_dir / "checkpoints_consistency_cd"),
        checkpoint_every=50,
    )

    consistency_ct_config = Consistency2DTrainConfig(
        epochs=args.consistency_epochs,
        lr=args.lr,
        device=args.device,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
        rho=args.rho,
        num_scales=args.num_scales,
        ema_decay=args.ema_decay,
        loss_type="l2",
        checkpoint_dir=str(output_dir / "checkpoints_consistency_ct"),
        checkpoint_every=50,
    )

    print("\n=== Training diffusion teacher ===")
    teacher, diffusion_history = train_diffusion_2d(
        train_loader=train_loader,
        val_loader=val_loader,
        config=diffusion_config,
    )
    save_history(diffusion_history, output_dir / "diffusion_history.json")

    print("\n=== Training consistency model by distillation ===")
    cm_cd, ema_cd, cd_history = train_consistency_distillation_2d(
        train_loader=train_loader,
        val_loader=val_loader,
        teacher=teacher,
        config=consistency_cd_config,
    )
    save_history(cd_history, output_dir / "consistency_cd_history.json")

    print("\n=== Training consistency model independently ===")
    cm_ct, ema_ct, ct_history = train_consistency_training_2d(
        train_loader=train_loader,
        val_loader=val_loader,
        config=consistency_ct_config,
    )
    save_history(ct_history, output_dir / "consistency_ct_history.json")

    print("\n=== Generating samples ===")
    device = torch.device(args.device)
    mean = stats["mean"].to(device)
    std = stats["std"].to(device)

    # collect some real validation data for plotting
    real_batch = next(iter(val_loader))[0][: args.num_plot_samples].to(device)
    if real_batch.shape[0] < args.num_plot_samples:
        real_batch = next(iter(train_loader))[0][: args.num_plot_samples].to(device)

    diffusion_samples = sample_diffusion_teacher(
        teacher=teacher,
        num_samples=args.num_plot_samples,
        sigma_max=args.sigma_max,
        sigma_min=args.sigma_min,
        num_steps=args.diffusion_sampling_steps,
    )

    cd_one_step = sample_consistency_one_step(
        model=ema_cd.ema_model,
        num_samples=args.num_plot_samples,
        sigma_max=args.sigma_max,
    )
    ct_one_step = sample_consistency_one_step(
        model=ema_ct.ema_model,
        num_samples=args.num_plot_samples,
        sigma_max=args.sigma_max,
    )

    cd_multi_step = sample_consistency_multi_step(
        model=ema_cd.ema_model,
        num_samples=args.num_plot_samples,
        sigma_max=args.sigma_max,
        sigma_min=args.sigma_min,
        num_steps=args.consistency_multistep_steps,
    )
    ct_multi_step = sample_consistency_multi_step(
        model=ema_ct.ema_model,
        num_samples=args.num_plot_samples,
        sigma_max=args.sigma_max,
        sigma_min=args.sigma_min,
        num_steps=args.consistency_multistep_steps,
    )

    # de-normalize all plotted points back to original moon coordinates
    real_batch_dn = denormalize_points(real_batch, mean, std)
    diffusion_samples_dn = denormalize_points(diffusion_samples, mean, std)
    cd_one_step_dn = denormalize_points(cd_one_step, mean, std)
    ct_one_step_dn = denormalize_points(ct_one_step, mean, std)
    cd_multi_step_dn = denormalize_points(cd_multi_step, mean, std)
    ct_multi_step_dn = denormalize_points(ct_multi_step, mean, std)

    save_2d_comparison_grid(
        real_data=real_batch_dn,
        diffusion_samples=diffusion_samples_dn,
        cd_one_step=cd_one_step_dn,
        ct_one_step=ct_one_step_dn,
        cd_multi_step=cd_multi_step_dn,
        ct_multi_step=ct_multi_step_dn,
        path=output_dir / "toy_2d_comparison.png",
    )

    summary = {
        "output_dir": str(output_dir),
        "diffusion_history_file": str(output_dir / "diffusion_history.json"),
        "consistency_cd_history_file": str(output_dir / "consistency_cd_history.json"),
        "consistency_ct_history_file": str(output_dir / "consistency_ct_history.json"),
        "plot_file": str(output_dir / "toy_2d_comparison.png"),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    print("\nDone.")
    print(f"Saved plot to: {output_dir / 'toy_2d_comparison.png'}")
    print(f"Saved summary to: {output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()