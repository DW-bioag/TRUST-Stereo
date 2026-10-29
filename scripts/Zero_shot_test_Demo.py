

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from omegaconf import OmegaConf


def find_repo_root() -> Path:
    """Work both when this script is placed in the repo root or in scripts/."""
    location = Path(__file__).resolve()
    for candidate in location.parents:
        if (candidate / "core" / "models" / "TRUST_Stereo_Stage2.py").is_file():
            return candidate
    raise FileNotFoundError(
        "Could not find core/models/TRUST_Stereo_Stage2.py. "
        "Put zero_shot_test.py in the TRUST-Stereo repository or its scripts/ folder."
    )


def read_rgb(path: Path, device: torch.device) -> torch.Tensor:
    """RGB, float32, Bx3xHxW, 0..255; same input scale as the evaluation script."""
    if not path.is_file():
        raise FileNotFoundError(f"Image not found: {path}")
    with Image.open(path) as img:
        rgb = np.asarray(img.convert("RGB"), dtype=np.uint8).copy()
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0)
    return tensor.to(device=device, dtype=torch.float32)


def save_pfm(path: Path, disparity: np.ndarray) -> None:
    """Write single-channel PFM with standard bottom-up row ordering."""
    if disparity.ndim != 2:
        raise ValueError(f"PFM requires HxW, got {disparity.shape}")
    arr = np.flipud(disparity).astype("<f4", copy=False)
    with path.open("wb") as stream:
        stream.write(b"Pf\n")
        stream.write(f"{disparity.shape[1]} {disparity.shape[0]}\n".encode("ascii"))
        stream.write(b"-1.0\n")  # negative scale means little-endian
        arr.tofile(stream)


def save_disparity(disparity: np.ndarray, out_dir: Path) -> None:
    if disparity.ndim != 2:
        raise ValueError(f"Expected a 2D disparity map; got {disparity.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)
    disparity = np.asarray(disparity, dtype=np.float32)
    np.save(out_dir / "disparity.npy", disparity)
    save_pfm(out_dir / "disparity.pfm", disparity)

    # 16-bit PNG convention: disparity in pixels multiplied by 256.
    # This is a convenient fixed-point format, not lossless for fractional values.
    disp16 = np.rint(np.clip(np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0),
                                0.0, 65535.0 / 256.0) * 256.0).astype(np.uint16)
    Image.fromarray(disp16).save(out_dir / "disparity_16bit.png")

    valid = np.isfinite(disparity)
    if not valid.any():
        raise ValueError("Prediction has no finite disparity values.")
    # Fixed zero-minimum and percentile-based upper limit for visual contrast.
    vis_max = max(float(np.percentile(np.maximum(disparity[valid], 0), 99)), 1e-6)
    normalized = np.clip(np.nan_to_num(disparity, nan=0.0, posinf=0.0, neginf=0.0),
                         0.0, vis_max) / vis_max
    from matplotlib import colormaps
    colored = np.round(colormaps["turbo"](normalized)[..., :3] * 255).astype(np.uint8)
    colored[~valid] = 0
    Image.fromarray(colored, mode="RGB").save(out_dir / "disparity_vis.png")


def load_checkpoint(model: torch.nn.Module, weights_path: Path) -> None:
    if not weights_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

    # Only load weights you trust. Some training checkpoints include Python objects.
    checkpoint = torch.load(str(weights_path), map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a PyTorch state_dict or checkpoint dictionary.")

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    if not isinstance(state_dict, dict):
        raise TypeError("Checkpoint 'model'/'state_dict' is not a state dictionary.")
    # Support checkpoints saved from DistributedDataParallel.
    state_dict = {
        (k[7:] if isinstance(k, str) and k.startswith("module.") else k): v
        for k, v in state_dict.items()
    }
    # Fail loudly if weights don't match: never silently evaluate a partial model.
    model.load_state_dict(state_dict, strict=True)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run TRUST-Stereo Stage 2 inference on a rectified RGB pair."
    )
    parser.add_argument("--left", type=Path, required=True, help="Left RGB image")
    parser.add_argument("--right", type=Path, required=True, help="Right RGB image")
    parser.add_argument("--weights", type=Path, required=True, help="TRUST .pth checkpoint")
    parser.add_argument(
        "--config", type=Path, default=None,
        help="Training/inference YAML; defaults to the YAML used in the original Middlebury script",
    )
    parser.add_argument("--output_dir", type=Path, default=Path("outputs/zero_shot"))
    parser.add_argument("--iters", type=int, default=4,
                        help="Preserved for compatibility with the original evaluation call")
    return parser.parse_args()


def main():
    args = parse_args()
    if args.iters < 1:
        raise ValueError("--iters must be >= 1")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required by the current TRUST-Stereo model implementation.")

    root = find_repo_root()
    sys.path.insert(0, str(root))

    # Correct Stage 2 model and class (not the old URS import).
    from core.models.TRUST_Stereo_Stage2 import TRUST
    from core.utils.utils import InputPadder

    config_path = args.config or root / "config" / "URS3_sf_M10_4M_zero_shot_stage2.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Config not found: {config_path}. Pass the YAML used to train your checkpoint with --config."
        )
    cfg = OmegaConf.load(str(config_path))
    cfg.valid_iters = args.iters

    device = torch.device("cuda:0")
    left = read_rgb(args.left, device)
    right = read_rgb(args.right, device)
    if left.shape != right.shape:
        raise ValueError(f"Left/right image dimensions must match: {left.shape} vs {right.shape}")
    original_hw = tuple(left.shape[-2:])

    print(f"Loading TRUST-Stereo Stage 2 from {args.weights}")
    model = TRUST(cfg)
    load_checkpoint(model, args.weights)
    model = model.to(device)
    model.eval()

    # Exact padding/inference/unpadding pattern from the Middlebury evaluator.
    padder = InputPadder(left.shape, divis_by=32)
    left, right = padder.pad(left, right)

    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with torch.no_grad():
        start.record()
        predicted = model(left, right, iters=cfg.valid_iters, test_mode=True)
        end.record()
    end.synchronize()

    predicted = padder.unpad(predicted)
    if predicted.ndim == 4 and predicted.shape[0] == 1 and predicted.shape[1] == 1:
        disparity = predicted[0, 0]
    elif predicted.ndim == 3 and predicted.shape[0] == 1:
        disparity = predicted[0]
    else:
        raise ValueError(f"Unexpected model disparity output shape: {tuple(predicted.shape)}")

    if tuple(disparity.shape) != original_hw:
        raise ValueError(f"Output size {tuple(disparity.shape)} != input size {original_hw}")

    prediction_np = disparity.detach().float().cpu().numpy()
    save_disparity(prediction_np, args.output_dir)

    print(f"Input size          : {original_hw[1]} x {original_hw[0]}")
    print(f"Predicted disparity : {prediction_np.shape[1]} x {prediction_np.shape[0]}")
    print(f"Forward time        : {start.elapsed_time(end):.3f} ms")
    print(f"Disparity min / max : {float(np.nanmin(prediction_np)):.3f} / "
          f"{float(np.nanmax(prediction_np)):.3f} pixels")
    print(f"Saved outputs       : {args.output_dir.resolve()}")
    for filename in ["disparity.npy", "disparity.pfm", "disparity_16bit.png", "disparity_vis.png"]:
        print(f"  - {filename}")


if __name__ == "__main__":
    main()
