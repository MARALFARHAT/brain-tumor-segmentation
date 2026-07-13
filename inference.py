import argparse
import os
import tempfile
from difflib import get_close_matches
from pathlib import Path

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from model import UNet


class GradCAM:
    def __init__(self, model: UNet, target_layer):
        self.model = model
        self.activations = None
        self.gradients = None

        def save_activations(_, __, output):
            self.activations = output.detach()

        def save_gradients(_, __, grad_output):
            self.gradients = grad_output[0].detach()

        target_layer.register_forward_hook(save_activations)
        target_layer.register_full_backward_hook(save_gradients)

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        logits = self.model(input_tensor)

        probs = F.softmax(logits, dim=1)[:, 1]
        score = probs.max()
        self.model.zero_grad()
        score.backward()

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)

        cam = F.interpolate(
            cam,
            size=input_tensor.shape[-2:],
            mode="bilinear",
            align_corners=False,
        )
        cam = cam.squeeze().cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam


def load_model(checkpoint_path: str, device: str) -> UNet:
    model = UNet(in_channels=1, num_classes=2).to(device)
    ckpt = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(
        f"Loaded checkpoint (epoch {ckpt.get('epoch', '?')}, "
        f"val Dice={ckpt.get('val_dice', 0):.4f})"
    )
    return model


def resolve_image_path(image_path: str) -> Path:
    candidate = Path(image_path).expanduser()
    if candidate.exists():
        return candidate

    candidate = (Path.cwd() / image_path).resolve()
    if candidate.exists():
        return candidate

    dataset_root = Path.cwd() / "lgg-mri-segmentation" / "kaggle_3m"
    if dataset_root.exists():
        basename = Path(image_path).name
        matches = [
            path
            for path in dataset_root.rglob(basename)
            if path.is_file() and not path.name.endswith("_mask.tif")
        ]
        if len(matches) == 1:
            print(f"Resolved '{image_path}' to '{matches[0]}'")
            return matches[0]

        tif_files = [
            path
            for path in dataset_root.rglob("*.tif")
            if path.is_file() and not path.name.endswith("_mask.tif")
        ]
        names = [path.name for path in tif_files]
        close = get_close_matches(basename, names, n=3, cutoff=0.6)
        if close:
            suggestions = [str(next(path for path in tif_files if path.name == name)) for name in close]
            raise FileNotFoundError(
                f"Image not found: '{image_path}'.\n"
                f"Closest dataset matches:\n- " + "\n- ".join(suggestions)
            )

    raise FileNotFoundError(
        f"Image not found: '{image_path}'. "
        "Pass a valid MRI slice path or omit --image to use demo mode."
    )


def preprocess(image_path: str, img_size: int = 256) -> tuple[torch.Tensor, Path]:
    resolved_path = resolve_image_path(image_path)
    img = Image.open(resolved_path).convert("L").resize((img_size, img_size))
    arr = np.array(img, dtype=np.float32) / 255.0
    return torch.tensor(arr).unsqueeze(0).unsqueeze(0), resolved_path


def predict(model, input_tensor, device) -> tuple:
    input_tensor = input_tensor.to(device)
    with torch.no_grad():
        logits = model(input_tensor)
    probs = torch.softmax(logits, dim=1)[0, 1].cpu().numpy()
    pred = (probs > 0.5).astype(np.uint8)
    return pred, probs


def visualise(image_np, pred_mask, prob_map, gradcam_map, save_path):
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))

    axes[0].imshow(image_np, cmap="gray")
    axes[0].set_title("Input MRI", fontweight="bold")

    axes[1].imshow(image_np, cmap="gray")
    overlay = np.zeros((*image_np.shape, 4))
    overlay[pred_mask == 1] = [1, 0, 0, 0.45]
    axes[1].imshow(overlay)
    patch = mpatches.Patch(color=(1, 0, 0, 0.6), label="Predicted tumor")
    axes[1].legend(handles=[patch], loc="lower left", fontsize=8)
    axes[1].set_title("Segmentation Overlay", fontweight="bold")

    im = axes[2].imshow(prob_map, cmap="hot", vmin=0, vmax=1)
    plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    axes[2].set_title("Tumor Probability Map", fontweight="bold")

    axes[3].imshow(image_np, cmap="gray")
    axes[3].imshow(gradcam_map, cmap="jet", alpha=0.5)
    axes[3].set_title("Grad-CAM Attention", fontweight="bold")

    for ax in axes:
        ax.axis("off")

    plt.suptitle("Brain Tumor Segmentation - Inference Results", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {save_path}")


def generate_demo_image(img_size=256, seed=7):
    rng = np.random.default_rng(seed)
    img = rng.uniform(0, 0.3, (img_size, img_size)).astype(np.float32)
    cx, cy = img_size // 2, img_size // 2
    y_grid, x_grid = np.ogrid[:img_size, :img_size]
    brain = ((x_grid - cx) ** 2 / 90**2 + (y_grid - cy) ** 2 / 80**2) < 1
    img[brain] += rng.uniform(0.3, 0.65, brain.sum())
    tx, ty, tr = cx + 30, cy - 20, 25
    tumor = ((x_grid - tx) ** 2 + (y_grid - ty) ** 2) < tr**2
    img[tumor & brain] = np.clip(img[tumor & brain] + 0.35, 0, 1)
    img = np.clip(img + rng.normal(0, 0.015, img.shape), 0, 1)
    demo_path = os.path.join(tempfile.gettempdir(), "demo_mri.png")
    Image.fromarray((img * 255).astype(np.uint8)).save(demo_path)
    return demo_path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image", type=str, default=None, help="Path to MRI slice (grayscale). Omit for demo.")
    parser.add_argument("--output", type=str, default="inference_output.png")
    parser.add_argument("--img_size", type=int, default=256)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    image_path = args.image or generate_demo_image(args.img_size)

    model = load_model(args.checkpoint, device)
    gradcam = GradCAM(model, target_layer=model.bottleneck.block[-1])

    input_tensor, resolved_path = preprocess(image_path, args.img_size)
    print(f"Running inference on: {resolved_path}")
    pred_mask, prob_map = predict(model, input_tensor, device)

    input_grad = input_tensor.to(device).requires_grad_(True)
    cam_map = gradcam.generate(input_grad)

    image_np = input_tensor[0, 0].detach().numpy()
    visualise(image_np, pred_mask, prob_map, cam_map, args.output)


if __name__ == "__main__":
    main()
