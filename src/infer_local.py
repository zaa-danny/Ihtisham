from pathlib import Path

import cv2
import torch
from inference import infer_image, load_models

device = "cuda" if torch.cuda.is_available() else "cpu"

if (
    device == "cpu"
    and hasattr(torch.backends, "mps")
    and torch.backends.mps.is_available()
):
    device = "mps"

deepc, refinenet = load_models(
    "./reference/longrun-epoch=99-step=369700.ckpt",
    "./reference/second-refinenet-epoch-100-step=373k.ckpt",
    n_ids=16,
    device=device,
)

image_folder = Path("my_pictures")
output_folder = Path("results")
output_folder.mkdir(exist_ok=True)

extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
image_paths = [
    path for path in image_folder.iterdir()
    if path.is_file() and path.suffix.lower() in extensions
]

if not image_paths:
    raise FileNotFoundError(
        f"No supported images were found in {image_folder.resolve()}. "
        "Add image files to the 'src/my_pictures' folder."
    )

for image_path in image_paths:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"Skipping unreadable image: {image_path}")
        continue

    keypoints, output_image = infer_image(
        image,
        16,
        deepc,
        refinenet,
        draw_pred=True,
        device=device,
    )

    output_path = output_folder / (
        f"{image_path.stem}_{image_path.suffix[1:]}_result.jpg"
    )
    cv2.imwrite(str(output_path), output_image)

    print(
        f"{image_path.name}: detected {len(keypoints)} corners; "
        f"saved {output_path}"
    )