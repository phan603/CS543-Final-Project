import os
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image

from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import get_image_colorizer


def main():
    in_dir = Path("frames_in")
    out_dir = Path("frames_out2")
    out_dir.mkdir(parents=True, exist_ok=True)

    deoldify_root = Path("DeOldify")
    device.set(DeviceId.GPU0 if torch.cuda.is_available() else DeviceId.CPU)

    colorizer = get_image_colorizer(
        root_folder=deoldify_root,
        artistic=False,
        render_factor=35
    )

    start_idx = 300  # change this if you want
    files = sorted(in_dir.glob("*.png"))[start_idx:]
    assert files, f"no frames found in {in_dir}"

    prev_ab = None
    alpha = 0.90  # higher = smoother color, but no motion blur now (0.85~0.95)

    for i, f in enumerate(files, 1):
        out_path = out_dir / f.name
        if out_path.exists():
            continue

        # DeOldify colorized frame (RGB PIL)
        out_img = colorizer.get_transformed_image(str(f), render_factor=35, post_process=True)
        out_rgb = np.array(out_img)  # HxWx3 RGB uint8

        # original frame (keep sharp luminance from here)
        orig_bgr = cv2.imread(str(f))
        if orig_bgr is None:
            raise RuntimeError(f"failed to read {f}")

        # LAB of DeOldify output (for chroma)
        out_bgr = cv2.cvtColor(out_rgb, cv2.COLOR_RGB2BGR)
        out_lab = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        cur_ab = out_lab[:, :, 1:3]  # A,B channels

        # temporal smoothing on AB only
        if prev_ab is None:
            sm_ab = cur_ab
        else:
            sm_ab = alpha * prev_ab + (1 - alpha) * cur_ab
        prev_ab = sm_ab

        # LAB of original (for L channel)
        orig_lab = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
        out_lab[:, :, 0] = orig_lab[:, :, 0]     # keep sharp L
        out_lab[:, :, 1:3] = sm_ab               # smooth color only

        # back to BGR
        final_bgr = cv2.cvtColor(np.clip(out_lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

        # optional: mild sharpen (uncomment if you want)
        # blur = cv2.GaussianBlur(final_bgr, (0, 0), 1.0)
        # final_bgr = cv2.addWeighted(final_bgr, 1.2, blur, -0.2, 0)

        cv2.imwrite(str(out_path), final_bgr)

        if i % 50 == 0:
            print(f"done {i}/{len(files)}")

    print("all frames colorized into frames_out/")


if __name__ == "__main__":
    main()
