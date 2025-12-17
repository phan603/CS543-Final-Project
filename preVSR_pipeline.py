import os
from pathlib import Path

import torch
import numpy as np
import cv2
from PIL import Image
from deoldify import device
from deoldify.device_id import DeviceId
from deoldify.visualize import get_image_colorizer
import skimage
from scipy.ndimage import generic_filter
from sklearn.neighbors import NearestNeighbors
from skimage.feature import canny, blob_log
from skimage.morphology import dilation, disk
from skimage.filters import gaussian

def blob_detection(gray, freq, texture, min_radius, max_radius, num_sigma, threshold, texture_thresh=0.5):
    blobs = blob_log(
        freq,
        min_sigma=min_radius / np.sqrt(2),
        max_sigma=max_radius / np.sqrt(2),
        num_sigma=num_sigma,
        threshold=threshold
    )
    blobs[:, 2] *= np.sqrt(2)

    filtered_blobs = []
    for y, x, r in blobs:
        if texture[int(y), int(x)] < texture_thresh:
            filtered_blobs.append((y, x, r))

    filtered_blobs = np.array(filtered_blobs)

    coords = np.array([[y, x] for y, x, _ in filtered_blobs])
    
    nbrs = NearestNeighbors(radius=10).fit(coords)
    neighbors = nbrs.radius_neighbors(coords, return_distance=False)

    final_blobs = []
    for i, neigh in enumerate(neighbors):
        if len(neigh) < 6:
            final_blobs.append(filtered_blobs[i])

    final_blobs = np.array(final_blobs)

    edges = canny(gray, sigma=3.0)
    edges = dilation(edges, disk(7))

    edge_filtered = []
    for y, x, r in final_blobs:
        if not edges[int(y), int(x)]:
            edge_filtered.append((y, x, r))
    edge_filtered = np.array(edge_filtered)
    return edge_filtered

def preProcess(in_dir,out_dir):
    in_dir = Path(in_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    deoldify_root = Path("DeOldify")
    device.set(DeviceId.GPU0 if torch.cuda.is_available() else DeviceId.CPU)

    colorizer = get_image_colorizer(
        root_folder=deoldify_root,
        artistic=False,
        render_factor=35
    )
    MAX_SMALL_RADIUS = 6
    MIN_SMALL_RADIUS = 1.5
    NUM_SIGMA_SMALL = 12
    THRESHOLD_SMALL = 0.02
    TEXTURE_THRESHOLD_SMALL = 0.5

    MAX_LARGE_RADIUS = 16
    MIN_LARGE_RADIUS = 8
    NUM_SIGMA_LARGE = 32
    THRESHOLD_LARGE = 0.01
    TEXTURE_THRESHOLD_LARGE = 0.7
    contrast_threshold = .25

    start_idx = 0  # change this if you want
    files = sorted(in_dir.glob("*.png"))[start_idx:]
    assert files, f"no frames found in {in_dir}"

    prev_ab = None
    alpha = 0.90  # higher = smoother color, but no motion blur now (0.85~0.95)

    for i, f in enumerate(files, 1):
        out_path = out_dir / f.name
        image = Image.open(str(f))
        image = np.array(image)
        H,W,_ = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        texture = generic_filter(gray, np.std, size=7)
        texture /= texture.max()
        low_freq = gaussian(gray, sigma=2)
        high_freq = gray - low_freq
        edge_filtered_small = blob_detection(gray, high_freq, texture, MIN_SMALL_RADIUS, MAX_SMALL_RADIUS, NUM_SIGMA_SMALL, THRESHOLD_SMALL, TEXTURE_THRESHOLD_SMALL)
        edge_filtered_large = blob_detection(gray, high_freq, texture, MIN_LARGE_RADIUS, MAX_LARGE_RADIUS, NUM_SIGMA_LARGE, THRESHOLD_LARGE, TEXTURE_THRESHOLD_LARGE)
        filtered_large_blobs = []
        for j in range(len(edge_filtered_large)):
            y,x,r = edge_filtered_large[j]
            y,x,r = int(y),int(x),int(r)
            y0 = max(0, y - r)
            y1 = min(H, y + r + 1)
            x0 = max(0, x - r)
            x1 = min(W, x + r + 1)
            patch = gray[y0:y1, x0:x1]

            if patch.std()/patch.mean() > contrast_threshold:
                filtered_large_blobs.append([y,x,r])

        filtered_large_blobs = np.array(filtered_large_blobs)

        all_detected_blobs = np.concatenate([edge_filtered_small, filtered_large_blobs], axis=0)
        mask = np.zeros(gray.shape, dtype=np.uint8)

        for y, x, r in all_detected_blobs:
            cv2.circle(mask, (int(x), int(y)), int(r), 255, -1)
        restored = cv2.inpaint(image, mask, 3, cv2.INPAINT_NS)
        cv2.imwrite(str(out_path),restored)
        # DeOldify colorized frame (RGB PIL)
        out_img = colorizer.get_transformed_image(str(out_path), render_factor=35, post_process=True)
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

    print("all frames preprocessed into ",out_dir)
