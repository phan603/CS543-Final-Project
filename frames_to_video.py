import os, cv2

in_dir = "frames_out2"
out_vid = "video_test/colorized.mp4"
fps = 24.0

files = sorted([f for f in os.listdir(in_dir) if f.lower().endswith(".png")])
assert files, "no frames found in frames_out"

first = cv2.imread(os.path.join(in_dir, files[0]))
h, w = first.shape[:2]

os.makedirs(os.path.dirname(out_vid), exist_ok=True)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
vw = cv2.VideoWriter(out_vid, fourcc, fps, (w, h))

for f in files:
    img = cv2.imread(os.path.join(in_dir, f))
    vw.write(img)

vw.release()
print("wrote", out_vid, "frames:", len(files))
