import os, cv2

inp = "historical_footage/00010330.mp4"
out_dir = "frames_in"
os.makedirs(out_dir, exist_ok=True)

cap = cv2.VideoCapture(inp)
fps = cap.get(cv2.CAP_PROP_FPS)
i = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    cv2.imwrite(os.path.join(out_dir, f"{i:06d}.png"), frame)
    i += 1
cap.release()
print("saved", i, "frames. fps =", fps)
