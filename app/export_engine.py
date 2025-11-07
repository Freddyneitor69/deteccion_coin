import argparse
from ultralytics import YOLO

ap = argparse.ArgumentParser()
ap.add_argument("--weights", required=True)
ap.add_argument("--imgsz", type=int, default=640)
args = ap.parse_args()

print("[INFO] Exportando a TensorRT (.engine)...")
model = YOLO(args.weights)
model.export(format="engine", imgsz=args.imgsz, half=True, dynamic=False)  # genera models/best.engine
print("[INFO] Export listo: models/best.engine")
