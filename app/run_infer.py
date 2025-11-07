import argparse, os, time
import cv2
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 (USB), ruta video o pipeline GStreamer")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--half", type=int, default=1)
    ap.add_argument("--use_trt", type=int, default=0)
    return ap.parse_args()

def main():
    args = parse_args()

    weights = "models/best.engine" if (args.use_trt and os.path.exists("models/best.engine")) else "models/best.pt"
    print(f"[INFO] Cargando pesos: {weights}")
    model = YOLO(weights)

    overrides = dict(imgsz=args.imgsz, conf=args.conf)
    if weights.endswith(".pt"):
        overrides["half"] = bool(args.half)

    src = 0 if args.source == "0" else args.source
    print(f"[INFO] Fuente: {src}")

    os.makedirs("out", exist_ok=True)
    t0, frames = time.time(), 0

    for i, r in enumerate(model.predict(source=src, stream=True, **overrides)):
        frames += 1

        # Render y guardado periódico para modo headless
        im = r.plot()  # BGR
        if i % 10 == 0:
            cv2.imwrite(f"out/frame_{i:06d}.jpg", im)

        # Log simple
        if frames % 100 == 0:
            dt = time.time() - t0
            print(f"[INFO] {frames} frames, {frames/dt:.2f} FPS")

        # (Opcional) imprimir detecciones
        # for b in r.boxes:
        #     cls_id = int(b.cls[0])
        #     conf = float(b.conf[0])
        #     print(model.names[cls_id], f"{conf:.2f}")

if __name__ == "__main__":
    main()
