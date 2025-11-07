import argparse, os, time
import cv2
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0 (USB), ruta a video o pipeline GStreamer")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--half", type=int, default=1, help="FP16 si usa .pt y hay GPU")
    ap.add_argument("--use_trt", type=int, default=0, help="usar models/best.engine si existe")
    ap.add_argument("--show", type=int, default=1, help="1 = abrir ventana con detecciones")
    return ap.parse_args()

def main():
    args = parse_args()

    # Selección de pesos
    use_engine = bool(args.use_trt) and os.path.exists("models/best.engine")
    weights = "models/best.engine" if use_engine else "models/best.pt"
    print(f"[INFO] Cargando pesos: {weights}")

    model = YOLO(weights)

    # Overrides para predicción
    overrides = dict(imgsz=args.imgsz, conf=args.conf)
    if weights.endswith(".pt"):
        overrides["half"] = bool(args.half)

    # Fuente de video
    src = 0 if str(args.source) == "0" else args.source
    print(f"[INFO] Fuente: {src}")

    os.makedirs("out", exist_ok=True)
    t0, frames = time.time(), 0

    for i, r in enumerate(model.predict(source=src, stream=True, **overrides)):
        frames += 1
        im = r.plot()  # BGR con cajas dibujadas

        # Guardar cada 10 frames (útil si estás headless)
        if i % 10 == 0:
            cv2.imwrite(f"out/frame_{i:06d}.jpg", im)

        # Mostrar ventana si se pide
        if args.show:
            cv2.imshow("YOLO Coins", im)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Log FPS
        if frames % 100 == 0:
            dt = time.time() - t0
            print(f"[INFO] {frames} frames, {frames/dt:.2f} FPS")

    if args.show:
        cv2.destroyAllWindows()
    print("[INFO] Finalizado.")

if __name__ == "__main__":
    main()
