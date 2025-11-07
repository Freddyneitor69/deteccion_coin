from pathlib import Path
import argparse
import cv2
from ultralytics import YOLO

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="0", help="0=webcam, ruta a video o pipeline GStreamer")
    ap.add_argument("--conf", type=float, default=0.5)
    return ap.parse_args()

def main():
    args = parse_args()

    # best.pt en la MISMA carpeta que este script
    weights_path = Path(__file__).with_name("best.pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"No se encontró {weights_path}. Coloca best.pt junto a detect_cam.py")

    # Cargar modelo
    model = YOLO(str(weights_path))

    # Fuente de video
    src = 0 if str(args.source) == "0" else args.source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {src}")

    cv2.namedWindow("YOLO Cam", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Predicción (conf por argumento)
        results = model(frame, conf=args.conf)

        # Dibujar detecciones
        annotated = results[0].plot()

        cv2.imshow("YOLO Cam", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
