#!/usr/bin/env bash
set -euo pipefail

# ------------------------
# Config
# ------------------------
VENV_DIR=".yoloenv"
SOURCE="${1:-0}"         # 0 webcam | /ruta/video.mp4 | pipeline GStreamer
CONF="${2:-0.5}"         # umbral de confianza 0..1
PYTHON_BIN="python3"

echo "[INFO] Fuente: ${SOURCE} | conf=${CONF}"

ARCH="$(uname -m)"       # x86_64 o aarch64
echo "[INFO] Arquitectura: ${ARCH}"

# ------------------------
# Dependencias del sistema (Jetson)
# ------------------------
if [[ "$ARCH" == "aarch64" ]]; then
  echo "[INFO] Detectado Jetson/aarch64. Instalando OpenCV del sistema (puede pedir sudo)..."
  if command -v apt-get >/dev/null 2>&1; then
    sudo apt-get update -y
    sudo apt-get install -y python3-venv python3-pip python3-opencv libgl1-mesa-glx libgtk2.0-0
  else
    echo "[WARN] apt-get no disponible; asegúrate de tener OpenCV del sistema."
  fi
fi

# ------------------------
# Crear venv
# ------------------------
if [[ -d "$VENV_DIR" ]]; then
  echo "[INFO] Reutilizando entorno: $VENV_DIR"
else
  echo "[INFO] Creando entorno: $VENV_DIR"
  if [[ "$ARCH" == "aarch64" ]]; then
    # Para heredar python3-opencv del sistema
    $PYTHON_BIN -m venv "$VENV_DIR" --system-site-packages
  else
    $PYTHON_BIN -m venv "$VENV_DIR"
  fi
fi

# Activar venv
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -m pip install --upgrade pip wheel setuptools

# ------------------------
# Instalar Torch
# ------------------------
have_torch() { python - <<'PY' >/dev/null 2>&1
import torch, sys; print(torch.__version__)
PY
}
if have_torch; then
  echo "[INFO] torch ya está instalado."
else
  if [[ "$ARCH" == "x86_64" ]]; then
    echo "[INFO] Instalando torch/vision/audio CPU (x86_64)..."
    pip install --extra-index-url https://download.pytorch.org/whl/cpu \
      torch torchvision torchaudio
  else
    echo "[INFO] Instalando torch/vision para Jetson (JP4.6/L4T r32.x comunes)..."
    PYV=$($PYTHON_BIN -c "import sys;print(f'cp{sys.version_info.major}{sys.version_info.minor}')")
    BASE="https://developer.download.nvidia.com/compute/redist/jp/v46/pytorch"
    # Candidatos para cp36 y cp38 (Nano suele ser 3.6; algunos usan 3.8)
    CANDIDATES=(
      "$BASE/torch-1.10.0%2Bnv22.01-cp36-cp36m-linux_aarch64.whl"
      "$BASE/torchvision-0.11.1%2Bnv22.01-cp36-cp36m-linux_aarch64.whl"
      "$BASE/torch-1.10.0%2Bnv22.01-cp38-cp38-linux_aarch64.whl"
      "$BASE/torchvision-0.11.1%2Bnv22.01-cp38-cp38-linux_aarch64.whl"
    )
    ok=0
    for url in "${CANDIDATES[@]}"; do
      if [[ "$url" == *"${PYV}"* ]]; then
        echo "[INFO] Probando: $url"
        if pip install "$url"; then ok=1; break; fi
      fi
    done
    if [[ "$ok" -eq 0 ]]; then
      echo "[WARN] No se pudo instalar torch automáticamente para ${PYV}."
      echo "      Revisa la URL de tu wheel según tu JetPack/Python y vuelve a ejecutar."
      exit 1
    fi
  fi
fi

# ------------------------
# Requerimientos del proyecto
# ------------------------
REQ_FILE="requirements.txt"
if [[ ! -f "$REQ_FILE" ]]; then
  echo "[INFO] Generando requirements.txt..."
  # Ultralytics estable + numpy. En Jetson usamos OpenCV del sistema.
  {
    echo "ultralytics==8.0.196"
    echo "numpy==1.26.4"
    if [[ "$ARCH" == "x86_64" ]]; then
      echo "opencv-python"   # en PC, con GUI
    fi
  } > "$REQ_FILE"
fi

echo "[INFO] Instalando requirements..."
pip install -r "$REQ_FILE"

# Mostrar versiones clave
python - <<'PY'
import sys, cv2, numpy, importlib
print("Python:", sys.version.split()[0])
try:
  import torch
  print("Torch:", torch.__version__)
except Exception as e:
  print("Torch: ERROR", e)
print("OpenCV:", cv2.__version__)
print("NumPy:", numpy.__version__)
print("Ultralytics:", importlib.import_module("ultralytics").__version__)
PY

# ------------------------
# Generar detect_cam.py si no existe
# ------------------------
if [[ ! -f "detect_cam.py" ]]; then
  cat > detect_cam.py <<'PY'
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
    weights_path = Path(__file__).with_name("best.pt")
    if not weights_path.exists():
        raise FileNotFoundError(f"No se encontró {weights_path}. Coloca best.pt junto a detect_cam.py")

    model = YOLO(str(weights_path))
    src = 0 if str(args.source) == "0" else args.source

    # En Jetson/CSI puedes pasar un pipeline GStreamer como --source
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError(f"No se pudo abrir la fuente: {src}")

    cv2.namedWindow("YOLO Cam", cv2.WINDOW_NORMAL)
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        results = model(frame, conf=args.conf)
        annotated = results[0].plot()
        cv2.imshow("YOLO Cam", annotated)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
PY
  echo "[INFO] Creado detect_cam.py"
fi

# ------------------------
# Verificar modelo y ejecutar
# ------------------------
if [[ ! -f "best.pt" ]]; then
  echo "[ERROR] No se encontró 'best.pt' en $(pwd)."
  echo "       Debe estar en la MISMA carpeta que run_cam.sh y detect_cam.py."
  exit 1
fi

echo "[INFO] Lanzando cámara..."
python detect_cam.py --source "${SOURCE}" --conf "${CONF}"
