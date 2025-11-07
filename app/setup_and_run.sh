#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"  # ir a app/

echo "[INFO] Verificando Docker..."
command -v docker >/dev/null || { echo "[ERROR] Docker no está instalado"; exit 1; }

echo "[INFO] Verificando runtime NVIDIA..."
if ! docker info 2>/dev/null | grep -q 'Runtimes: .*nvidia'; then
  echo "[WARN] No se detecta runtime NVIDIA."
  echo "      Instala nvidia-container-toolkit y reinicia Docker:"
  echo "      sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker"
fi

# Modelos están una carpeta arriba
if [ ! -f "../models/best.pt" ] && [ ! -f "../models/best.engine" ]; then
  echo "[ERROR] Falta ../models/best.pt (o best.engine)"; exit 1
fi

# Habilitar X11 para mostrar ventana
if command -v xhost >/dev/null; then
  xhost +local:root >/dev/null 2>&1 || true
else
  echo "[WARN] 'xhost' no encontrado; si no ves ventana, instala x11-xserver-utils"
fi

mkdir -p ../out

echo "[INFO] Construyendo imagen (Jetson)..."
docker compose build

echo "[INFO] Levantando servicio..."
docker compose up -d

echo "[INFO] Logs en vivo (Ctrl+C para salir):"
docker logs -f jetson-yolo-coins
