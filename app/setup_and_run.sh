#!/usr/bin/env bash
set -e

echo "[INFO] Verificando Docker..."
if ! command -v docker &>/dev/null; then
  echo "[ERROR] Docker no está instalado"; exit 1
fi

echo "[INFO] Verificando runtime NVIDIA..."
if ! docker info 2>/dev/null | grep -q 'Runtimes: .*nvidia'; then
  echo "[WARN] No se detecta runtime NVIDIA."
  echo "      Instala nvidia-container-toolkit y reinicia Docker:"
  echo "      sudo apt-get install -y nvidia-container-toolkit && sudo systemctl restart docker"
fi

if [ ! -f "models/best.pt" ]; then
  echo "[ERROR] Falta models/best.pt"; exit 1
fi

mkdir -p out

echo "[INFO] Construyendo imagen..."
docker compose build

echo "[INFO] Arrancando servicio..."
docker compose up -d

echo "[INFO] Logs en vivo (Ctrl+C para salir):"
docker logs -f jetson-yolo-coins
