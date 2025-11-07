#!/usr/bin/env bash
set -e

echo "[INFO] Iniciando contenedor..."
python3 - <<'PY'
import torch
print("[INFO] CUDA disponible:", torch.cuda.is_available())
PY

# Exporta a TensorRT si se pide y no existe
if [ "${USE_TRT}" = "1" ] && [ ! -f "models/best.engine" ]; then
  echo "[INFO] Exportando a TensorRT (.engine) por primera vez..."
  python3 app/export_engine.py --weights models/best.pt --imgsz ${IMG_SIZE}
fi

# Ejecuta inferencia
python3 app/run_infer.py \
  --source "${INFER_SOURCE}" \
  --imgsz ${IMG_SIZE} \
  --conf ${CONF_THRES} \
  --half ${HALF} \
  --use_trt ${USE_TRT}
