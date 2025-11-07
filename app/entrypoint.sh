#!/usr/bin/env bash
set -e

echo "[INFO] Iniciando contenedor..."
python3 - <<'PY'
import torch
print("[INFO] CUDA disponible:", torch.cuda.is_available())
PY

# Export a TensorRT si se pide
if [ "${USE_TRT}" = "1" ] && [ ! -f "models/best.engine" ]; then
  echo "[INFO] Exportando a TensorRT (.engine) por primera vez..."
  python3 export_engine.py --weights models/best.pt --imgsz ${IMG_SIZE}
fi

# Inferencia
python3 run_infer.py \
  --source "${INFER_SOURCE}" \
  --imgsz ${IMG_SIZE} \
  --conf ${CONF_THRES} \
  --half ${HALF} \
  --use_trt ${USE_TRT}
