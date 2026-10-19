#!/system/bin/sh
set -eu
DIR=${1:-/data/local/tmp/ame-api}
MODEL=${AME_MODEL:-/data/local/tmp/ame-engine/models/Qwen3.5-2B-Q4_K_M.gguf}
PORT=${AME_PORT:-8080}
exec "$DIR/ame-server" --model "$MODEL" --model-id "${AME_MODEL_ID:-Qwen3.5-2B}" \
  --context "${AME_CONTEXT:-1024}" --threads "${AME_THREADS:-4}" --port "$PORT"
