#!/system/bin/sh
set -eu
DIR=${1:-/data/local/tmp/ame-api}
MODEL=${AME_MODEL:-/data/local/tmp/ame-engine/models/Qwen1.5-MoE-A2.7B_q4_k_m.gguf}
PORT=${AME_PORT:-8080}
MANIFEST=${AME_MANIFEST:-$DIR/qwen15.manifest}
SHA=${AME_ARTIFACT_SHA:?set AME_ARTIFACT_SHA}
exec "$DIR/ame-server" --model "$MODEL" --model-id "${AME_MODEL_ID:-Qwen1.5-MoE}" \
  --stream --manifest "$MANIFEST" --artifact-sha "$SHA" \
  --slots "${AME_SLOTS:-64}" --arena-mib "${AME_ARENA_MIB:-1024}" \
  --context "${AME_CONTEXT:-512}" --threads "${AME_THREADS:-4}" --port "$PORT"