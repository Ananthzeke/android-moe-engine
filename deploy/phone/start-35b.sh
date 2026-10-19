#!/system/bin/sh
set -eu
DIR=${1:-/data/local/tmp/ame-api}
MODEL=${AME_MODEL:-/data/local/tmp/ame-engine/models/Qwen3.6-35B-A3B-Q4_K_M.gguf}
MANIFEST=${AME_MANIFEST:-$DIR/qwen36-35b-v2.manifest}
SHA=${AME_ARTIFACT_SHA:-671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7}
exec "$DIR/ame-server" --model "$MODEL" --model-id "${AME_MODEL_ID:-Qwen3.6-35B-A3B}" \
  --stream --manifest "$MANIFEST" --artifact-sha "$SHA" \
  --slots "${AME_SLOTS:-64}" --arena-mib "${AME_ARENA_MIB:-256}" \
  --context "${AME_CONTEXT:-512}" --threads "${AME_THREADS:-4}" \
  --port "${AME_PORT:-8081}"
