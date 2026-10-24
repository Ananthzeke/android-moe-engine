#!/system/bin/sh
set -eu
DIR=${1:-/data/local/tmp/ame-api}
MODEL=${AME_MODEL:-/data/local/tmp/ame-engine/models/Qwen3.6-35B-A3B-Q4_K_M.gguf}
MANIFEST=${AME_MANIFEST:-$DIR/qwen36-35b-v2.manifest}
SHA=${AME_ARTIFACT_SHA:-671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7}
# Layered expert cache: 32 slots per layer (2.2 GB arena), least recently used, and the kernel is
# told to drop its own copy of each extent. Measured on the vivo 1920 against the shipped cache
# (evidence/core-opt/unit4): same token IDs, 28-40% less request time, 2.2 GB more resident memory.
# Four reader threads fetch a layer's missing experts in parallel (evidence/core-opt/unit5: a further 22-39%
# less request time, same token IDs; pinning them to the little cores made no difference). It needs about
# 3.6 GB of MemAvailable after start. AME_CACHE=legacy restores the 64-slot cache.
CACHE=${AME_CACHE:-layered}
if [ "$CACHE" = layered ]; then
  ARENA=${AME_ARENA_MIB:-2200}
  CACHE_FLAGS="--cache layered --layer-slots ${AME_LAYER_SLOTS:-32} --fadvise-random --fadvise-dontneed --readers ${AME_READERS:-4}"
else
  ARENA=${AME_ARENA_MIB:-256}
  CACHE_FLAGS="--cache legacy"
fi
# shellcheck disable=SC2086
exec "$DIR/ame-server" --model "$MODEL" --model-id "${AME_MODEL_ID:-Qwen3.6-35B-A3B}" \
  --stream --manifest "$MANIFEST" --artifact-sha "$SHA" \
  --slots "${AME_SLOTS:-64}" --arena-mib "$ARENA" $CACHE_FLAGS \
  --context "${AME_CONTEXT:-512}" --threads "${AME_THREADS:-4}" \
  --port "${AME_PORT:-8081}" ${AME_REQUEST_LOG:+--request-log "$AME_REQUEST_LOG"}
