#!/bin/sh
# Run one measured baseline on the connected phone, pulling its record into RUN_DIR.
# Environment: AME_DEVICE_DIR (default /data/local/tmp/ame-engine),
#              AME_MODEL_SHA (required unless the harness records null),
#              AME_PROMPT (default: capital of France).
set -eu
RUN_DIR=${1:?run directory}
RUN_ID=${2:?run id}
DEVICE_DIR=${AME_DEVICE_DIR:-/data/local/tmp/ame-engine}
MODEL_SHA=${AME_MODEL_SHA:-}
BIN_SHA=$(adb shell sha256sum "$DEVICE_DIR/llama_baseline" | cut -d' ' -f1)
adb shell /data/local/tmp/ame-probe probe --output "$DEVICE_DIR/probe-before-$RUN_ID.json"
adb pull "$DEVICE_DIR/probe-before-$RUN_ID.json" "$RUN_DIR/probe-before.json" >/dev/null
adb shell "$DEVICE_DIR/llama_baseline \
  --model $DEVICE_DIR/models/Qwen3.5-2B-Q4_K_M.gguf \
  --prompt-file $DEVICE_DIR/prompt.txt --chat \
  --max-new-tokens 16 --context 512 --threads 4 --batch-size 512 --seed 1234 \
  --artifact-sha $MODEL_SHA --binary-sha $BIN_SHA \
  --backend-commit ff5a23376725e5a889fa0bfb987b7541dc304da0 \
  --run-id $RUN_ID --tag physical_target \
  --output $DEVICE_DIR/$RUN_ID.json 2> $DEVICE_DIR/$RUN_ID.err"
adb pull "$DEVICE_DIR/$RUN_ID.json" "$RUN_DIR/measurement.json" >/dev/null
adb pull "$DEVICE_DIR/$RUN_ID.err" "$RUN_DIR/device.stderr" >/dev/null
adb shell /data/local/tmp/ame-probe probe --output "$DEVICE_DIR/probe-after-$RUN_ID.json"
adb pull "$DEVICE_DIR/probe-after-$RUN_ID.json" "$RUN_DIR/probe-after.json" >/dev/null