# Run models on an Android phone

This walkthrough uses **Granite 3.1 1B-A400M Instruct Q4_K_M** as the small
example, then shows the separately documented 35B configuration.

## 1. Authorized ADB

Enable Developer options and USB debugging, authorize the host, then verify:

```sh
adb devices -l
# device ... model:vivo_1920
```

Wireless debugging may use a random port. The tested device was reached with:

```sh
adb connect 192.168.0.4:5555
```

Do not expose the API beyond loopback. The server binds `127.0.0.1:8080`;
forward it to the host over ADB:

```sh
adb forward tcp:8080 tcp:8080
```

## 2. Storage and memory preflight

```sh
adb shell 'df -h /data/local/tmp; cat /proc/meminfo | head -3'
```

The 35B artifact is 20,419,565,568 bytes and needs more free space than its
download plus temporary copies. Granite 1B is 821,847,360 bytes. Keep weights
outside the repository and do not commit them.

## 3. Download the exact Granite artifact

From `evidence/small-moe/20261021/artifacts.json`, the measured Granite 1B
identity is:

- repository: `bartowski/granite-3.1-1b-a400m-instruct-GGUF`
- revision: `940d2e1f9f65330615c7c8e980e6c5ac73d3360c`
- file: `granite-3.1-1b-a400m-instruct-Q4_K_M.gguf`
- bytes: 821,847,360
- SHA-256: `3a2ec1c2a78cb29d901e29bbf5162dcd03381e13803d2cbdcff838d4d08142eb`

```sh
mkdir -p "$AME_TOOLS/models"
.venv/bin/python -m pip install 'huggingface_hub==0.34.4'
huggingface-cli download \
  bartowski/granite-3.1-1b-a400m-instruct-GGUF \
  --revision 940d2e1f9f65330615c7c8e980e6c5ac73d3360c \
  --include granite-3.1-1b-a400m-instruct-Q4_K_M.gguf \
  --local-dir "$AME_TOOLS/models/granite-1b"

sha256sum "$AME_TOOLS/models/granite-1b/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf"
```

The hash must match exactly. A different hash is a different artifact, even if
the filename matches.

## 4. Generate the expert manifest

```sh
MODEL="$AME_TOOLS/models/granite-1b/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf"
SIZE=$(stat -c%s "$MODEL")
HASH=$(sha256sum "$MODEL" | cut -d' ' -f1)

.venv/bin/python tools/gguf_index.py \
  --input "$MODEL" --file-size "$SIZE" --artifact-sha256 "$HASH" \
  --manifest-out "$AME_TOOLS/models/granite-1b/granite.manifest" \
  --output "$AME_TOOLS/models/granite-1b/granite.index.json"
```

The top-k defaults to the model metadata. Do not force a different top-k.

## 5. Deploy

```sh
adb shell 'mkdir -p /data/local/tmp/ame-engine/models'
adb push "$MODEL" /data/local/tmp/ame-engine/models/
adb push "$AME_TOOLS/models/granite-1b/granite.manifest" /data/local/tmp/ame-api/
adb push "$AME_TOOLS/build/harness/ame-server-android" /data/local/tmp/ame-api/ame-server
adb push deploy/phone/start-dense.sh deploy/phone/start-moe.sh deploy/phone/stop.sh /data/local/tmp/ame-api/
adb shell 'chmod 755 /data/local/tmp/ame-api/ame-server /data/local/tmp/ame-api/*.sh'
```

## 6. Resident mode

Resident mode uses the backend's own model mapping. It is the fast baseline for
models that fit, not the streaming implementation.

```sh
adb shell 'cd /data/local/tmp/ame-api && \
  AME_MODEL=/data/local/tmp/ame-engine/models/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf \
  AME_MODEL_ID=granite-1b AME_CONTEXT=512 AME_THREADS=4 \
  nohup ./start-dense.sh >granite-resident.log 2>&1 </dev/null &'
adb forward tcp:8080 tcp:8080
curl -s http://127.0.0.1:8080/health
```

## 7. Streamed mode

Streamed mode needs the manifest, the artifact hash, an arena, and a slot count:

```sh
adb shell 'cd /data/local/tmp/ame-api && \
  AME_MODEL=/data/local/tmp/ame-engine/models/granite-3.1-1b-a400m-instruct-Q4_K_M.gguf \
  AME_MODEL_ID=granite-1b AME_SLOTS=64 AME_ARENA_MIB=256 \
  AME_CONTEXT=512 AME_THREADS=4 \
  nohup ./start-moe.sh >granite-streamed.log 2>&1 </dev/null &'
```

The measured small-model campaign used 64 slots and a 256 MiB arena. Liquid
LFM2.5 used **32 slots** under the documented pre-run amendment, because 64 slots
would require 309.75 MiB.

## 8. Readiness and restart

```sh
for i in $(seq 1 120); do
  curl -s http://127.0.0.1:8080/health
  echo
  sleep 2
done
```

Readiness is JSON `"status":"ready"`. A loading listener may answer health while
the model is still being hashed/loaded; generation endpoints are not ready until
the full server starts.

Stop and restart:

```sh
adb shell 'cd /data/local/tmp/ame-api && ./stop.sh'
adb shell 'cd /data/local/tmp/ame-api && nohup ./start-moe.sh >serve.log 2>&1 </dev/null &'
```

## 9. 35B configuration (experimental)

```sh
adb shell 'cd /data/local/tmp/ame-api && nohup ./ame-server \
  --model /data/local/tmp/ame-engine/models/Qwen3.6-35B-A3B-Q4_K_M.gguf \
  --model-id Qwen3.6-35B-A3B --stream \
  --manifest /data/local/tmp/ame-api/qwen36-35b.manifest \
  --artifact-sha 671e47e0ec53c665d048b98c3ecbfd5236b5ca9c3e02ed19fc8f81f7b85140c7 \
  --slots 64 --arena-mib 256 --context 512 --threads 4 --port 8080 \
  >qwen36.log 2>&1 </dev/null &'
```

The 35B artifact is 20,419,565,568 bytes. Startup hashes the entire file; this
took 173.759–339.631 seconds depending on run/thermal state. The 35B phone API
is experimental and has no real-model reference-parity qualification.

## Ports and paths

- Server: `127.0.0.1:8080`
- Host forward: `adb forward tcp:8080 tcp:8080`
- Device package: `/data/local/tmp/ame-api`
- Device models: `/data/local/tmp/ame-engine/models`
