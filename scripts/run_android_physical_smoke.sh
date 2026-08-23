#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 DEVICE_SERIAL SMOKE.apk RUNTIME_PACK OUTPUT.json" >&2
  exit 2
fi

serial=$1
apk=$2
runtime_pack=$3
output=$4
package=com.humancto.scamguard.smoke
component="$package/com.scamguard.smoke.ScamGuardSmokeActivity"
request_source=mobile/android/smoke/control-request.json
model_source="$runtime_pack/Qwen3.5-0.8B-Q4_0.gguf"
manifest_source="$runtime_pack/scamguard_gguf_pack.json"
calibration_source="$runtime_pack/scamguard_calibration.json"

if [[ ! "$serial" =~ ^[A-Za-z0-9._:-]+$ ]]; then
  echo "Android device serial contains unsupported characters" >&2
  exit 1
fi
for required in \
  "$apk" \
  "$request_source" \
  "$model_source" \
  "$manifest_source" \
  "$calibration_source"; do
  if [[ ! -f "$required" ]]; then
    echo "missing Android physical smoke input: $required" >&2
    exit 1
  fi
done
if [[ -e "$output" ]]; then
  echo "refusing to overwrite Android physical smoke result: $output" >&2
  exit 1
fi
if [[ $(adb -s "$serial" get-state) != device ]]; then
  echo "Android device is not online: $serial" >&2
  exit 1
fi
abi=$(adb -s "$serial" shell getprop ro.product.cpu.abi | tr -d '\r')
if [[ "$abi" != arm64-v8a ]]; then
  echo "Android physical smoke requires arm64-v8a, found: $abi" >&2
  exit 1
fi
qemu=$(adb -s "$serial" shell getprop ro.kernel.qemu | tr -d '\r')
if [[ "$qemu" == 1 ]]; then
  echo "Android physical smoke refuses emulator devices" >&2
  exit 1
fi

adb -s "$serial" install -r "$apk" >/dev/null
run_id="$$"
request_name="scamguard-smoke-request-$run_id.json"
result_name="scamguard-smoke-result-$run_id.json"
remote_stage="/data/local/tmp/scamguard-smoke-$run_id"
app_root=$(adb -s "$serial" shell run-as "$package" pwd | tr -d '\r')
if [[ ! "$app_root" =~ ^/data/(data|user/[0-9]+)/com\.humancto\.scamguard\.smoke$ ]]; then
  echo "Android app-private root is unexpected: $app_root" >&2
  exit 1
fi
app_files="$app_root/files"
adb -s "$serial" shell mkdir -p "$remote_stage"
adb -s "$serial" push "$model_source" "$remote_stage/Qwen3.5-0.8B-Q4_0.gguf" >/dev/null
adb -s "$serial" push "$manifest_source" "$remote_stage/scamguard_gguf_pack.json" >/dev/null
adb -s "$serial" push "$calibration_source" "$remote_stage/scamguard_calibration.json" >/dev/null
adb -s "$serial" push "$request_source" "$remote_stage/$request_name" >/dev/null
adb -s "$serial" shell run-as "$package" mkdir -p "$app_files"
for name in \
  Qwen3.5-0.8B-Q4_0.gguf \
  scamguard_gguf_pack.json \
  scamguard_calibration.json \
  "$request_name"; do
  adb -s "$serial" shell run-as "$package" cp "$remote_stage/$name" "$app_files/$name"
  adb -s "$serial" shell rm -f "$remote_stage/$name"
done
adb -s "$serial" shell rmdir "$remote_stage"
adb -s "$serial" shell am start -W \
  -n "$component" \
  --es request "$request_name" \
  --es result "$result_name" >/dev/null

for _attempt in $(seq 1 120); do
  if adb -s "$serial" shell run-as "$package" test -f "$app_files/$result_name"; then
    mkdir -p "$(dirname "$output")"
    adb -s "$serial" exec-out run-as "$package" cat "$app_files/$result_name" >"$output"
    adb -s "$serial" shell am force-stop "$package"
    printf '%s\n' "$output"
    exit 0
  fi
  sleep 1
done

adb -s "$serial" shell am force-stop "$package"
echo "physical Android smoke timed out before producing $result_name" >&2
echo "inspect adb logcat for ScamGuardSmokeActivity and native runtime failures" >&2
exit 1
