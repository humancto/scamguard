#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 SIMULATOR_UDID APP_BUNDLE RUNTIME_PACK OUTPUT.json" >&2
  exit 2
fi

simulator_udid=$1
app_bundle=$2
runtime_pack=$3
output=$4
bundle_id=com.humancto.scamguard.smoke
request_source=mobile/ios/smoke/control-request.json
model_source="$runtime_pack/Qwen3.5-0.8B-Q4_0.gguf"
manifest_source="$runtime_pack/scamguard_gguf_pack.json"
calibration_source="$runtime_pack/scamguard_calibration.json"

for required in \
  "$app_bundle/ScamGuardSmoke" \
  "$request_source" \
  "$model_source" \
  "$manifest_source" \
  "$calibration_source"; do
  if [[ ! -f "$required" ]]; then
    echo "missing iOS Simulator smoke input: $required" >&2
    exit 1
  fi
done
if [[ -e "$output" ]]; then
  echo "refusing to overwrite smoke result: $output" >&2
  exit 1
fi

xcrun simctl install "$simulator_udid" "$app_bundle"
data_container=$(xcrun simctl get_app_container "$simulator_udid" "$bundle_id" data)
documents="$data_container/Documents"
run_id="$$"
request_name="scamguard-smoke-request-$run_id.json"
result_name="scamguard-smoke-result-$run_id.json"

cp "$model_source" "$manifest_source" "$calibration_source" "$documents/"
cp "$request_source" "$documents/$request_name"
xcrun simctl launch --terminate-running-process \
  "$simulator_udid" "$bundle_id" "$request_name" "$result_name" >/dev/null

for _attempt in $(seq 1 60); do
  if [[ -f "$documents/$result_name" ]]; then
    mkdir -p "$(dirname "$output")"
    cp "$documents/$result_name" "$output"
    printf '%s\n' "$output"
    exit 0
  fi
  sleep 1
done

echo "simulator smoke timed out before producing $result_name" >&2
echo "inspect the latest ScamGuardSmoke crash report and simulator unified log" >&2
exit 1
