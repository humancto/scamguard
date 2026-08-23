#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 LIBSCAMGUARD_JNI.so ANDROID_SDK DEBUG_KEYSTORE OUTPUT.apk" >&2
  exit 2
fi

jni_library=$1
android_sdk=$2
debug_keystore=$3
output=$4
build_tools="$android_sdk/build-tools/30.0.1"
signer="$android_sdk/build-tools/29.0.2/apksigner"
android_jar="$android_sdk/platforms/android-30/android.jar"
manifest=mobile/android/smoke/AndroidManifest.xml
runtime_source=mobile/android/src/main/kotlin/com/scamguard/runtime/ScamGuardNative.kt
smoke_source=mobile/android/smoke/ScamGuardSmokeActivity.kt

for required in \
  "$jni_library" \
  "$android_jar" \
  "$debug_keystore" \
  "$build_tools/aapt" \
  "$build_tools/aapt2" \
  "$build_tools/d8" \
  "$build_tools/zipalign" \
  "$signer" \
  "$manifest" \
  "$runtime_source" \
  "$smoke_source"; do
  if [[ ! -f "$required" ]]; then
    echo "missing Android smoke build input: $required" >&2
    exit 1
  fi
done
if [[ -e "$output" ]]; then
  echo "refusing to overwrite Android smoke APK: $output" >&2
  exit 1
fi

work=$(mktemp -d)
trap 'rm -rf "$work"' EXIT
unsigned="$work/scamguard-smoke-unsigned.apk"
aligned="$work/scamguard-smoke-aligned.apk"
classes="$work/scamguard-smoke-classes.jar"
stage="$work/stage"
mkdir -p "$stage/lib/arm64-v8a" "$work/dex" "$(dirname "$output")"

"$build_tools/aapt2" link \
  -I "$android_jar" \
  --manifest "$manifest" \
  --min-sdk-version 28 \
  --target-sdk-version 30 \
  -o "$unsigned"

kotlinc \
  "$runtime_source" \
  "$smoke_source" \
  -classpath "$android_jar" \
  -jvm-target 1.8 \
  -include-runtime \
  -d "$classes"
"$build_tools/d8" \
  --lib "$android_jar" \
  --min-api 28 \
  --output "$work/dex" \
  "$classes"

cp "$work/dex/classes.dex" "$stage/classes.dex"
cp "$jni_library" "$stage/lib/arm64-v8a/libscamguard-jni.so"
(
  cd "$stage"
  "$build_tools/aapt" add "$unsigned" classes.dex lib/arm64-v8a/libscamguard-jni.so
)
"$build_tools/zipalign" -f 4 "$unsigned" "$aligned"

cp "$aligned" "$output"
"$signer" sign \
  --ks "$debug_keystore" \
  --ks-key-alias androiddebugkey \
  --ks-pass pass:android \
  --key-pass pass:android \
  --min-sdk-version 28 \
  "$output"
"$signer" verify --verbose --print-certs "$output"
"$build_tools/aapt" dump badging "$output"
