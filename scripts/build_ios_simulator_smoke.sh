#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
  echo "usage: $0 SCAMGUARD_XCFRAMEWORK OUTPUT.app" >&2
  exit 2
fi

xcframework=$1
output=$2
slice="$xcframework/ios-arm64-simulator"
library="$slice/libScamGuardGGUF-ios-simulator.a"
headers="$slice/Headers"
if [[ ! -f "$library" || ! -d "$headers" ]]; then
  echo "ScamGuard simulator XCFramework slice is incomplete: $slice" >&2
  exit 1
fi
if [[ -e "$output" ]]; then
  echo "refusing to overwrite existing smoke app: $output" >&2
  exit 1
fi

mkdir -p "$output"
cp mobile/ios/smoke/Info.plist "$output/Info.plist"
xcrun --sdk iphonesimulator swiftc \
  -target arm64-apple-ios16.4-simulator \
  -O \
  -I "$headers" \
  mobile/ios/ScamGuardRuntime.swift \
  mobile/ios/smoke/ScamGuardSmokeMain.swift \
  "$library" \
  -Xlinker -lc++ \
  -framework Accelerate \
  -framework Metal \
  -framework Foundation \
  -o "$output/ScamGuardSmoke"
codesign --force --sign - "$output"
