# Mobile runtime integration

ScamGuard's native scorer lives behind the versioned C ABI in
`native/include/scamguard_gguf.h`. The desktop protocol runner, iOS wrapper, and Android JNI
wrapper all call this same in-process implementation; they do not reimplement the scoring math.

This source package is runtime plumbing, not mobile performance evidence. Release claims remain
blocked until the frozen trained GGUF and calibration pass the physical-device protocol in
`docs/MOBILE_BENCHMARK_PROTOCOL.md` on both iOS and Android.

## iOS

Build the device and Apple-silicon simulator slices into one local XCFramework:

```bash
scripts/build_ios_xcframework.sh ../llama.cpp build/ScamGuardGGUF.xcframework
```

Add that XCFramework and `mobile/ios/ScamGuardRuntime.swift` to the app target. The framework's
module map exposes `import ScamGuardGGUF`. Bundle or install the selected GGUF in an app-private
location, pass its file URL and the frozen prompt prefix to `ScamGuardRuntime`, and keep one runtime
alive across requests so the model and prefix cache stay resident.

The Swift wrapper reports both native tokenization-to-score time and Swift-call end-to-end time.
Use the latter for the physical-device raw trace.

After committing the exact source revision, create the hash-bound runtime ZIP with:

```bash
make mobile-ios-package
```

## Android

Build the arm64 JNI library with the pinned Android NDK r27d (`27.3.13750724`):

```bash
scripts/build_android_runtime.sh ../llama.cpp "$ANDROID_NDK_HOME" build/android-arm64
```

Package `libscamguard-jni.so` under `jniLibs/arm64-v8a/` and add
`mobile/android/src/main/kotlin/com/scamguard/runtime/ScamGuardNative.kt` to the Android library or
app module. Copy the selected GGUF to app-private storage before constructing `ScamGuardNative`.
The wrapper defaults to CPU execution (`gpuLayers = 0`) until a device-specific accelerator path is
separately validated.

The Kotlin wrapper records `SystemClock.elapsedRealtimeNanos()` around the complete JNI call. Use
that value for the physical-device raw trace; the native microseconds field is diagnostic only.

After committing the exact source revision, create the hash-bound runtime ZIP with:

```bash
make mobile-android-package ANDROID_RUNTIME_BUILD=build/android-arm64-r27d
```

Both package builders refuse a dirty ScamGuard source tree, a different llama.cpp revision, or an
existing output path. ZIP member timestamps, ordering, permissions, and compression are fixed, so
the same inputs produce the same package hash.

## Release boundary

Do not publish either runtime package as validated merely because it compiles. The final package
ZIPs, GGUF, calibration, quantized prediction ledger, and raw physical-device reports must be
hashed and accepted together by `make mobile-benchmark-check` and the Hugging Face release gate.
