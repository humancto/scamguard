package com.scamguard.smoke

import android.app.Activity
import android.os.Build
import android.os.Bundle
import android.os.SystemClock
import com.scamguard.runtime.ScamGuardCalibration
import com.scamguard.runtime.ScamGuardNative
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import org.json.JSONObject

class ScamGuardSmokeActivity : Activity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        Thread {
            val resultName = intent.getStringExtra("result") ?: "control-result.json"
            try {
                runSmoke(
                    intent.getStringExtra("request") ?: "control-request.json",
                    resultName
                )
            } catch (error: Throwable) {
                writeResult(
                    resultName,
                    JSONObject()
                        .put("artifact_schema_version", 1)
                        .put("diagnostic_only", true)
                        .put("error_class", error.javaClass.name)
                        .put("passed", false)
                )
            } finally {
                runOnUiThread { finishAndRemoveTask() }
            }
        }.start()
    }

    private fun runSmoke(requestName: String, resultName: String) {
        val root = filesDir
        val request = JSONObject(File(root, requestName).readText(Charsets.UTF_8))
        val pack = JSONObject(
            File(root, request.getString("packManifest")).readText(Charsets.UTF_8)
        )
        if (pack.getBoolean("publication_authorized")) {
            throw IllegalStateException("smoke control pack cannot authorize publication")
        }
        val prompt = pack.getJSONObject("prompt")
        val runtime = pack.getJSONObject("runtime")
        val calibrationRecord = JSONObject(
            File(root, request.getString("calibration")).readText(Charsets.UTF_8)
        )
        val calibration = ScamGuardCalibration(
            promptSuffix = prompt.getString("suffix"),
            temperature = calibrationRecord.getDouble("temperature"),
            scamThreshold = calibrationRecord.getDouble("scam_threshold"),
            safeThreshold = calibrationRecord.getDouble("safe_threshold")
        )
        val startupStarted = SystemClock.elapsedRealtimeNanos()
        val scanner = ScamGuardNative(
            modelPath = File(root, request.getString("model")).absolutePath,
            prefix = prompt.getString("prefix"),
            contextSize = runtime.getInt("ctx_size"),
            batchSize = runtime.getInt("batch_size"),
            microBatchSize = runtime.getInt("ubatch_size"),
            threads = runtime.getInt("threads"),
            gpuLayers = 0
        )
        val startupNanos = SystemClock.elapsedRealtimeNanos() - startupStarted
        scanner.use {
            val info = it.info()
            val decision = it.classify(request.getString("message"), calibration)
            val emulator = isEmulator()
            writeResult(
                resultName,
                JSONObject()
                    .put("artifact_schema_version", 1)
                    .put("diagnostic_only", true)
                    .put("physical_device", !emulator)
                    .put("simulator", emulator)
                    .put("platform", "Android")
                    .put("manufacturer", Build.MANUFACTURER)
                    .put("model", Build.MODEL)
                    .put("android_api", Build.VERSION.SDK_INT)
                    .put("abi", Build.SUPPORTED_ABIS.firstOrNull() ?: "unknown")
                    .put("backend", "llama.cpp CPU")
                    .put("verdict", decision.verdict.name)
                    .put("safe_probability", decision.safeProbability)
                    .put("uncertain_probability", decision.uncertainProbability)
                    .put("scam_probability", decision.scamProbability)
                    .put("raw_safe_score", decision.rawScore.safe)
                    .put("raw_uncertain_score", decision.rawScore.uncertain)
                    .put("raw_scam_score", decision.rawScore.scam)
                    .put("startup_ms", startupNanos / 1_000_000.0)
                    .put("complete_elapsed_ms", decision.completeElapsedNanos / 1_000_000.0)
                    .put("native_elapsed_ms", decision.rawScore.nativeElapsedMicros / 1_000.0)
                    .put("prefix_reused", decision.rawScore.prefixReused)
                    .put("prefix_tokens", decision.rawScore.prefixTokens)
                    .put("model_tensor_bytes", info.modelBytes)
                    .put("protocol_version", info.protocolVersion)
                    .put("passed", true)
            )
        }
    }

    private fun writeResult(name: String, value: JSONObject) {
        val root = filesDir
        val output = File(root, name)
        if (output.exists()) {
            throw IllegalStateException("refusing to overwrite smoke result")
        }
        val temporary = File(root, "$name.tmp")
        FileOutputStream(temporary).use {
            it.write((value.toString(2) + "\n").toByteArray(Charsets.UTF_8))
            it.fd.sync()
        }
        if (!temporary.renameTo(output)) {
            throw IllegalStateException("failed to atomically publish smoke result")
        }
    }

    private fun isEmulator(): Boolean {
        val fingerprint = Build.FINGERPRINT.toLowerCase(Locale.US)
        val model = Build.MODEL.toLowerCase(Locale.US)
        return fingerprint.startsWith("generic") ||
            fingerprint.contains("emulator") ||
            fingerprint.contains("vbox") ||
            model.contains("emulator") ||
            model.contains("google_sdk")
    }
}
