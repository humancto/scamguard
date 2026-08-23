package com.scamguard.runtime

import android.os.SystemClock
import java.io.Closeable
import kotlin.math.exp

data class ScamGuardRuntimeInfo(
    val protocolVersion: Long,
    val modelBytes: Long,
    val contextSize: Int,
    val prefixTokens: Int
)

data class ScamGuardRawScore(
    val safe: Double,
    val uncertain: Double,
    val scam: Double,
    val nativeElapsedMicros: Long,
    val endToEndElapsedNanos: Long,
    val maximumSequenceTokens: Int,
    val prefixReused: Boolean,
    val prefixTokens: Int
)

enum class ScamGuardVerdict {
    SAFE,
    UNCERTAIN,
    SCAM
}

data class ScamGuardCalibration(
    val promptSuffix: String,
    val temperature: Double,
    val scamThreshold: Double,
    val safeThreshold: Double
) {
    init {
        require(promptSuffix.startsWith("</message>")) { "prompt suffix must close the message" }
        require(temperature.isFinite() && temperature > 0.0) { "temperature must be positive" }
        require(scamThreshold in 0.0..1.0) { "scam threshold must be in [0, 1]" }
        require(safeThreshold in 0.0..1.0) { "safe threshold must be in [0, 1]" }
    }
}

data class ScamGuardDecision(
    val verdict: ScamGuardVerdict,
    val safeProbability: Double,
    val uncertainProbability: Double,
    val scamProbability: Double,
    val completeElapsedNanos: Long,
    val rawScore: ScamGuardRawScore
)

class ScamGuardNative(
    modelPath: String,
    prefix: String,
    contextSize: Int = 640,
    batchSize: Int = 640,
    microBatchSize: Int = 128,
    threads: Int = 4,
    gpuLayers: Int = 0
) : Closeable {
    private val promptPrefix: String = prefix
    private var handle: Long = nativeCreate(
        modelPath.toByteArray(Charsets.UTF_8),
        prefix.toByteArray(Charsets.UTF_8),
        contextSize,
        batchSize,
        microBatchSize,
        threads,
        gpuLayers
    )

    @Synchronized
    fun info(): ScamGuardRuntimeInfo {
        val values = nativeInfo(requireHandle())
        return ScamGuardRuntimeInfo(values[0], values[1], values[2].toInt(), values[3].toInt())
    }

    @Synchronized
    fun score(question: String): ScamGuardRawScore {
        require(question.isNotEmpty()) { "question must not be empty" }
        val started = SystemClock.elapsedRealtimeNanos()
        val values = nativeScore(requireHandle(), question.toByteArray(Charsets.UTF_8))
        val elapsed = SystemClock.elapsedRealtimeNanos() - started
        return ScamGuardRawScore(
            safe = values[0],
            uncertain = values[1],
            scam = values[2],
            nativeElapsedMicros = values[3].toLong(),
            endToEndElapsedNanos = elapsed,
            maximumSequenceTokens = values[4].toInt(),
            prefixReused = values[5] == 1.0,
            prefixTokens = values[6].toInt()
        )
    }

    @Synchronized
    fun classify(message: String, calibration: ScamGuardCalibration): ScamGuardDecision {
        require(message.isNotEmpty()) { "message must not be empty" }
        val started = SystemClock.elapsedRealtimeNanos()
        val question = promptPrefix + "<message>" + message + calibration.promptSuffix
        val raw = score(question)
        val scaled = doubleArrayOf(raw.safe, raw.uncertain, raw.scam).map {
            it / calibration.temperature
        }
        require(scaled.all { it.isFinite() }) { "native scores must be finite" }
        val maximum = scaled.max()!!
        val exponentials = scaled.map { exp(it - maximum) }
        val denominator = exponentials.sum()
        require(denominator.isFinite() && denominator > 0.0) {
            "calibrated probabilities are invalid"
        }
        val probabilities = exponentials.map { it / denominator }
        val verdict = when {
            probabilities[2] >= calibration.scamThreshold -> ScamGuardVerdict.SCAM
            probabilities[0] >= calibration.safeThreshold -> ScamGuardVerdict.SAFE
            else -> ScamGuardVerdict.UNCERTAIN
        }
        return ScamGuardDecision(
            verdict = verdict,
            safeProbability = probabilities[0],
            uncertainProbability = probabilities[1],
            scamProbability = probabilities[2],
            completeElapsedNanos = SystemClock.elapsedRealtimeNanos() - started,
            rawScore = raw
        )
    }

    @Synchronized
    override fun close() {
        if (handle != 0L) {
            nativeDestroy(handle)
            handle = 0L
        }
    }

    private fun requireHandle(): Long {
        check(handle != 0L) { "ScamGuard runtime is closed" }
        return handle
    }

    private external fun nativeCreate(
        modelPathUtf8: ByteArray,
        prefixUtf8: ByteArray,
        contextSize: Int,
        batchSize: Int,
        microBatchSize: Int,
        threads: Int,
        gpuLayers: Int
    ): Long

    private external fun nativeInfo(handle: Long): LongArray
    private external fun nativeScore(handle: Long, questionUtf8: ByteArray): DoubleArray
    private external fun nativeDestroy(handle: Long)

    companion object {
        init {
            System.loadLibrary("scamguard-jni")
        }
    }
}
