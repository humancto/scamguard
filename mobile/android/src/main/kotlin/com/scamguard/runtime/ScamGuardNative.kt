package com.scamguard.runtime

import android.os.SystemClock
import java.io.Closeable

data class ScamGuardRuntimeInfo(
    val protocolVersion: Long,
    val modelBytes: Long,
    val contextSize: Int,
    val prefixTokens: Int,
)

data class ScamGuardRawScore(
    val safe: Double,
    val uncertain: Double,
    val scam: Double,
    val nativeElapsedMicros: Long,
    val endToEndElapsedNanos: Long,
    val maximumSequenceTokens: Int,
    val prefixReused: Boolean,
    val prefixTokens: Int,
)

class ScamGuardNative(
    modelPath: String,
    prefix: String,
    contextSize: Int = 640,
    batchSize: Int = 640,
    microBatchSize: Int = 128,
    threads: Int = 4,
    gpuLayers: Int = 0,
) : Closeable {
    private var handle: Long = nativeCreate(
        modelPath.toByteArray(Charsets.UTF_8),
        prefix.toByteArray(Charsets.UTF_8),
        contextSize,
        batchSize,
        microBatchSize,
        threads,
        gpuLayers,
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
            prefixTokens = values[6].toInt(),
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
        gpuLayers: Int,
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
