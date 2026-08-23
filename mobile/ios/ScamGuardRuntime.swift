import Foundation
import ScamGuardGGUF

public struct ScamGuardRuntimeInfo: Sendable {
    public let protocolVersion: UInt32
    public let modelBytes: UInt64
    public let contextSize: Int32
    public let prefixTokens: Int32
}

public struct ScamGuardRawScore: Sendable {
    public let safe: Double
    public let uncertain: Double
    public let scam: Double
    public let nativeElapsedMicroseconds: Int64
    public let endToEndElapsedNanoseconds: UInt64
    public let maximumSequenceTokens: Int32
    public let prefixReused: Bool
    public let prefixTokens: Int32
}

public enum ScamGuardVerdict: String, Sendable {
    case safe = "SAFE"
    case uncertain = "UNCERTAIN"
    case scam = "SCAM"
}

public struct ScamGuardCalibration: Sendable {
    public let promptSuffix: String
    public let temperature: Double
    public let scamThreshold: Double
    public let safeThreshold: Double

    public init(
        promptSuffix: String,
        temperature: Double,
        scamThreshold: Double,
        safeThreshold: Double
    ) throws {
        guard promptSuffix.hasPrefix("</message>"), temperature.isFinite, temperature > 0,
              (0 ... 1).contains(scamThreshold), (0 ... 1).contains(safeThreshold) else {
            throw ScamGuardRuntimeError.native("invalid ScamGuard calibration")
        }
        self.promptSuffix = promptSuffix
        self.temperature = temperature
        self.scamThreshold = scamThreshold
        self.safeThreshold = safeThreshold
    }
}

public struct ScamGuardDecision: Sendable {
    public let verdict: ScamGuardVerdict
    public let safeProbability: Double
    public let uncertainProbability: Double
    public let scamProbability: Double
    public let completeElapsedNanoseconds: UInt64
    public let rawScore: ScamGuardRawScore
}

public enum ScamGuardRuntimeError: Error, LocalizedError {
    case native(String)
    case closed

    public var errorDescription: String? {
        switch self {
        case .native(let message): message
        case .closed: "ScamGuard runtime is closed"
        }
    }
}

public final class ScamGuardRuntime: @unchecked Sendable {
    private let lock = NSLock()
    private let promptPrefix: String
    private var runtime: OpaquePointer?

    public init(
        modelURL: URL,
        prefix: String,
        contextSize: Int32 = 640,
        batchSize: Int32 = 640,
        microBatchSize: Int32 = 128,
        threads: Int32 = 4,
        gpuLayers: Int32 = 99
    ) throws {
        var config = sg_gguf_config()
        sg_gguf_config_init(&config)
        config.context_size = contextSize
        config.batch_size = batchSize
        config.micro_batch_size = microBatchSize
        config.threads = threads
        config.gpu_layers = gpuLayers

        var created: OpaquePointer?
        var error = [CChar](repeating: 0, count: 512)
        let prefixBytes = Array(prefix.utf8)
        let status = modelURL.withUnsafeFileSystemRepresentation { modelPath in
            prefixBytes.withUnsafeBytes { rawPrefix in
                sg_gguf_runtime_create(
                    modelPath,
                    rawPrefix.baseAddress?.assumingMemoryBound(to: CChar.self),
                    prefixBytes.count,
                    &config,
                    &created,
                    &error,
                    error.count
                )
            }
        }
        guard status == SG_GGUF_OK, let created else {
            throw ScamGuardRuntimeError.native(String(cString: error))
        }
        promptPrefix = prefix
        runtime = created
    }

    deinit {
        if let runtime { sg_gguf_runtime_destroy(runtime) }
    }

    public func info() throws -> ScamGuardRuntimeInfo {
        lock.lock()
        defer { lock.unlock() }
        guard let runtime else { throw ScamGuardRuntimeError.closed }
        var info = sg_gguf_runtime_info()
        sg_gguf_runtime_info_init(&info)
        var error = [CChar](repeating: 0, count: 512)
        guard sg_gguf_runtime_get_info(runtime, &info, &error, error.count) == SG_GGUF_OK else {
            throw ScamGuardRuntimeError.native(String(cString: error))
        }
        return ScamGuardRuntimeInfo(
            protocolVersion: info.protocol_version,
            modelBytes: info.model_bytes,
            contextSize: info.context_size,
            prefixTokens: info.prefix_tokens
        )
    }

    public func score(_ question: String) throws -> ScamGuardRawScore {
        guard !question.isEmpty else {
            throw ScamGuardRuntimeError.native("question must not be empty")
        }
        lock.lock()
        defer { lock.unlock() }
        guard let runtime else { throw ScamGuardRuntimeError.closed }
        let questionBytes = Array(question.utf8)
        var result = sg_gguf_score_result()
        sg_gguf_score_result_init(&result)
        var error = [CChar](repeating: 0, count: 512)
        let started = DispatchTime.now().uptimeNanoseconds
        let status = questionBytes.withUnsafeBytes { rawQuestion in
            sg_gguf_runtime_score(
                runtime,
                rawQuestion.baseAddress?.assumingMemoryBound(to: CChar.self),
                questionBytes.count,
                &result,
                &error,
                error.count
            )
        }
        let elapsed = DispatchTime.now().uptimeNanoseconds - started
        guard status == SG_GGUF_OK else {
            throw ScamGuardRuntimeError.native(String(cString: error))
        }
        return ScamGuardRawScore(
            safe: result.safe_score,
            uncertain: result.uncertain_score,
            scam: result.scam_score,
            nativeElapsedMicroseconds: result.elapsed_microseconds,
            endToEndElapsedNanoseconds: elapsed,
            maximumSequenceTokens: result.maximum_sequence_tokens,
            prefixReused: result.prefix_reused == 1,
            prefixTokens: result.prefix_tokens
        )
    }

    public func classify(
        message: String,
        calibration: ScamGuardCalibration
    ) throws -> ScamGuardDecision {
        guard !message.isEmpty else {
            throw ScamGuardRuntimeError.native("message must not be empty")
        }
        let started = DispatchTime.now().uptimeNanoseconds
        let question = promptPrefix + "<message>" + message + calibration.promptSuffix
        let raw = try score(question)
        let scaled = [raw.safe, raw.uncertain, raw.scam].map { $0 / calibration.temperature }
        guard scaled.allSatisfy(\.isFinite), let maximum = scaled.max() else {
            throw ScamGuardRuntimeError.native("native scores must be finite")
        }
        let exponentials = scaled.map { exp($0 - maximum) }
        let denominator = exponentials.reduce(0, +)
        guard denominator.isFinite, denominator > 0 else {
            throw ScamGuardRuntimeError.native("calibrated probabilities are invalid")
        }
        let probabilities = exponentials.map { $0 / denominator }
        let verdict: ScamGuardVerdict
        if probabilities[2] >= calibration.scamThreshold {
            verdict = .scam
        } else if probabilities[0] >= calibration.safeThreshold {
            verdict = .safe
        } else {
            verdict = .uncertain
        }
        return ScamGuardDecision(
            verdict: verdict,
            safeProbability: probabilities[0],
            uncertainProbability: probabilities[1],
            scamProbability: probabilities[2],
            completeElapsedNanoseconds: DispatchTime.now().uptimeNanoseconds - started,
            rawScore: raw
        )
    }

    public func close() {
        lock.lock()
        defer { lock.unlock() }
        if let runtime {
            sg_gguf_runtime_destroy(runtime)
            self.runtime = nil
        }
    }
}
