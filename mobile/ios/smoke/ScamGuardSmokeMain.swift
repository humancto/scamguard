import Foundation

private struct SmokeRequest: Decodable {
    let model: String
    let packManifest: String
    let calibration: String
    let message: String
}

private struct PackManifest: Decodable {
    struct Prompt: Decodable {
        let prefix: String
        let suffix: String
    }

    struct Runtime: Decodable {
        let ctxSize: Int32
        let batchSize: Int32
        let ubatchSize: Int32
        let threads: Int32
        let gpuLayers: Int32

        enum CodingKeys: String, CodingKey {
            case ctxSize = "ctx_size"
            case batchSize = "batch_size"
            case ubatchSize = "ubatch_size"
            case threads
            case gpuLayers = "n_gpu_layers"
        }
    }

    let prompt: Prompt
    let runtime: Runtime
    let publicationAuthorized: Bool

    enum CodingKeys: String, CodingKey {
        case prompt
        case runtime
        case publicationAuthorized = "publication_authorized"
    }
}

private struct Calibration: Decodable {
    let temperature: Double
    let scamThreshold: Double
    let safeThreshold: Double

    enum CodingKeys: String, CodingKey {
        case temperature
        case scamThreshold = "scam_threshold"
        case safeThreshold = "safe_threshold"
    }
}

private struct SmokeResult: Encodable {
    let artifactSchemaVersion = 1
    let diagnosticOnly = true
    let physicalDevice = false
    let simulator = true
    let verdict: String
    let safeProbability: Double
    let uncertainProbability: Double
    let scamProbability: Double
    let rawSafeScore: Double
    let rawUncertainScore: Double
    let rawScamScore: Double
    let startupMilliseconds: Double
    let completeElapsedMilliseconds: Double
    let nativeElapsedMilliseconds: Double
    let prefixReused: Bool
    let prefixTokens: Int32
    let modelTensorBytes: UInt64
    let protocolVersion: UInt32

    enum CodingKeys: String, CodingKey {
        case artifactSchemaVersion = "artifact_schema_version"
        case diagnosticOnly = "diagnostic_only"
        case physicalDevice = "physical_device"
        case simulator
        case verdict
        case safeProbability = "safe_probability"
        case uncertainProbability = "uncertain_probability"
        case scamProbability = "scam_probability"
        case rawSafeScore = "raw_safe_score"
        case rawUncertainScore = "raw_uncertain_score"
        case rawScamScore = "raw_scam_score"
        case startupMilliseconds = "startup_ms"
        case completeElapsedMilliseconds = "complete_elapsed_ms"
        case nativeElapsedMilliseconds = "native_elapsed_ms"
        case prefixReused = "prefix_reused"
        case prefixTokens = "prefix_tokens"
        case modelTensorBytes = "model_tensor_bytes"
        case protocolVersion = "protocol_version"
    }
}

@main
private struct ScamGuardSmokeMain {
    static func main() throws {
        guard CommandLine.arguments.count == 3 else {
            throw ScamGuardRuntimeError.native("usage: ScamGuardSmoke REQUEST.json RESULT.json")
        }
        let documents = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
        let requestURL = documents.appendingPathComponent(CommandLine.arguments[1])
        let resultURL = documents.appendingPathComponent(CommandLine.arguments[2])
        let decoder = JSONDecoder()
        let request = try decoder.decode(SmokeRequest.self, from: Data(contentsOf: requestURL))
        let pack = try decoder.decode(
            PackManifest.self,
            from: Data(contentsOf: documents.appendingPathComponent(request.packManifest))
        )
        guard pack.publicationAuthorized == false else {
            throw ScamGuardRuntimeError.native("smoke control pack cannot authorize publication")
        }
        let calibrationRecord = try decoder.decode(
            Calibration.self,
            from: Data(contentsOf: documents.appendingPathComponent(request.calibration))
        )
        let calibration = try ScamGuardCalibration(
            promptSuffix: pack.prompt.suffix,
            temperature: calibrationRecord.temperature,
            scamThreshold: calibrationRecord.scamThreshold,
            safeThreshold: calibrationRecord.safeThreshold
        )

        let startupStarted = DispatchTime.now().uptimeNanoseconds
        let runtime = try ScamGuardRuntime(
            modelURL: documents.appendingPathComponent(request.model),
            prefix: pack.prompt.prefix,
            contextSize: pack.runtime.ctxSize,
            batchSize: pack.runtime.batchSize,
            microBatchSize: pack.runtime.ubatchSize,
            threads: pack.runtime.threads,
            // MTLSimDriver aborts while creating llama.cpp's mapped prefix-cache
            // buffer. Keep this diagnostic CPU-only; physical-device evidence is
            // collected separately with the pack's configured Metal offload.
            gpuLayers: 0
        )
        let startupElapsed = DispatchTime.now().uptimeNanoseconds - startupStarted
        defer { runtime.close() }
        let info = try runtime.info()
        let decision = try runtime.classify(message: request.message, calibration: calibration)
        let output = SmokeResult(
            verdict: decision.verdict.rawValue,
            safeProbability: decision.safeProbability,
            uncertainProbability: decision.uncertainProbability,
            scamProbability: decision.scamProbability,
            rawSafeScore: decision.rawScore.safe,
            rawUncertainScore: decision.rawScore.uncertain,
            rawScamScore: decision.rawScore.scam,
            startupMilliseconds: Double(startupElapsed) / 1_000_000,
            completeElapsedMilliseconds: Double(decision.completeElapsedNanoseconds) / 1_000_000,
            nativeElapsedMilliseconds: Double(decision.rawScore.nativeElapsedMicroseconds) / 1_000,
            prefixReused: decision.rawScore.prefixReused,
            prefixTokens: decision.rawScore.prefixTokens,
            modelTensorBytes: info.modelBytes,
            protocolVersion: info.protocolVersion
        )
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        try encoder.encode(output).write(to: resultURL, options: .atomic)
        print(String(data: try encoder.encode(output), encoding: .utf8) ?? "{}")
    }
}
