#include "scamguard_gguf.h"

#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::array<std::string_view, 3> k_answers = {"SAFE\"", "UNCERTAIN\"", "SCAM\""};
constexpr int32_t k_sequence_count = 4;
constexpr int32_t k_context_headroom = 64;

std::mutex g_backend_mutex;
size_t g_backend_users = 0;
std::once_flag g_backend_plugins_once;

class backend_lease {
  public:
    backend_lease() {
        std::lock_guard<std::mutex> lock(g_backend_mutex);
        std::call_once(g_backend_plugins_once, [] { ggml_backend_load_all(); });
        if (g_backend_users++ == 0) {
            llama_backend_init();
        }
    }

    backend_lease(const backend_lease &) = delete;
    backend_lease & operator=(const backend_lease &) = delete;

    ~backend_lease() {
        std::lock_guard<std::mutex> lock(g_backend_mutex);
        if (--g_backend_users == 0) {
            llama_backend_free();
        }
    }
};

void write_error(char * destination, size_t capacity, std::string_view message) noexcept {
    if (destination == nullptr || capacity == 0) return;
    const size_t copied = std::min(capacity - 1, message.size());
    if (copied > 0) std::memcpy(destination, message.data(), copied);
    destination[copied] = '\0';
}

void clear_error(char * destination, size_t capacity) noexcept {
    if (destination != nullptr && capacity > 0) destination[0] = '\0';
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, std::string_view text) {
    if (text.size() > static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        throw std::runtime_error("input exceeds tokenizer length limit");
    }
    const int32_t length = static_cast<int32_t>(text.size());
    const int32_t required = -llama_tokenize(vocab, text.data(), length, nullptr, 0, true, false);
    if (required <= 0) throw std::runtime_error("tokenizer did not return a positive token count");
    std::vector<llama_token> tokens(static_cast<size_t>(required));
    const int32_t written = llama_tokenize(
        vocab, text.data(), length, tokens.data(), static_cast<int32_t>(tokens.size()), true, false);
    if (written != required) {
        throw std::runtime_error("tokenizer output size changed between calls");
    }
    return tokens;
}

void clear_batch(llama_batch & batch) { batch.n_tokens = 0; }

int32_t add_token(
    llama_batch & batch,
    llama_token token,
    llama_pos position,
    const std::vector<llama_seq_id> & sequences,
    bool output) {
    const int32_t index = batch.n_tokens;
    batch.token[index] = token;
    batch.pos[index] = position;
    batch.n_seq_id[index] = static_cast<int32_t>(sequences.size());
    for (size_t sequence = 0; sequence < sequences.size(); ++sequence) {
        batch.seq_id[index][sequence] = sequences[sequence];
    }
    batch.logits[index] = output ? 1 : 0;
    ++batch.n_tokens;
    return index;
}

double token_log_probability(const float * logits, int32_t vocabulary_size, llama_token target) {
    if (logits == nullptr || target < 0 || target >= vocabulary_size) {
        throw std::runtime_error("invalid logits row or target token");
    }
    float maximum = -std::numeric_limits<float>::infinity();
    for (int32_t token = 0; token < vocabulary_size; ++token) {
        maximum = std::max(maximum, logits[token]);
    }
    double denominator = 0.0;
    for (int32_t token = 0; token < vocabulary_size; ++token) {
        denominator += std::exp(static_cast<double>(logits[token] - maximum));
    }
    return static_cast<double>(logits[target] - maximum) - std::log(denominator);
}

struct internal_score_result {
    std::array<double, 3> scores;
    int64_t elapsed_microseconds;
    size_t maximum_sequence_tokens;
    bool prefix_reused;
    size_t prefix_tokens;
};

void initialize_prefix_cache(
    llama_context * context,
    llama_batch & batch,
    const std::vector<llama_token> & prefix_tokens,
    int32_t batch_size) {
    if (prefix_tokens.empty()) return;
    if (prefix_tokens.size() >= static_cast<size_t>(llama_n_ctx(context))) {
        throw std::runtime_error("cached prefix does not fit in the GGUF context");
    }
    llama_memory_clear(llama_get_memory(context), true);
    clear_batch(batch);
    for (size_t position = 0; position < prefix_tokens.size(); ++position) {
        add_token(batch, prefix_tokens[position], static_cast<llama_pos>(position), {3}, false);
    }
    for (int32_t offset = 0; offset < batch.n_tokens; offset += batch_size) {
        const int32_t token_count = std::min(batch_size, batch.n_tokens - offset);
        llama_batch view = {
            token_count, batch.token + offset, nullptr, batch.pos + offset,
            batch.n_seq_id + offset, batch.seq_id + offset, batch.logits + offset,
        };
        const int32_t status = llama_decode(context, view);
        if (status != 0) {
            throw std::runtime_error(
                "failed to initialize prefix cache with status " + std::to_string(status));
        }
    }
}

}  // namespace

struct sg_gguf_runtime {
    backend_lease backend;
    llama_model * model = nullptr;
    llama_context * context = nullptr;
    llama_batch batch{};
    bool batch_initialized = false;
    const llama_vocab * vocab = nullptr;
    int32_t vocabulary_size = 0;
    int32_t configured_context_size = 0;
    int32_t batch_size = 0;
    std::vector<llama_token> cached_prefix;
    std::mutex score_mutex;

    ~sg_gguf_runtime() {
        if (batch_initialized) llama_batch_free(batch);
        if (context != nullptr) llama_free(context);
        if (model != nullptr) llama_model_free(model);
    }
};

namespace {

internal_score_result score_request(sg_gguf_runtime & runtime, std::string_view question) {
    const auto started = std::chrono::steady_clock::now();
    std::array<std::vector<llama_token>, 3> sequences;
    size_t maximum_candidate_tokens = 0;
    for (size_t answer = 0; answer < k_answers.size(); ++answer) {
        std::string candidate(question);
        candidate.append(k_answers[answer]);
        sequences[answer] = tokenize(runtime.vocab, candidate);
        maximum_candidate_tokens = std::max(maximum_candidate_tokens, sequences[answer].size());
        if (sequences[answer].size() > static_cast<size_t>(runtime.configured_context_size)) {
            throw std::runtime_error("request exceeds the configured per-sequence context");
        }
    }

    size_t common_prefix = maximum_candidate_tokens;
    for (const auto & sequence : sequences) common_prefix = std::min(common_prefix, sequence.size());

    const bool prefix_reused = !runtime.cached_prefix.empty();
    if (prefix_reused) {
        if (runtime.cached_prefix.size() >= common_prefix) {
            throw std::runtime_error("cached prefix leaves no common request token to score");
        }
        for (const auto & sequence : sequences) {
            if (!std::equal(runtime.cached_prefix.begin(), runtime.cached_prefix.end(), sequence.begin())) {
                throw std::runtime_error("cached prefix tokenization differs from request tokenization");
            }
        }
    }
    size_t shared = 0;
    while (shared < common_prefix) {
        const llama_token token = sequences[0][shared];
        if (sequences[1][shared] != token || sequences[2][shared] != token) break;
        ++shared;
    }
    common_prefix = shared;
    if (common_prefix == 0) throw std::runtime_error("verdict candidates have no common prompt prefix");
    for (const auto & sequence : sequences) {
        if (common_prefix >= sequence.size()) {
            throw std::runtime_error("verdict answer produced no continuation token");
        }
    }

    if (common_prefix > static_cast<size_t>(llama_n_ctx(runtime.context))) {
        throw std::runtime_error("request does not fit in the shared GGUF context");
    }

    llama_memory_t memory = llama_get_memory(runtime.context);
    if (prefix_reused) {
        for (llama_seq_id sequence = 0; sequence < 3; ++sequence) {
            if (!llama_memory_seq_rm(memory, sequence, -1, -1)) {
                throw std::runtime_error("failed to clear a dynamic GGUF sequence");
            }
            llama_memory_seq_cp(
                memory, 3, sequence, 0, static_cast<llama_pos>(runtime.cached_prefix.size()));
        }
    } else {
        llama_memory_clear(memory, true);
    }

    clear_batch(runtime.batch);
    const std::vector<llama_seq_id> all_sequences = {0, 1, 2};
    for (size_t position = runtime.cached_prefix.size(); position < common_prefix; ++position) {
        add_token(runtime.batch, sequences[0][position], static_cast<llama_pos>(position), all_sequences, false);
    }
    runtime.batch.logits[runtime.batch.n_tokens - 1] = 1;
    std::vector<float> copied_logits(
        static_cast<size_t>(runtime.vocabulary_size));
    int32_t copied_outputs = 0;
    for (int32_t offset = 0; offset < runtime.batch.n_tokens; offset += runtime.batch_size) {
        const int32_t token_count = std::min(runtime.batch_size, runtime.batch.n_tokens - offset);
        llama_batch view = {
            token_count, runtime.batch.token + offset, nullptr, runtime.batch.pos + offset,
            runtime.batch.n_seq_id + offset, runtime.batch.seq_id + offset, runtime.batch.logits + offset,
        };
        const int32_t decode_status = llama_decode(runtime.context, view);
        if (decode_status != 0) {
            throw std::runtime_error("llama_decode failed with status " + std::to_string(decode_status));
        }
        int32_t outputs = 0;
        for (int32_t index = 0; index < token_count; ++index) outputs += view.logits[index] != 0;
        std::memcpy(
            copied_logits.data()
                + static_cast<size_t>(copied_outputs) * static_cast<size_t>(runtime.vocabulary_size),
            llama_get_logits(runtime.context),
            static_cast<size_t>(outputs) * static_cast<size_t>(runtime.vocabulary_size) * sizeof(float));
        copied_outputs += outputs;
    }
    if (copied_outputs != 1) {
        throw std::runtime_error("llama_decode returned an unexpected output count");
    }

    std::array<double, 3> scores{};
    const float * all_logits = copied_logits.data();
    for (size_t answer = 0; answer < sequences.size(); ++answer) {
        scores[answer] = token_log_probability(
            all_logits, runtime.vocabulary_size, sequences[answer][common_prefix]);
    }
    const auto finished = std::chrono::steady_clock::now();
    return {
        scores,
        std::chrono::duration_cast<std::chrono::microseconds>(finished - started).count(),
        common_prefix + 1,
        prefix_reused,
        runtime.cached_prefix.size(),
    };
}

bool valid_output_header(uint32_t struct_size, uint32_t abi_version, size_t expected_size) {
    return struct_size >= expected_size && abi_version == SG_GGUF_ABI_VERSION;
}

}  // namespace

extern "C" {

void sg_gguf_config_init(sg_gguf_config * config) {
    if (config == nullptr) return;
    std::memset(config, 0, sizeof(*config));
    config->struct_size = sizeof(*config);
    config->abi_version = SG_GGUF_ABI_VERSION;
    config->context_size = 640;
    config->batch_size = 640;
    config->micro_batch_size = 128;
    config->threads = 4;
    config->gpu_layers = 99;
}

void sg_gguf_runtime_info_init(sg_gguf_runtime_info * info) {
    if (info == nullptr) return;
    std::memset(info, 0, sizeof(*info));
    info->struct_size = sizeof(*info);
    info->abi_version = SG_GGUF_ABI_VERSION;
}

void sg_gguf_score_result_init(sg_gguf_score_result * result) {
    if (result == nullptr) return;
    std::memset(result, 0, sizeof(*result));
    result->struct_size = sizeof(*result);
    result->abi_version = SG_GGUF_ABI_VERSION;
}

sg_gguf_status sg_gguf_runtime_create(
    const char * model_path,
    const char * prefix_utf8,
    size_t prefix_bytes,
    const sg_gguf_config * config,
    sg_gguf_runtime ** runtime,
    char * error_message,
    size_t error_capacity) {
    clear_error(error_message, error_capacity);
    if (runtime != nullptr) *runtime = nullptr;
    if (model_path == nullptr || model_path[0] == '\0' || config == nullptr || runtime == nullptr
        || (prefix_bytes > 0 && prefix_utf8 == nullptr)
        || config->struct_size < sizeof(*config)
        || config->abi_version != SG_GGUF_ABI_VERSION
        || config->context_size < 1 || config->batch_size < 1
        || config->micro_batch_size < 1 || config->micro_batch_size > config->batch_size
        || config->threads < 1 || config->gpu_layers < 0
        || config->context_size > std::numeric_limits<int32_t>::max() - k_context_headroom) {
        write_error(error_message, error_capacity, "invalid ScamGuard GGUF runtime configuration");
        return SG_GGUF_INVALID_ARGUMENT;
    }

    try {
        auto created = std::make_unique<sg_gguf_runtime>();
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = config->gpu_layers;
        created->model = llama_model_load_from_file(model_path, model_params);
        if (created->model == nullptr) {
            write_error(error_message, error_capacity, "failed to load GGUF model");
            return SG_GGUF_MODEL_LOAD_FAILED;
        }

        llama_context_params context_params = llama_context_default_params();
        context_params.n_ctx = static_cast<uint32_t>(config->context_size + k_context_headroom);
        context_params.n_batch = static_cast<uint32_t>(config->batch_size);
        context_params.n_ubatch = static_cast<uint32_t>(config->micro_batch_size);
        context_params.n_seq_max = k_sequence_count;
        context_params.n_threads = config->threads;
        context_params.n_threads_batch = config->threads;
        context_params.kv_unified = true;
        context_params.no_perf = false;
        created->context = llama_init_from_model(created->model, context_params);
        if (created->context == nullptr) throw std::runtime_error("failed to create GGUF context");

        created->vocab = llama_model_get_vocab(created->model);
        created->vocabulary_size = llama_vocab_n_tokens(created->vocab);
        created->configured_context_size = config->context_size;
        created->batch_size = config->batch_size;
        created->batch = llama_batch_init(
            static_cast<int32_t>(context_params.n_ctx), 0, static_cast<int32_t>(context_params.n_seq_max));
        created->batch_initialized = true;
        if (prefix_bytes > 0) {
            created->cached_prefix = tokenize(created->vocab, std::string_view(prefix_utf8, prefix_bytes));
        }
        initialize_prefix_cache(created->context, created->batch, created->cached_prefix, created->batch_size);
        *runtime = created.release();
        return SG_GGUF_OK;
    } catch (const std::exception & error) {
        write_error(error_message, error_capacity, error.what());
        return SG_GGUF_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_message, error_capacity, "unknown GGUF runtime initialization error");
        return SG_GGUF_RUNTIME_ERROR;
    }
}

sg_gguf_status sg_gguf_runtime_get_info(
    const sg_gguf_runtime * runtime,
    sg_gguf_runtime_info * info,
    char * error_message,
    size_t error_capacity) {
    clear_error(error_message, error_capacity);
    if (runtime == nullptr || info == nullptr
        || !valid_output_header(info->struct_size, info->abi_version, sizeof(*info))) {
        write_error(error_message, error_capacity, "invalid ScamGuard GGUF runtime info buffer");
        return SG_GGUF_INVALID_ARGUMENT;
    }
    info->protocol_version = SG_GGUF_PROTOCOL_VERSION;
    info->model_bytes = llama_model_size(runtime->model);
    info->context_size = runtime->configured_context_size;
    info->prefix_tokens = static_cast<int32_t>(runtime->cached_prefix.size());
    return SG_GGUF_OK;
}

sg_gguf_status sg_gguf_runtime_score(
    sg_gguf_runtime * runtime,
    const char * question_utf8,
    size_t question_bytes,
    sg_gguf_score_result * result,
    char * error_message,
    size_t error_capacity) {
    clear_error(error_message, error_capacity);
    if (runtime == nullptr || result == nullptr || question_utf8 == nullptr || question_bytes == 0
        || !valid_output_header(result->struct_size, result->abi_version, sizeof(*result))) {
        write_error(error_message, error_capacity, "invalid ScamGuard GGUF scoring request");
        return SG_GGUF_INVALID_ARGUMENT;
    }
    try {
        std::lock_guard<std::mutex> lock(runtime->score_mutex);
        const internal_score_result scored = score_request(
            *runtime, std::string_view(question_utf8 == nullptr ? "" : question_utf8, question_bytes));
        result->safe_score = scored.scores[0];
        result->uncertain_score = scored.scores[1];
        result->scam_score = scored.scores[2];
        result->elapsed_microseconds = scored.elapsed_microseconds;
        result->maximum_sequence_tokens = static_cast<int32_t>(scored.maximum_sequence_tokens);
        result->prefix_reused = scored.prefix_reused ? 1 : 0;
        result->prefix_tokens = static_cast<int32_t>(scored.prefix_tokens);
        return SG_GGUF_OK;
    } catch (const std::exception & error) {
        write_error(error_message, error_capacity, error.what());
        return SG_GGUF_RUNTIME_ERROR;
    } catch (...) {
        write_error(error_message, error_capacity, "unknown GGUF scoring error");
        return SG_GGUF_RUNTIME_ERROR;
    }
}

void sg_gguf_runtime_destroy(sg_gguf_runtime * runtime) { delete runtime; }

}  // extern "C"
