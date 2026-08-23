#include "ggml-backend.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::array<std::string_view, 3> k_answers = {"SAFE\"", "UNCERTAIN\"", "SCAM\""};

struct options {
    std::string model;
    int32_t ctx_size = 640;
    int32_t batch_size = 640;
    int32_t ubatch_size = 128;
    int32_t threads = 4;
    int32_t gpu_layers = 99;
};

[[noreturn]] void usage(const char * program) {
    std::cerr << "usage: " << program
              << " --model MODEL.gguf [--ctx-size 640] [--batch-size 640]"
                 " [--ubatch-size 128] [--threads 4] [--n-gpu-layers 99]\n";
    std::exit(2);
}

int32_t parse_positive(const char * value, const char * name) {
    try {
        const long parsed = std::stol(value);
        if (parsed < 1 || parsed > std::numeric_limits<int32_t>::max()) {
            throw std::out_of_range(name);
        }
        return static_cast<int32_t>(parsed);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string(name) + " must be a positive integer");
    }
}

int32_t parse_non_negative(const char * value, const char * name) {
    try {
        const long parsed = std::stol(value);
        if (parsed < 0 || parsed > std::numeric_limits<int32_t>::max()) {
            throw std::out_of_range(name);
        }
        return static_cast<int32_t>(parsed);
    } catch (const std::exception &) {
        throw std::runtime_error(std::string(name) + " must be a non-negative integer");
    }
}

options parse_options(int argc, char ** argv) {
    options result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (index + 1 >= argc) {
            usage(argv[0]);
        }
        const char * value = argv[++index];
        if (argument == "--model") {
            result.model = value;
        } else if (argument == "--ctx-size") {
            result.ctx_size = parse_positive(value, "--ctx-size");
        } else if (argument == "--batch-size") {
            result.batch_size = parse_positive(value, "--batch-size");
        } else if (argument == "--ubatch-size") {
            result.ubatch_size = parse_positive(value, "--ubatch-size");
        } else if (argument == "--threads") {
            result.threads = parse_positive(value, "--threads");
        } else if (argument == "--n-gpu-layers") {
            result.gpu_layers = parse_non_negative(value, "--n-gpu-layers");
        } else {
            usage(argv[0]);
        }
    }
    if (result.model.empty()) {
        usage(argv[0]);
    }
    if (result.ubatch_size > result.batch_size) {
        throw std::runtime_error("--ubatch-size cannot exceed --batch-size");
    }
    return result;
}

int hex_digit(char value) {
    if (value >= '0' && value <= '9') {
        return value - '0';
    }
    if (value >= 'a' && value <= 'f') {
        return value - 'a' + 10;
    }
    if (value >= 'A' && value <= 'F') {
        return value - 'A' + 10;
    }
    return -1;
}

std::string decode_hex(std::string_view encoded) {
    if (encoded.empty() || encoded.size() % 2 != 0) {
        throw std::runtime_error("prompt hex must be non-empty and even-length");
    }
    std::string decoded;
    decoded.reserve(encoded.size() / 2);
    for (size_t index = 0; index < encoded.size(); index += 2) {
        const int high = hex_digit(encoded[index]);
        const int low = hex_digit(encoded[index + 1]);
        if (high < 0 || low < 0) {
            throw std::runtime_error("prompt contains invalid hex");
        }
        decoded.push_back(static_cast<char>((high << 4) | low));
    }
    return decoded;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text) {
    const int32_t required = -llama_tokenize(
        vocab, text.data(), static_cast<int32_t>(text.size()), nullptr, 0, true, false);
    if (required <= 0) {
        throw std::runtime_error("tokenizer did not return a positive token count");
    }
    std::vector<llama_token> tokens(static_cast<size_t>(required));
    const int32_t written = llama_tokenize(
        vocab,
        text.data(),
        static_cast<int32_t>(text.size()),
        tokens.data(),
        static_cast<int32_t>(tokens.size()),
        true,
        false);
    if (written != required) {
        throw std::runtime_error("tokenizer output size changed between calls");
    }
    return tokens;
}

void clear_batch(llama_batch & batch) {
    batch.n_tokens = 0;
}

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

struct score_result {
    std::array<double, 3> scores;
    int64_t elapsed_microseconds;
    size_t maximum_sequence_tokens;
};

score_result score_request(
    llama_context * context,
    const llama_vocab * vocab,
    llama_batch & batch,
    int32_t vocabulary_size,
    int32_t context_size,
    int32_t batch_size,
    const std::string & question) {
    const auto started = std::chrono::steady_clock::now();
    std::array<std::vector<llama_token>, 3> sequences;
    size_t maximum_sequence_tokens = 0;
    for (size_t answer = 0; answer < k_answers.size(); ++answer) {
        sequences[answer] = tokenize(vocab, question + std::string(k_answers[answer]));
        maximum_sequence_tokens = std::max(maximum_sequence_tokens, sequences[answer].size());
        if (sequences[answer].size() > static_cast<size_t>(context_size)) {
            throw std::runtime_error("request exceeds the configured per-sequence context");
        }
    }

    size_t common_prefix = maximum_sequence_tokens;
    for (const auto & sequence : sequences) {
        common_prefix = std::min(common_prefix, sequence.size());
    }
    size_t shared = 0;
    while (shared < common_prefix) {
        const llama_token token = sequences[0][shared];
        if (sequences[1][shared] != token || sequences[2][shared] != token) {
            break;
        }
        ++shared;
    }
    common_prefix = shared;
    if (common_prefix == 0) {
        throw std::runtime_error("verdict candidates have no common prompt prefix");
    }
    for (const auto & sequence : sequences) {
        if (common_prefix >= sequence.size()) {
            throw std::runtime_error("verdict answer produced no continuation token");
        }
    }

    size_t required_tokens = common_prefix;
    for (const auto & sequence : sequences) {
        required_tokens += sequence.size() - common_prefix;
    }
    if (required_tokens > static_cast<size_t>(llama_n_ctx(context))) {
        throw std::runtime_error("request does not fit in the shared GGUF context");
    }

    clear_batch(batch);
    const std::vector<llama_seq_id> all_sequences = {0, 1, 2};
    for (size_t position = 0; position < common_prefix; ++position) {
        add_token(
            batch,
            sequences[0][position],
            static_cast<llama_pos>(position),
            all_sequences,
            false);
    }
    batch.logits[batch.n_tokens - 1] = 1;
    std::array<std::vector<int32_t>, 3> continuation_output_indices;
    int32_t next_output_index = 1;
    for (size_t answer = 0; answer < sequences.size(); ++answer) {
        for (size_t position = common_prefix; position < sequences[answer].size(); ++position) {
            const bool needs_output = position + 1 < sequences[answer].size();
            add_token(
                batch,
                sequences[answer][position],
                static_cast<llama_pos>(position),
                {static_cast<llama_seq_id>(answer)},
                needs_output);
            if (needs_output) {
                continuation_output_indices[answer].push_back(next_output_index++);
            }
        }
    }

    llama_memory_clear(llama_get_memory(context), true);
    std::vector<float> copied_logits(
        static_cast<size_t>(next_output_index) * static_cast<size_t>(vocabulary_size));
    int32_t copied_outputs = 0;
    for (int32_t offset = 0; offset < batch.n_tokens; offset += batch_size) {
        const int32_t token_count = std::min(batch_size, batch.n_tokens - offset);
        llama_batch view = {
            token_count,
            batch.token + offset,
            nullptr,
            batch.pos + offset,
            batch.n_seq_id + offset,
            batch.seq_id + offset,
            batch.logits + offset,
        };
        const int32_t decode_status = llama_decode(context, view);
        if (decode_status != 0) {
            throw std::runtime_error(
                "llama_decode failed with status " + std::to_string(decode_status));
        }
        int32_t outputs = 0;
        for (int32_t index = 0; index < token_count; ++index) {
            outputs += view.logits[index] != 0;
        }
        std::memcpy(
            copied_logits.data()
                + static_cast<size_t>(copied_outputs) * static_cast<size_t>(vocabulary_size),
            llama_get_logits(context),
            static_cast<size_t>(outputs) * static_cast<size_t>(vocabulary_size) * sizeof(float));
        copied_outputs += outputs;
    }
    if (copied_outputs != next_output_index) {
        throw std::runtime_error("llama_decode returned an unexpected output count");
    }

    std::array<double, 3> scores{};
    const float * all_logits = copied_logits.data();
    const float * first_logits = all_logits;
    for (size_t answer = 0; answer < sequences.size(); ++answer) {
        double log_probability = token_log_probability(
            first_logits, vocabulary_size, sequences[answer][common_prefix]);
        size_t count = 1;
        for (size_t offset = 0; offset < continuation_output_indices[answer].size(); ++offset) {
            const size_t target_position = common_prefix + offset + 1;
            const double continuation_probability = token_log_probability(
                all_logits
                    + static_cast<size_t>(continuation_output_indices[answer][offset])
                        * static_cast<size_t>(vocabulary_size),
                vocabulary_size,
                sequences[answer][target_position]);
            log_probability += continuation_probability;
            ++count;
        }
        scores[answer] = log_probability / static_cast<double>(count);
    }
    const auto finished = std::chrono::steady_clock::now();
    return {
        scores,
        std::chrono::duration_cast<std::chrono::microseconds>(finished - started).count(),
        maximum_sequence_tokens,
    };
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const options settings = parse_options(argc, argv);
        ggml_backend_load_all();
        llama_backend_init();

        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = settings.gpu_layers;
        llama_model * model = llama_model_load_from_file(settings.model.c_str(), model_params);
        if (model == nullptr) {
            throw std::runtime_error("failed to load GGUF model");
        }

        llama_context_params context_params = llama_context_default_params();
        // The common prompt prefix is stored once and shared across the three verdict
        // continuations. One 64-token bucket of headroom covers their divergent suffixes.
        context_params.n_ctx = static_cast<uint32_t>(settings.ctx_size + 64);
        context_params.n_batch = static_cast<uint32_t>(settings.batch_size);
        context_params.n_ubatch = static_cast<uint32_t>(settings.ubatch_size);
        context_params.n_seq_max = 4;
        context_params.n_threads = settings.threads;
        context_params.n_threads_batch = settings.threads;
        context_params.kv_unified = true;
        context_params.no_perf = false;
        llama_context * context = llama_init_from_model(model, context_params);
        if (context == nullptr) {
            llama_model_free(model);
            throw std::runtime_error("failed to create GGUF context");
        }

        const llama_vocab * vocab = llama_model_get_vocab(model);
        const int32_t vocabulary_size = llama_vocab_n_tokens(vocab);
        llama_batch batch = llama_batch_init(
            static_cast<int32_t>(context_params.n_ctx), 0, static_cast<int32_t>(context_params.n_seq_max));

        std::cout << "READY\t1\t" << llama_model_size(model) << '\t' << settings.ctx_size << '\n';
        std::cout.flush();

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") {
                break;
            }
            const size_t separator = line.find('\t');
            const std::string identifier = line.substr(0, separator);
            try {
                if (separator == std::string::npos || identifier.empty()) {
                    throw std::runtime_error("request must be ID followed by tab and prompt hex");
                }
                const std::string question = decode_hex(std::string_view(line).substr(separator + 1));
                const score_result result = score_request(
                    context,
                    vocab,
                    batch,
                    vocabulary_size,
                    settings.ctx_size,
                    settings.batch_size,
                    question);
                std::cout << "RESULT\t" << identifier << std::setprecision(12);
                for (const double score : result.scores) {
                    std::cout << '\t' << score;
                }
                std::cout << '\t' << result.elapsed_microseconds << '\t'
                          << result.maximum_sequence_tokens << '\n';
            } catch (const std::exception & error) {
                std::cout << "ERROR\t" << identifier << '\t' << error.what() << '\n';
            }
            std::cout.flush();
        }

        llama_batch_free(batch);
        llama_free(context);
        llama_model_free(model);
        llama_backend_free();
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
