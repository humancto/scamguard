#include "scamguard_gguf.h"

#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

struct options {
    std::string model;
    std::string prefix;
    sg_gguf_config config{};

    options() { sg_gguf_config_init(&config); }
};

[[noreturn]] void usage(const char * program) {
    std::cerr << "usage: " << program
              << " --model MODEL.gguf [--ctx-size 640] [--batch-size 640]"
                 " [--ubatch-size 128] [--threads 4] [--n-gpu-layers 99]"
                 " [--prefix-hex HEX]\n";
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

int hex_digit(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
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
        if (high < 0 || low < 0) throw std::runtime_error("prompt contains invalid hex");
        decoded.push_back(static_cast<char>((high << 4) | low));
    }
    return decoded;
}

options parse_options(int argc, char ** argv) {
    options result;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (index + 1 >= argc) usage(argv[0]);
        const char * value = argv[++index];
        if (argument == "--model") result.model = value;
        else if (argument == "--prefix-hex") result.prefix = decode_hex(value);
        else if (argument == "--ctx-size") result.config.context_size = parse_positive(value, "--ctx-size");
        else if (argument == "--batch-size") result.config.batch_size = parse_positive(value, "--batch-size");
        else if (argument == "--ubatch-size") result.config.micro_batch_size = parse_positive(value, "--ubatch-size");
        else if (argument == "--threads") result.config.threads = parse_positive(value, "--threads");
        else if (argument == "--n-gpu-layers") result.config.gpu_layers = parse_non_negative(value, "--n-gpu-layers");
        else usage(argv[0]);
    }
    if (result.model.empty()) usage(argv[0]);
    if (result.config.micro_batch_size > result.config.batch_size) {
        throw std::runtime_error("--ubatch-size cannot exceed --batch-size");
    }
    return result;
}

std::string protocol_safe(std::string message) {
    for (char & value : message) {
        if (value == '\t' || value == '\r' || value == '\n') value = ' ';
    }
    return message;
}

}  // namespace

int main(int argc, char ** argv) {
    try {
        const options settings = parse_options(argc, argv);
        char error[512] = {};
        sg_gguf_runtime * runtime = nullptr;
        const sg_gguf_status created = sg_gguf_runtime_create(
            settings.model.c_str(), settings.prefix.data(), settings.prefix.size(),
            &settings.config, &runtime, error, sizeof(error));
        if (created != SG_GGUF_OK) {
            throw std::runtime_error(error[0] == '\0' ? "failed to create GGUF runtime" : error);
        }

        sg_gguf_runtime_info info;
        sg_gguf_runtime_info_init(&info);
        if (sg_gguf_runtime_get_info(runtime, &info, error, sizeof(error)) != SG_GGUF_OK) {
            sg_gguf_runtime_destroy(runtime);
            throw std::runtime_error(error);
        }
        std::cout << "READY\t" << info.protocol_version << '\t' << info.model_bytes << '\t'
                  << info.context_size << '\t' << info.prefix_tokens << '\n';
        std::cout.flush();

        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") break;
            const size_t separator = line.find('\t');
            const std::string identifier = line.substr(0, separator);
            try {
                if (separator == std::string::npos || identifier.empty()) {
                    throw std::runtime_error("request must be ID followed by tab and prompt hex");
                }
                const std::string question = decode_hex(std::string_view(line).substr(separator + 1));
                sg_gguf_score_result result;
                sg_gguf_score_result_init(&result);
                if (sg_gguf_runtime_score(
                        runtime, question.data(), question.size(), &result, error, sizeof(error))
                    != SG_GGUF_OK) {
                    throw std::runtime_error(error[0] == '\0' ? "GGUF scoring failed" : error);
                }
                std::cout << "RESULT\t" << identifier << std::setprecision(12)
                          << '\t' << result.safe_score << '\t' << result.uncertain_score
                          << '\t' << result.scam_score << '\t' << result.elapsed_microseconds
                          << '\t' << result.maximum_sequence_tokens << '\t' << result.prefix_reused
                          << '\t' << result.prefix_tokens << '\n';
            } catch (const std::exception & caught) {
                std::cout << "ERROR\t" << identifier << '\t' << protocol_safe(caught.what()) << '\n';
            }
            std::cout.flush();
        }

        sg_gguf_runtime_destroy(runtime);
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
