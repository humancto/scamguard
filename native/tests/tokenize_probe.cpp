// Model-backed diagnostic: ID<TAB>UTF-8-HEX in, TOKENS<TAB>ID<TAB>CSV-IDS out.
// Load only the vocabulary; use exactly the production tokenization helper.
#include "scamguard_gguf.h"

#include "ggml-backend.h"
#include "llama.h"

#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace scamguard::detail {
// Internal C++ symbol, deliberately not added to the stable public C ABI.
std::vector<llama_token> tokenize_rendered_chat(const llama_vocab *, std::string_view);
}

namespace {

struct backend_lease {
    backend_lease() {
        ggml_backend_load_all();
        llama_backend_init();
    }
    ~backend_lease() { llama_backend_free(); }
};

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

}  // namespace

int main(int argc, char ** argv) {
    if (argc != 3 || std::string_view(argv[1]) != "--model") {
        std::cerr << "usage: " << argv[0] << " --model MODEL.gguf\n";
        return 2;
    }
    try {
        backend_lease backend;
        llama_model_params parameters = llama_model_default_params();
        parameters.vocab_only = true;
        parameters.n_gpu_layers = 0;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(
            llama_model_load_from_file(argv[2], parameters), llama_model_free);
        if (!model) throw std::runtime_error("failed to load GGUF vocabulary");
        const llama_vocab * vocab = llama_model_get_vocab(model.get());
        std::cout << "READY\t" << SG_GGUF_PROTOCOL_VERSION << '\n';
        std::cout.flush();
        std::string line;
        while (std::getline(std::cin, line)) {
            if (line == "QUIT") break;
            const size_t separator = line.find('\t');
            if (separator == std::string::npos || separator == 0) {
                throw std::runtime_error("request must be ID followed by tab and prompt hex");
            }
            const std::string_view identifier(line.data(), separator);
            if (identifier.find_first_of("\r\n") != std::string_view::npos) {
                throw std::runtime_error("request ID contains a control character");
            }
            const auto text = decode_hex(std::string_view(line).substr(separator + 1));
            const auto tokens = scamguard::detail::tokenize_rendered_chat(vocab, text);
            std::cout << "TOKENS\t" << identifier << '\t';
            for (size_t index = 0; index < tokens.size(); ++index) {
                if (index) std::cout << ',';
                std::cout << tokens[index];
            }
            std::cout << '\n';
            std::cout.flush();
        }
        return 0;
    } catch (const std::exception & error) {
        std::cerr << "fatal: " << error.what() << '\n';
        return 1;
    }
}
