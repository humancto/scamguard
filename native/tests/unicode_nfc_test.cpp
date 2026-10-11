#include "unicode_nfc.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

using namespace std::string_literals;

namespace {

void expect(std::string_view name, std::string_view input, std::string_view expected) {
    const std::string actual = scamguard::detail::normalize_nfc(input);
    if (actual != expected) throw std::runtime_error(std::string(name) + ": unexpected normalization");
    if (scamguard::detail::normalize_nfc(actual) != actual) {
        throw std::runtime_error(std::string(name) + ": normalization was not idempotent");
    }
}

void reject(std::string_view name, std::string_view input) {
    try {
        (void) scamguard::detail::normalize_nfc(input);
    } catch (const std::invalid_argument &) {
        return;
    }
    throw std::runtime_error(std::string(name) + ": accepted invalid UTF-8");
}

}  // namespace

int main() {
    try {
        expect("empty", {}, "");
        expect("ascii and controls", "<|im_start|>user\r\n\tHi", "<|im_start|>user\r\n\tHi");
        expect("decomposed accent", u8"cafe\u0301", u8"caf\u00E9");
        expect("composed accent", u8"caf\u00E9", u8"caf\u00E9");
        expect("hangul composition", u8"\u1100\u1161\u11A8", u8"\uAC01");
        expect("canonical mark order", u8"a\u0315\u0300", u8"\u00E0\u0315");
        expect("Hindi marks", u8"\u0915\u093E", u8"\u0915\u093E");
        expect("no compatibility folding", u8"\u2460", u8"\u2460");
        expect("emoji variation and joiner", u8"\U0001F469\u200D\U0001F4BB\uFE0F", u8"\U0001F469\u200D\U0001F4BB\uFE0F");
        expect("embedded NUL", u8"e\u0301\0A\u030A"s, u8"\u00E9\0\u00C5"s);
        expect("NUL only", "\0"s, "\0"s);
        reject("lone continuation", "\x80");
        reject("overlong encoding", "\xC0\xAF");
        reject("truncated sequence", "\xE2\x82");
        reject("surrogate", "\xED\xA0\x80");
        reject("beyond Unicode range", "\xF4\x90\x80\x80");
        reject("invalid after NUL", "valid\0\x80"s);
        std::cout << "17 Unicode normalization cases passed\n";
        return 0;
    } catch (const std::exception & error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
