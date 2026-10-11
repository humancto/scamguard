#include "unicode_nfc.h"

#include "utf8proc.h"

#include <cstdlib>
#include <limits>
#include <memory>
#include <stdexcept>

namespace scamguard::detail {

std::string normalize_nfc(std::string_view text) {
    if (text.empty()) return {};
    if (text.size() > static_cast<size_t>(std::numeric_limits<utf8proc_ssize_t>::max())) {
        throw std::length_error("input exceeds Unicode normalizer length limit");
    }

    utf8proc_uint8_t * mapped = nullptr;
    const utf8proc_ssize_t length = utf8proc_map(
        reinterpret_cast<const utf8proc_uint8_t *>(text.data()),
        static_cast<utf8proc_ssize_t>(text.size()), &mapped,
        static_cast<utf8proc_option_t>(UTF8PROC_STABLE | UTF8PROC_COMPOSE));
    // utf8proc allocates with malloc. Keep ownership even on an error path.
    const std::unique_ptr<utf8proc_uint8_t, decltype(&std::free)> normalized(mapped, &std::free);
    if (length == UTF8PROC_ERROR_INVALIDUTF8) {
        throw std::invalid_argument("invalid UTF-8 in rendered chat");
    }
    if (length < 0) {
        throw std::runtime_error(std::string("Unicode normalization failed: ") + utf8proc_errmsg(length));
    }
    if (length == 0) return {};
    if (!normalized) throw std::runtime_error("Unicode normalization returned no buffer");
    return std::string(reinterpret_cast<const char *>(normalized.get()), static_cast<size_t>(length));
}

}  // namespace scamguard::detail
