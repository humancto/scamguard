#ifndef SCAMGUARD_UNICODE_NFC_H
#define SCAMGUARD_UNICODE_NFC_H

#include <string>
#include <string_view>

namespace scamguard::detail {

// Normalize a length-delimited UTF-8 string without discarding embedded NULs.
// Invalid UTF-8 throws std::invalid_argument; allocation failures also propagate.
std::string normalize_nfc(std::string_view text);

}  // namespace scamguard::detail

#endif
