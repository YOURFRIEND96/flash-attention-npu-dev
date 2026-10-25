#ifndef FAG_PREFIX_ROUTE_HPP
#define FAG_PREFIX_ROUTE_HPP

#include <cstdint>
#include "kernel_operator.h"

namespace fag_route {

// Compare contents on every invocation; no shape-based inference or persistent
// cache of user lengths. Call uniformly on all AIC/AIV cores before any
// collective or output write. Inputs must remain immutable for the launch.
__aicore__ inline bool EqualPrefixes(GM_ADDR qPrefix, GM_ADDR kPrefix, int64_t count)
{
    if (qPrefix == kPrefix) {
        return true;
    }
    const auto* q = reinterpret_cast<__gm__ int32_t*>(qPrefix);
    const auto* k = reinterpret_cast<__gm__ int32_t*>(kPrefix);
    for (int64_t b = 0; b < count; ++b) {
        if (q[b] != k[b]) {
            return false;
        }
    }
    return true;
}

} // namespace fag_route
#endif
