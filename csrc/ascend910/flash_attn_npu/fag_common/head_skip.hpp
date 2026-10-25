#ifndef FAG_HEAD_SKIP_HPP
#define FAG_HEAD_SKIP_HPP

#include "common_header.h"

// Skip complete heads belonging to another core only when all of their
// tiles fit before the existing segment flush. The actual per-batch tile
// geometry is used; sequence lengths need not be uniform across batches.
template<MaskType maskType, typename Addr>
__aicore__ inline bool SkipUnownedHeads(Addr &addr, int32_t targetSegment)
{
    if (addr.qSeqIdx != 0 || addr.seqKIdx != 0 || addr.nheadsIdx + 1 >= addr.nheads ||
        addr.coreSegmentBlockNum == targetSegment) {
        return false;
    }
    int64_t headTiles;
    if constexpr (maskType == MaskType::NO_MASK) {
        headTiles = static_cast<int64_t>(addr.s1BlockNum) * addr.s2BlockNum;
    } else {
        headTiles = static_cast<int64_t>(addr.s1BlockNum) * (addr.s1BlockNum + 1) / 2;
    }
    const int32_t remaining = 15 - addr.blockNum;
    if (headTiles <= 0 || headTiles > remaining) {
        return false;
    }
    const int32_t fittingHeads = remaining / static_cast<int32_t>(headTiles);
    // Leave the last head for the original batch transition and termination.
    const int32_t remainingHeads = addr.nheads - addr.nheadsIdx - 1;
    const int32_t skipHeads = fittingHeads < remainingHeads ? fittingHeads : remainingHeads;
    addr.nheadsIdx += skipHeads;
    addr.blockNum += skipHeads * static_cast<int32_t>(headTiles);
    addr.SegmentBlockNum += skipHeads * ((addr.s1BlockNum + 1) / 2);
    return true;
}

#endif
