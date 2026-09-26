#pragma once

#include "ops/common/memory.cuh"
#include "ops/linear/q8/q8_geometry.h"

#include <cstdint>

namespace ninfer::ops::detail {

enum class Q8KSplitScaleAccess : std::uint8_t {
    Direct,
    Shared,
};

enum class Q8KSplitActivationStage : std::uint8_t {
    // Stage the compile-time column extent; tiled calls zero-fill its inactive columns.
    ActiveOnly,
    // Stage the full MMA tile, including padding beyond the compile-time column extent.
    PaddedZero,
    // Stage only live columns. Inactive MMA columns are discarded by the store epilogue.
    RuntimeActive,
};

template <int RequestedWarps, int TileTokens>
inline constexpr int q8_physical_k_warps =
#if defined(NINFER_SM75)
    // Per K warp: 16x64 code bytes, 16x4 scale bytes, and TileTokens x64 BF16 values.
    RequestedWarps * (1088 + TileTokens * 128) <= 48 * 1024 ? RequestedWarps :
    (RequestedWarps / 2) * (1088 + TileTokens * 128) <= 48 * 1024 ? RequestedWarps / 2 :
    (RequestedWarps / 4) * (1088 + TileTokens * 128) <= 48 * 1024 ? RequestedWarps / 4 :
    RequestedWarps / 8;
#else
    RequestedWarps;
#endif

template <int KWarps, int TileTokens, int MinBlocksPerSm, Q8KSplitScaleAccess ScaleAccess,
          Cache ActivationCache = Cache::ca, Cache WeightCache = Cache::cg,
          Q8KSplitActivationStage ActivationStage = Q8KSplitActivationStage::ActiveOnly>
struct Q8KSplitSchedule {
    static_assert(KWarps == 4 || KWarps == 8 || KWarps == 16);
    static_assert(TileTokens == 8 || TileTokens == 16 || TileTokens == 24 || TileTokens == 32 ||
                  TileTokens == 40 || TileTokens == 48 || TileTokens == 56 || TileTokens == 64 ||
                  TileTokens == 72 || TileTokens == 80 || TileTokens == 88);
    static_assert(MinBlocksPerSm > 0);

    static constexpr int kKWarps            = q8_physical_k_warps<KWarps, TileTokens>;
    static constexpr int kTileTokens        = TileTokens;
    static constexpr int kMinBlocksPerSm    = MinBlocksPerSm;
    static constexpr auto kScaleAccess      = ScaleAccess;
    static constexpr auto kActivationCache  = ActivationCache;
    static constexpr auto kWeightCache      = WeightCache;
    static constexpr auto kActivationStage  = ActivationStage;
    static constexpr int kThreads           = kKWarps * 32;
    static constexpr int kTileKPerWarp      = 64;
    static constexpr int kGroupK            = kKWarps * kTileKPerWarp;
    static constexpr int kRowsPerCta        = 16;
    static constexpr int kRowsPerLoaderWarp = kRowsPerCta / kKWarps;
    static constexpr int kScaleBytesPerRow  = kGroupK / 16;
};

template <int TileTokens, int ActiveTokens>
using Q8KSplitDefaultSchedule = Q8KSplitSchedule<
    8, TileTokens, TileTokens == 8 ? 5 : (TileTokens == 16 ? 4 : (TileTokens == 24 ? 3 : 2)),
    (ActiveTokens > 4 ? Q8KSplitScaleAccess::Shared : Q8KSplitScaleAccess::Direct)>;

} // namespace ninfer::ops::detail
