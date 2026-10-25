// Short packed MHA: eight heads per Vector task, Q/K/V resident in UB.
// General masks, GQA, dropout and longer sequences retain the Cube path.
#pragma once
#include "kernel_operator.h"

namespace short_varlen {
template <typename T>
__global__ __aicore__ void Forward(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR cq, GM_ADDR ck,
                                 GM_ADDR out, GM_ADDR lse, uint32_t batch, uint32_t heads,
                                 uint32_t totalQ, float scale, bool causal)
{
#ifdef __DAV_C220_VEC__
    using namespace AscendC;
    constexpr uint32_t H = 8, D = 128, ROW = H * D, CAP = 4 * ROW;
    TPipe pipe;
    TBuf<TPosition::VECCALC> lowBuf, floatBuf, scratchBuf;
    pipe.InitBuffer(lowBuf, (3 * CAP + ROW) * sizeof(T));
    pipe.InitBuffer(floatBuf, 3 * CAP * sizeof(float));
    pipe.InitBuffer(scratchBuf, (3 * ROW + 256) * sizeof(float));
    auto low = lowBuf.Get<T>();
    auto fq = floatBuf.Get<float>();
    auto fk = fq[CAP];
    auto fv = fq[2 * CAP];
    auto prod = scratchBuf.Get<float>();
    auto tmp = prod[ROW];
    auto acc = prod[2 * ROW];
    auto scores = prod[3 * ROW];
    auto maximum = scores[32];
    auto denom = scores[40];
    auto ls = scores[48];
    auto weights = scores[64];
    GlobalTensor<T> gq, gk, gv, go;
    GlobalTensor<float> gl;
    GlobalTensor<int32_t> gcuq, gcuk;
    gq.SetGlobalBuffer((__gm__ T*)q); gk.SetGlobalBuffer((__gm__ T*)k);
    gv.SetGlobalBuffer((__gm__ T*)v); go.SetGlobalBuffer((__gm__ T*)out);
    gl.SetGlobalBuffer((__gm__ float*)lse);
    gcuq.SetGlobalBuffer((__gm__ int32_t*)cq); gcuk.SetGlobalBuffer((__gm__ int32_t*)ck);
    const uint32_t groups = heads / H;
    const uint32_t workers = GetBlockNum() * GetSubBlockNum();
    for (uint32_t task = GetBlockIdx(); task < batch * groups; task += workers) {
        const uint32_t b = task / groups, head = (task % groups) * H;
        const uint32_t qb = gcuq.GetValue(b), kb = gcuk.GetValue(b);
        const uint32_t nq = gcuq.GetValue(b + 1) - qb, nk = gcuk.GetValue(b + 1) - kb;
        if (nq == 0) continue;
        PipeBarrier<PIPE_ALL>();
        DataCopyPad(low, gq[(qb * heads + head) * D],
                    DataCopyExtParams(nq, ROW * sizeof(T), (heads - H) * D * sizeof(T), 0, 0),
                    DataCopyPadExtParams<T>(false, 0, 0, 0));
        if (nk > 0) {
            DataCopyPad(low[CAP], gk[(kb * heads + head) * D],
                        DataCopyExtParams(nk, ROW * sizeof(T), (heads - H) * D * sizeof(T), 0, 0),
                        DataCopyPadExtParams<T>(false, 0, 0, 0));
            DataCopyPad(low[2 * CAP], gv[(kb * heads + head) * D],
                        DataCopyExtParams(nk, ROW * sizeof(T), (heads - H) * D * sizeof(T), 0, 0),
                        DataCopyPadExtParams<T>(false, 0, 0, 0));
        }
        SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
        Cast(fq, low, RoundMode::CAST_NONE, nq * ROW);
        if (nk > 0) {
            Cast(fk, low[CAP], RoundMode::CAST_NONE, nk * ROW);
            Cast(fv, low[2 * CAP], RoundMode::CAST_NONE, nk * ROW);
        }
        PipeBarrier<PIPE_V>();
        for (uint32_t i = 0; i < nq; ++i) {
            const int32_t valid = causal ? Min(static_cast<int32_t>(nk),
                                               static_cast<int32_t>(i + nk) - static_cast<int32_t>(nq) + 1)
                                         : static_cast<int32_t>(nk);
            PipeBarrier<PIPE_ALL>();
            Duplicate(acc, 0.0f, ROW);
            if (valid > 0) {
                for (int32_t j = 0; j < valid; ++j) {
                    Mul(prod, fq[i * ROW], fk[j * ROW], ROW);
                    PipeBarrier<PIPE_V>();
                    // Fold D=128 to 64, then reduce each head independently.
                    Add(tmp, prod, prod[64], 64, H, {1, 1, 1, 8, 16, 16});
                    PipeBarrier<PIPE_V>();
                    WholeReduceSum(scores[j * H], tmp, 64, H, 1, 1, 8);
                    PipeBarrier<PIPE_V>();
                    Muls(scores[j * H], scores[j * H], scale, H);
                    PipeBarrier<PIPE_V>();
                }
                Adds(maximum, scores, 0.0f, H);
                PipeBarrier<PIPE_V>();
                for (int32_t j = 1; j < valid; ++j) {
                    Max(maximum, maximum, scores[j * H], H);
                    PipeBarrier<PIPE_V>();
                }
                Duplicate(denom, 0.0f, H);
                PipeBarrier<PIPE_V>();
                for (int32_t j = 0; j < valid; ++j) {
                    Sub(scores[j * H], scores[j * H], maximum, H);
                    PipeBarrier<PIPE_V>();
                    Exp(scores[j * H], scores[j * H], H);
                    PipeBarrier<PIPE_V>();
                    Add(denom, denom, scores[j * H], H);
                    PipeBarrier<PIPE_V>();
                }
                Ln(ls, denom, H);
                PipeBarrier<PIPE_V>();
                Add(ls, ls, maximum, H);
                PipeBarrier<PIPE_V>();
                for (int32_t j = 0; j < valid; ++j) {
                    Div(scores[j * H], scores[j * H], denom, H);
                    PipeBarrier<PIPE_V>();
                    Brcb(weights.ReinterpretCast<uint32_t>(), scores[j * H].ReinterpretCast<uint32_t>(), 1, {1, 8});
                    PipeBarrier<PIPE_V>();
                    Mul(tmp, fv[j * ROW], weights, 64, H, {1, 1, 0, 16, 16, 1});
                    Mul(tmp[64], fv[j * ROW + 64], weights, 64, H, {1, 1, 0, 16, 16, 1});
                    PipeBarrier<PIPE_V>();
                    Add(acc, acc, tmp, ROW);
                    PipeBarrier<PIPE_V>();
                }
            } else {
                Duplicate(ls, __builtin_inff(), H);
            }
            PipeBarrier<PIPE_V>();
            Cast(low[3 * CAP], acc, RoundMode::CAST_RINT, ROW);
            Brcb(weights.ReinterpretCast<uint32_t>(), ls.ReinterpretCast<uint32_t>(), 1, {1, 8});
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopy(go[((qb + i) * heads + head) * D], low[3 * CAP], ROW);
            DataCopyPad(gl[head * totalQ + qb + i], weights,
                        DataCopyExtParams(H, sizeof(float), 0, (totalQ - 1) * sizeof(float), 0));
        }
    }
    PipeBarrier<PIPE_ALL>();
#endif
}
} // namespace short_varlen
