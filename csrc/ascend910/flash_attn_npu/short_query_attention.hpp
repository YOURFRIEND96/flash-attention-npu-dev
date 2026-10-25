#pragma once
#include "kernel_operator.h"

namespace short_query {
using namespace AscendC;
constexpr uint32_t D = 128, R = 32, Q = 4, TILE = R * D;

__aicore__ inline void Dot(LocalTensor<float> dst, LocalTensor<float> prod, LocalTensor<float> tmp,
    LocalTensor<float> a, LocalTensor<float> b, uint32_t rows,
    uint8_t strideA, uint8_t strideB)
{
    Mul(prod, a, b, 64, rows, {1, 1, 1, 16, strideA, strideB});
    Mul(prod[64], a[64], b[64], 64, rows, {1, 1, 1, 16, strideA, strideB});
    PipeBarrier<PIPE_V>();
    Add(tmp, prod, prod[64], 64, rows, {1, 1, 1, 8, 16, 16});
    PipeBarrier<PIPE_V>();
    WholeReduceSum(dst, tmp, 64, rows, 1, 1, 8);
    PipeBarrier<PIPE_V>();
}

// Accumulate a weighted reduction over KV rows into one D-vector.
__aicore__ inline void WeightedAdd(LocalTensor<float> dst, LocalTensor<float> matrix,
    LocalTensor<float> weights, LocalTensor<float> broad,
    LocalTensor<float> prod, uint32_t rows)
{
    Duplicate(prod, 0.0f, TILE);
    Brcb(broad.ReinterpretCast<uint32_t>(), weights.ReinterpretCast<uint32_t>(), 4, {1, 8});
    PipeBarrier<PIPE_V>();
    Mul(prod, matrix, broad, 64, rows, {1, 1, 0, 16, 16, 1});
    Mul(prod[64], matrix[64], broad, 64, rows, {1, 1, 0, 16, 16, 1});
    PipeBarrier<PIPE_V>();
    for (uint32_t n = R / 2; n > 0; n /= 2) {
        Add(prod, prod, prod[n * D], n * D);
        PipeBarrier<PIPE_V>();
    }
    Add(dst, dst, prod, D);
    PipeBarrier<PIPE_V>();
}

__aicore__ inline void OuterAdd(LocalTensor<float> dst, LocalTensor<float> vector,
    LocalTensor<float> weights, LocalTensor<float> broad,
    LocalTensor<float> prod, uint32_t rows)
{
    Brcb(broad.ReinterpretCast<uint32_t>(), weights.ReinterpretCast<uint32_t>(), 4, {1, 8});
    PipeBarrier<PIPE_V>();
    Mul(prod, vector, broad, 64, rows, {1, 1, 0, 16, 0, 1});
    Mul(prod[64], vector[64], broad, 64, rows, {1, 1, 0, 16, 0, 1});
    PipeBarrier<PIPE_V>();
    Add(dst, dst, prod, rows * D);
    PipeBarrier<PIPE_V>();
}

// Each Vector task owns one batch/head and every gradient element it writes.
// Stream KV in 32-row tiles; keep all short Q rows and dQ in UB. No atomics,
// global score/gradient workspace, zeroing kernel or postprocessing kernel.
template <typename T>
__global__ __aicore__ void Backward(GM_ADDR q, GM_ADDR k, GM_ADDR v, GM_ADDR cq, GM_ADDR ck,
    GM_ADDR out, GM_ADDR lse, GM_ADDR dout, GM_ADDR dq,
    GM_ADDR dk, GM_ADDR dv, uint32_t batch, uint32_t heads,
    uint32_t totalQ, float scale, bool causal)
{
#ifdef __DAV_C220_VEC__
    TPipe pipe;
    TBuf<TPosition::VECCALC> lowBuf, dataBuf, scratchBuf;
    pipe.InitBuffer(lowBuf, (3 * Q * D + 3 * TILE) * sizeof(T));
    pipe.InitBuffer(dataBuf, (4 * Q * D + 4 * TILE) * sizeof(float));
    pipe.InitBuffer(scratchBuf, (2 * TILE + 768) * sizeof(float));
    auto low = lowBuf.Get<T>();
    auto fq = dataBuf.Get<float>();
    auto fdo = fq[Q * D], fo = fq[2 * Q * D], accum = fq[3 * Q * D];
    auto fk = fq[4 * Q * D], fv = fk[TILE], fdk = fk[2 * TILE], fdv = fk[3 * TILE];
    auto prod = scratchBuf.Get<float>(), tmp = prod[TILE], scores = prod[2 * TILE];
    auto dp = scores[32], broad = scores[64];
    auto lseSmall = scores[384], lseBroad = scores[416], delta = scores[480], deltaBroad = scores[512];
    auto lowK = low[3 * Q * D], lowV = lowK[TILE], lowResult = lowK[2 * TILE];
    GlobalTensor<T> gq, gk, gv, go, gdo, gdq, gdk, gdv;
    GlobalTensor<float> gl;
    GlobalTensor<int32_t> gcuq, gcuk;
    gq.SetGlobalBuffer((__gm__ T*)q); gk.SetGlobalBuffer((__gm__ T*)k);
    gv.SetGlobalBuffer((__gm__ T*)v); go.SetGlobalBuffer((__gm__ T*)out);
    gl.SetGlobalBuffer((__gm__ float*)lse);
    gcuq.SetGlobalBuffer((__gm__ int32_t*)cq); gcuk.SetGlobalBuffer((__gm__ int32_t*)ck);

    gdo.SetGlobalBuffer((__gm__ T*)dout); gdq.SetGlobalBuffer((__gm__ T*)dq);
    gdk.SetGlobalBuffer((__gm__ T*)dk); gdv.SetGlobalBuffer((__gm__ T*)dv);

    const uint32_t workers = GetBlockNum() * GetSubBlockNum();
    for (uint32_t task = GetBlockIdx(); task < batch * heads; task += workers) {
        const uint32_t b = task / heads, h = task % heads;
        const uint32_t qb = gcuq.GetValue(b), kb = gcuk.GetValue(b);
        const uint32_t nq = gcuq.GetValue(b + 1) - qb, nk = gcuk.GetValue(b + 1) - kb;
        PipeBarrier<PIPE_ALL>();
        if (nq > 0) {
            const DataCopyExtParams qCopy(nq, D * sizeof(T), (heads - 1) * D * sizeof(T), 0, 0);
            DataCopyPad(low, gq[(qb * heads + h) * D], qCopy, DataCopyPadExtParams<T>(false, 0, 0, 0));

            DataCopyPad(low[Q * D], gdo[(qb * heads + h) * D], qCopy, DataCopyPadExtParams<T>(false, 0, 0, 0));
            DataCopyPad(low[2 * Q * D], go[(qb * heads + h) * D], qCopy, DataCopyPadExtParams<T>(false, 0, 0, 0));
            DataCopyPad(lseSmall, gl[h * totalQ + qb], DataCopyExtParams(1, nq * sizeof(float), 0, 0, 0),
                DataCopyPadExtParams<float>(true, 0, 8 - nq, 0));

            SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(fq, low, RoundMode::CAST_NONE, nq * D);

            Cast(fdo, low[Q * D], RoundMode::CAST_NONE, nq * D);
            Cast(fo, low[2 * Q * D], RoundMode::CAST_NONE, nq * D);
            PipeBarrier<PIPE_V>();
            Dot(delta, prod, tmp, fdo, fo, nq, 16, 16);
            Brcb(deltaBroad.ReinterpretCast<uint32_t>(), delta.ReinterpretCast<uint32_t>(), 1, {1, 8});
            Brcb(lseBroad.ReinterpretCast<uint32_t>(), lseSmall.ReinterpretCast<uint32_t>(), 1, {1, 8});

        }
        Duplicate(accum, 0.0f, Q * D);

        PipeBarrier<PIPE_V>();
        for (uint32_t start = 0; start < nk; start += R) {
            const uint32_t rows = nk - start < R ? nk - start : R;
            PipeBarrier<PIPE_ALL>();
            const DataCopyExtParams kvCopy(rows, D * sizeof(T), (heads - 1) * D * sizeof(T), 0, 0);
            DataCopyPad(lowK, gk[((kb + start) * heads + h) * D], kvCopy, DataCopyPadExtParams<T>(false, 0, 0, 0));
            DataCopyPad(lowV, gv[((kb + start) * heads + h) * D], kvCopy, DataCopyPadExtParams<T>(false, 0, 0, 0));
            SetFlag<HardEvent::MTE2_V>(EVENT_ID0); WaitFlag<HardEvent::MTE2_V>(EVENT_ID0);
            Cast(fk, lowK, RoundMode::CAST_NONE, rows * D);
            Cast(fv, lowV, RoundMode::CAST_NONE, rows * D);
            Duplicate(fdk, 0.0f, TILE); Duplicate(fdv, 0.0f, TILE);
            PipeBarrier<PIPE_V>();
            for (uint32_t i = 0; i < nq; ++i) {
                int32_t valid = rows;
                if (causal) {
                    const int32_t end = static_cast<int32_t>(i + nk) - static_cast<int32_t>(nq) + 1;
                    valid = end - static_cast<int32_t>(start);
                    if (valid > static_cast<int32_t>(rows)) valid = rows;
                }
                if (valid <= 0) continue;
                Dot(scores, prod, tmp, fq[i * D], fk, valid, 0, 16);
                Muls(scores, scores, scale, valid);
                PipeBarrier<PIPE_V>();

                Sub(scores, scores, lseBroad[i * 8], valid, 1, {1, 1, 0, 8, 8, 0});
                PipeBarrier<PIPE_V>();
                Exp(scores, scores, valid);
                PipeBarrier<PIPE_V>();
                Dot(dp, prod, tmp, fdo[i * D], fv, valid, 0, 16);
                Sub(dp, dp, deltaBroad[i * 8], valid, 1, {1, 1, 0, 8, 8, 0});
                PipeBarrier<PIPE_V>();
                Mul(dp, dp, scores, valid);
                PipeBarrier<PIPE_V>();
                WeightedAdd(accum[i * D], fk, dp, broad, prod, valid);
                OuterAdd(fdk, fq[i * D], dp, broad, prod, valid);
                OuterAdd(fdv, fdo[i * D], scores, broad, prod, valid);

            }

            Muls(fdk, fdk, scale, rows * D);
            PipeBarrier<PIPE_V>();
            Cast(lowResult, fdk, RoundMode::CAST_RINT, rows * D);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(gdk[((kb + start) * heads + h) * D], lowResult,
                DataCopyExtParams(rows, D * sizeof(T), 0, (heads - 1) * D * sizeof(T), 0));
            PipeBarrier<PIPE_ALL>();
            Cast(lowResult, fdv, RoundMode::CAST_RINT, rows * D);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);
            DataCopyPad(gdv[((kb + start) * heads + h) * D], lowResult,
                DataCopyExtParams(rows, D * sizeof(T), 0, (heads - 1) * D * sizeof(T), 0));

        }
        for (uint32_t i = 0; i < nq; ++i) {
            PipeBarrier<PIPE_ALL>();

            Muls(accum[i * D], accum[i * D], scale, D);

            PipeBarrier<PIPE_V>();
            Cast(lowResult, accum[i * D], RoundMode::CAST_RINT, D);
            SetFlag<HardEvent::V_MTE3>(EVENT_ID0); WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

            DataCopy(gdq[((qb + i) * heads + h) * D], lowResult, D);

        }
    }
    PipeBarrier<PIPE_ALL>();
#endif
}
} // namespace short_query
