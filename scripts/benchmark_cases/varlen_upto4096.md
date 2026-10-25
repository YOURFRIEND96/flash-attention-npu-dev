# Varlen generalization performance cases up to 4096

Synthetic, reproducible ragged lengths are stored in `varlen_upto4096_lengths.json`.
The 4096 cases test the upper boundary; all other maximum lengths are below 4096.
Batch size changes with length to bound test cost; this is not a fixed-batch scaling curve.
All cases use positive lengths, Hq=8, dropout=0, no local window/ALiBi/softcap.
MHA/GQA4 self-attention cases use matching Q/K prefixes, with unequal lengths between batches.

| Case | B | Q shape | K shape = V shape | O shape | Max Q/K | Mask | Scale |
|---|---:|---|---|---|---|---|---|
| MHA causal S64 | 32 | [1327, 8, 128] | [1327, 8, 128] | [1327, 8, 128] | 64/64 | causal | 1/√128 |
| GQA4 causal S64 | 32 | [1327, 8, 128] | [1327, 2, 128] | [1327, 8, 128] | 64/64 | causal | 1/√128 |
| MHA causal S128 | 16 | [1375, 8, 128] | [1375, 8, 128] | [1375, 8, 128] | 128/128 | causal | 1/√128 |
| GQA4 causal S128 | 16 | [1375, 8, 128] | [1375, 2, 128] | [1375, 8, 128] | 128/128 | causal | 1/√128 |
| MHA causal S256 | 16 | [2751, 8, 128] | [2751, 8, 128] | [2751, 8, 128] | 256/256 | causal | 1/√128 |
| GQA4 causal S256 | 16 | [2751, 8, 128] | [2751, 2, 128] | [2751, 8, 128] | 256/256 | causal | 1/√128 |
| MHA causal S512 | 8 | [2943, 8, 128] | [2943, 8, 128] | [2943, 8, 128] | 512/512 | causal | 1/√128 |
| GQA4 causal S512 | 8 | [2943, 8, 128] | [2943, 2, 128] | [2943, 8, 128] | 512/512 | causal | 1/√128 |
| MHA causal S1024 | 4 | [3327, 8, 128] | [3327, 8, 128] | [3327, 8, 128] | 1024/1024 | causal | 1/√128 |
| GQA4 causal S1024 | 4 | [3327, 8, 128] | [3327, 2, 128] | [3327, 8, 128] | 1024/1024 | causal | 1/√128 |
| MHA causal S2048 | 2 | [3073, 8, 128] | [3073, 8, 128] | [3073, 8, 128] | 2048/2048 | causal | 1/√128 |
| GQA4 causal S2048 | 2 | [3073, 8, 128] | [3073, 2, 128] | [3073, 8, 128] | 2048/2048 | causal | 1/√128 |
| MHA causal S4096 | 2 | [6145, 8, 128] | [6145, 8, 128] | [6145, 8, 128] | 4096/4096 | causal | 1/√128 |
| GQA4 causal S4096 | 2 | [6145, 8, 128] | [6145, 2, 128] | [6145, 8, 128] | 4096/4096 | causal | 1/√128 |
| MHA full S512 | 8 | [2943, 8, 128] | [2943, 8, 128] | [2943, 8, 128] | 512/512 | full | 1/√128 |
| GQA4 full S2048 | 2 | [3073, 8, 128] | [3073, 2, 128] | [3073, 8, 128] | 2048/2048 | full | 1/√128 |
| MHA full S4095 | 2 | [6143, 8, 128] | [6143, 8, 128] | [6143, 8, 128] | 4095/4095 | full | 1/√128 |
| MHA cross Q4 K4095 | 8 | [22, 8, 128] | [23543, 8, 128] | [22, 8, 128] | 4/4095 | full | 1/√128 |
| GQA4 cross Q129 K2049 | 4 | [417, 8, 128] | [6657, 2, 128] | [417, 8, 128] | 129/2049 | causal | 1/√128 |
| MHA cross Q1025 K513 | 4 | [3329, 8, 128] | [1665, 8, 128] | [3329, 8, 128] | 1025/513 | causal | 1/√128 |
| MQA8 D64 full S1024 | 4 | [3327, 8, 64] | [3327, 1, 64] | [3327, 8, 64] | 1024/1024 | full | 1/√64 |
| GQA2 D256 causal S1024 | 4 | [3327, 8, 256] | [3327, 4, 256] | [3327, 8, 256] | 1024/1024 | causal | 1/√256 |
