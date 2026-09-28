# QSA stage-one PR 91 qualification

This run qualifies the guarded QSA stage-one candidates derived from the
scheduling ideas in `halo-box/strix-llama.cpp` PR 91. The source under test is
public commit `eb039f16447fa6d55112bedceeb25c685e4f312c`, rebased on GitHub `main`
`616789845daf6efade8f364c024b38c5f94c30ec`.

The run used an Apple M5 Max with MLX
`0.32.2.dev20260919+39400a0d4`, both required GPU file locks, five warmups,
and fifty rotating interleaved repetitions per arm. MLX's normal unset/default
TF32 policy was retained. Thermal checks reported no warning level before or
after any geometry.

## Result

| Candidate | Exactness | Component result | Full-route result | State |
| --- | --- | --- | --- | --- |
| Keys-stationary MPP scorer | Exact in all FP16/BF16 and full-route cases | Median +2.23%; range +1.62% to +5.15% | Median +0.70%; range −0.44% to +0.89% | Below 3% component and route gates, unselected, default-off |
| One-workgroup top-k | Exact for distinct scores, ties, oversized threshold buckets, partial validity, padding, and full-route cases | 3.61x to 4.76x the radix-selector cost | 2.03x to 2.77x baseline | Rejected for performance, unselected, default-off |

The candidates recorded 507 keys-stationary and 505 one-pass dispatches during
qualification. These are qualification observations, not production-use
receipts. Neither candidate is selected or observed-used in production; the
ordinary MPP scorer and exact radix selector remain the defaults.

The canonical machine-readable receipt is `qualification.json` (SHA-256
`a85bdce39128ec3390ca5c7858f8e91894bcd05dfcaec8842b471812a95c1a8a`).
