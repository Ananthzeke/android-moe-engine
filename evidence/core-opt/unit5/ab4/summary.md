| metric | r4a median [min, max] (n) | r4n median [min, max] (n) |
| --- | --- | --- |
| short_total | 23.82 [23.47, 25.3] (5) | 23.95 [23.68, 24.26] (5) |
| short_prefill_s | 21.38 [21.09, 23.02] (5) | 21.59 [21.3, 21.81] (5) |
| short_decode_s | 2.237 [2.125, 2.29] (5) | 2.252 [2.208, 2.301] (5) |
| long_total | 51.12 [50.53, 51.5] (5) | 50.47 [50.27, 51.64] (5) |
| long_decode_s | 29.94 [29.51, 30.25] (5) | 29.46 [29.33, 30.46] (5) |
| long_decode_median_step_s | 0.9216 [0.9095, 0.9285] (5) | 0.915 [0.9104, 0.942] (5) |
| long_tokens_per_s | 1.036 [1.025, 1.05] (5) | 1.052 [1.018, 1.057] (5) |
| short_prefill_read_bytes | 5.737e+09 [5.723e+09, 5.883e+09] (5) | 5.759e+09 [5.73e+09, 5.876e+09] (5) |
| long_decode_read_bytes | 3.86e+09 [3.856e+09, 3.87e+09] (5) | 3.863e+09 [3.854e+09, 3.871e+09] (5) |
| short_prefill_majflt | 585 [572, 635] (5) | 602 [571, 704] (5) |
| long_decode_majflt | 0 [0, 0] (5) | 0 [0, 0] (5) |
| vm_hwm_kib | 4.372e+06 [4.371e+06, 4.373e+06] (5) | 4.373e+06 [4.344e+06, 4.373e+06] (5) |
| mem_available_kib_after | 3.599e+06 [3.509e+06, 3.62e+06] (5) | 3.592e+06 [3.517e+06, 3.646e+06] (5) |
| swap_in_pages | 7158 [3794, 2.407e+04] (5) | 2.222e+04 [3766, 4.372e+04] (5) |

Failed trials: none
Token-ID groups (identical lists share a group):
- long: 10 trials, ids [16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11,...: r4a#1, r4n#2, r4a#3, r4n#4, r4a#5, r4n#6, r4a#7, r4n#8, r4a#9, r4n#10
- short: 10 trials, ids [9419, 0, 248046]: r4a#1, r4n#2, r4a#3, r4n#4, r4a#5, r4n#6, r4a#7, r4n#8, r4a#9, r4n#10
lmkd lines mentioning ame-server: none
