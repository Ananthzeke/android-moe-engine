| metric | l32 median [min, max] (n) | l32k median [min, max] (n) |
| --- | --- | --- |
| short_total | 40.45 [40.18, 43.08] (5) | 43.13 [40.99, 45.22] (5) |
| short_prefill_s | 37.42 [37.13, 39.99] (5) | 39.67 [37.75, 41.81] (5) |
| short_decode_s | 2.894 [2.841, 2.932] (5) | 3.248 [3.073, 3.362] (5) |
| long_total | 73.88 [72.63, 74.61] (5) | 80.11 [78.85, 81.23] (5) |
| long_decode_s | 38.92 [37.83, 39.32] (5) | 41.18 [40.07, 41.97] (5) |
| long_decode_median_step_s | 1.171 [1.126, 1.197] (5) | 1.213 [1.186, 1.214] (5) |
| long_tokens_per_s | 0.7965 [0.7884, 0.8194] (5) | 0.7528 [0.7386, 0.7736] (5) |
| short_prefill_read_bytes | 5.909e+09 [5.635e+09, 5.925e+09] (5) | 5.847e+09 [5.537e+09, 5.906e+09] (5) |
| long_decode_read_bytes | 3.633e+09 [3.597e+09, 3.878e+09] (5) | 3.896e+09 [3.843e+09, 3.916e+09] (5) |
| short_prefill_majflt | 572 [532, 704] (5) | 448 [366, 696] (5) |
| long_decode_majflt | 0 [0, 0] (5) | 8 [6, 12] (5) |
| vm_hwm_kib | 4.373e+06 [4.372e+06, 4.375e+06] (5) | 4.361e+06 [4.335e+06, 4.367e+06] (5) |
| mem_available_kib_after | 3.571e+06 [3.421e+06, 3.663e+06] (5) | 3.652e+06 [3.597e+06, 3.753e+06] (5) |
| swap_in_pages | 4.947e+04 [1.006e+04, 6.434e+04] (5) | 4.836e+04 [2.306e+04, 1.047e+05] (5) |

Failed trials: none
Token-ID groups (identical lists share a group):
- long: 10 trials, ids [16, 11, 220, 17, 11, 220, 18, 11, 220, 19, 11, 220, 20, 11,...: l32#1, l32k#2, l32#3, l32k#4, l32#5, l32k#6, l32#7, l32k#8, l32#9, l32k#10
- short: 10 trials, ids [9419, 0, 248046]: l32#1, l32k#2, l32#3, l32k#4, l32#5, l32k#6, l32#7, l32k#8, l32#9, l32k#10
lmkd lines mentioning ame-server: none
