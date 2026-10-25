### decode

| policy | budget GB | slots | decode slice hit rate | decode bytes read per graph (MB) |
| --- | ---: | ---: | ---: | ---: |
| store_lru | 0.57 | 960 | 36.3% | 361.0 |
| shared_layer_lru | 0.57 | 320 | 36.3% | 361.0 |
| per_layer_lru | 0.57 | 320 | 36.3% | 361.0 |
| per_layer_lfu | 0.57 | 320 | 36.3% | 361.0 |
| per_layer_lru_protect | 0.57 | 320 | 36.3% | 361.0 |
| per_layer_lfu_protect | 0.57 | 320 | 36.3% | 361.0 |
| store_lru | 1.13 | 1920 | 47.5% | 297.2 |
| shared_layer_lru | 1.13 | 640 | 47.5% | 297.0 |
| per_layer_lru | 1.13 | 640 | 50.6% | 279.8 |
| per_layer_lfu | 1.13 | 640 | 51.1% | 277.0 |
| per_layer_lru_protect | 1.13 | 640 | 50.6% | 279.8 |
| per_layer_lfu_protect | 1.13 | 640 | 51.1% | 277.0 |
| store_lru | 1.70 | 2880 | 58.8% | 233.3 |
| shared_layer_lru | 1.70 | 960 | 58.8% | 233.1 |
| per_layer_lru | 1.70 | 960 | 60.4% | 224.4 |
| per_layer_lfu | 1.70 | 960 | 59.6% | 228.7 |
| per_layer_lru_protect | 1.70 | 960 | 60.4% | 224.4 |
| per_layer_lfu_protect | 1.70 | 960 | 59.6% | 228.7 |
| store_lru | 2.26 | 3840 | 67.0% | 186.7 |
| shared_layer_lru | 2.26 | 1280 | 67.0% | 186.6 |
| per_layer_lru | 2.26 | 1280 | 67.5% | 183.8 |
| per_layer_lfu | 2.26 | 1280 | 66.9% | 187.1 |
| per_layer_lru_protect | 2.26 | 1280 | 67.5% | 183.8 |
| per_layer_lfu_protect | 2.26 | 1280 | 66.9% | 187.1 |

### prefill

| policy | budget GB | slots | prefill slice hit rate | prefill bytes read per graph (MB) |
| --- | ---: | ---: | ---: | ---: |
| store_lru | 0.57 | 960 | 0.0% | 2078.8 |
| shared_layer_lru | 0.57 | 320 | 0.0% | 2078.8 |
| per_layer_lru | 0.57 | 320 | 0.1% | 2077.0 |
| per_layer_lfu | 0.57 | 320 | 0.1% | 2077.0 |
| per_layer_lru_protect | 0.57 | 320 | 0.1% | 2077.0 |
| per_layer_lfu_protect | 0.57 | 320 | 0.1% | 2077.0 |
| store_lru | 1.13 | 1920 | 0.6% | 2066.0 |
| shared_layer_lru | 1.13 | 640 | 0.6% | 2066.0 |
| per_layer_lru | 1.13 | 640 | 0.1% | 2076.3 |
| per_layer_lfu | 1.13 | 640 | 0.1% | 2076.7 |
| per_layer_lru_protect | 1.13 | 640 | 0.1% | 2076.3 |
| per_layer_lfu_protect | 1.13 | 640 | 0.1% | 2076.7 |
| store_lru | 1.70 | 2880 | 2.3% | 2031.5 |
| shared_layer_lru | 1.70 | 960 | 2.3% | 2031.0 |
| per_layer_lru | 1.70 | 960 | 1.0% | 2057.5 |
| per_layer_lfu | 1.70 | 960 | 1.0% | 2057.5 |
| per_layer_lru_protect | 1.70 | 960 | 1.0% | 2057.5 |
| per_layer_lfu_protect | 1.70 | 960 | 1.0% | 2057.5 |
| store_lru | 2.26 | 3840 | 11.7% | 1835.2 |
| shared_layer_lru | 2.26 | 1280 | 16.1% | 1744.7 |
| per_layer_lru | 2.26 | 1280 | 14.0% | 1787.2 |
| per_layer_lfu | 2.26 | 1280 | 14.0% | 1788.2 |
| per_layer_lru_protect | 2.26 | 1280 | 14.0% | 1787.2 |
| per_layer_lfu_protect | 2.26 | 1280 | 14.0% | 1788.2 |
