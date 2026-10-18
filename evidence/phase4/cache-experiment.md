| slots | arena bytes | hits | reads | application bytes | read seconds | TTFT s | decode tok/s | peak VmHWM KiB |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1024 | 603979776 | 2349 | 7650 | 4,512,153,600 | 21.62 | 12.77 | 0.461 | 2,761,520 |
| 64 | 37748736 | 0 | 9999 | 5,897,650,176 | 16.62 | 9.60 | 0.561 | 2,209,768 |

Token IDs are identical. The 1024-slot cache is a measured loss: 23.5% fewer
application bytes, but 5.0 s more read time, 18% lower decode rate, and 551,752 KiB
more peak RSS. Keep demand reads and the small arena as the default.
