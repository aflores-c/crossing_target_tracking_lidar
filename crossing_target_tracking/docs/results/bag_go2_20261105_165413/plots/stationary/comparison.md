| run | valid_frames | mean_blobs_per_frame | mean_low_confirmed | mean_clusters | mean_high_tracks | high_tracks_total | high_tracks_ge_20_frames | moving_tracks_disp_ge_5.0m | static_tracks_disp_lt_1m (false-detection proxy) | fragmentation_proxy_restarts | moving_tracks_median_duration_s | moving_tracks_observed_fraction_median | moving_tracks_accel_abs_p50_median | moving_tracks_jerk_rms_median (velocity-stability proxy) | total ms mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| paper (default), odom registration | 962 | 9.46 | 8.50 | 4.81 | 8.54 | 151 | 128 | 45 | 50 | 17 | 6.86 | 0.59 | 0.06 | 1.31 | 76.8 |
| ablation: shadow off | 962 | 10.01 | 9.03 | 5.07 | 9.12 | 160 | 134 | 44 | 49 | 16 | 6.61 | 0.59 | 0.06 | 1.28 | 38.9 |
| ablation: position-only clustering | 962 | 9.46 | 8.50 | 4.71 | 8.44 | 148 | 127 | 43 | 50 | 18 | 7.16 | 0.58 | 0.05 | 1.41 | 75.6 |
| ablation: raw-count binarisation | 962 | 8.93 | 7.36 | 4.69 | 7.61 | 127 | 106 | 36 | 42 | 13 | 8.17 | 0.62 | 0.05 | 1.41 | 71.3 |
| registration: stationary assumption | 962 | 12.05 | 11.09 | 6.27 | 10.81 | 183 | 159 | 41 | 65 | 21 | 6.35 | 0.61 | 0.04 | 1.23 | 76.7 |
