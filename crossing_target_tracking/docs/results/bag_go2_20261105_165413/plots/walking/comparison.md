| run | valid_frames | mean_blobs_per_frame | mean_low_confirmed | mean_clusters | mean_high_tracks | high_tracks_total | high_tracks_ge_20_frames | moving_tracks_disp_ge_5.0m | static_tracks_disp_lt_1m (false-detection proxy) | fragmentation_proxy_restarts | moving_tracks_median_duration_s | moving_tracks_observed_fraction_median | moving_tracks_accel_abs_p50_median | moving_tracks_jerk_rms_median (velocity-stability proxy) | total ms mean |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| walking, odom registration (+CPD diagnostic) | 595 | 33.42 | 33.20 | 14.38 | 29.18 | 292 | 219 | 76 | 63 | 31 | 4.29 | 0.33 | 0.00 | 7.00 | 492.2 |
| walking, CPD registration (3D) | 440 | 26.24 | 20.05 | 9.97 | 18.33 | 300 | 214 | 70 | 48 | 30 | 2.82 | 0.29 | 0.00 | 5.58 | 439.6 |
| walking, CPD registration (2D BEV) | 595 | 55.90 | 63.74 | 27.12 | 50.99 | 421 | 338 | 121 | 86 | 39 | 4.24 | 0.30 | 0.00 | 6.76 | 460.8 |
| walking, NO ego-motion compensation (negative control) | 595 | 59.18 | 61.44 | 26.24 | 57.60 | 788 | 607 | 315 | 71 | 111 | 3.03 | 0.23 | 0.00 | 8.25 | 154.6 |
