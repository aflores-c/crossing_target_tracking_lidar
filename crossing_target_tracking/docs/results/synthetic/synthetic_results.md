| variant | scenario | GT obj | HL ids per GT obj | ID sw | false HL ids | precision | recall | pos RMSE [m] | vel RMSE [m/s] | ms/frame |
|---|---|---|---|---|---|---|---|---|---|---|
| paper (default) | s1_static | 0 | – | 0 | 0 | – | – | – | – | 32.7 |
| paper (default) | s2_constant_speed | 1 | car:1 | 0 | 3 | 0.79 | 1.00 | 1.17 | 1.04 | 42.3 |
| paper (default) | s3_turning | 1 | car:3 | 4 | 3 | 0.79 | 0.73 | 1.74 | 1.75 | 61.6 |
| paper (default) | s4_two_vehicles | 2 | car:3, motorcycle:2 | 3 | 2 | 0.85 | 0.66 | 1.14 | 1.72 | 65.9 |
| paper (default) | s4b_two_pedestrians | 2 | ped_a:1, ped_b:1 | 0 | 0 | 1.00 | 0.93 | 0.21 | 0.08 | 65.8 |
| paper (default) | s5_occlusion | 1 | cyclist:2 | 1 | 0 | 1.00 | 0.73 | 0.50 | 0.64 | 55.6 |
| paper (default) | s6_multi | 4 | cyclist:1, pedestrian:1, car_east:2, car_west:2 | 2 | 6 | 0.92 | 0.82 | 1.14 | 1.37 | 55.2 |
| paper (default) | s7_background_change | 2 | parking_car:3, new_box:1 | 2 | 2 | 0.86 | 0.21 | 0.99 | 1.56 | 42.5 |
| paper (default) | s9_shadow | 1 | bus:3 | 4 | 4 | 0.54 | 0.87 | 2.10 | 1.12 | 58.3 |
| paper (default) | s8_moving_sensor | 1 | car:1 | 0 | 3 | 0.61 | 1.00 | 1.06 | 1.15 | 133.5 |
| ablation: shadow off | s1_static | 0 | – | 0 | 0 | – | – | – | – | 40.8 |
| ablation: shadow off | s2_constant_speed | 1 | car:1 | 0 | 3 | 0.79 | 1.00 | 1.17 | 1.04 | 42.7 |
| ablation: shadow off | s3_turning | 1 | car:3 | 4 | 3 | 0.79 | 0.73 | 1.74 | 1.75 | 51.1 |
| ablation: shadow off | s4_two_vehicles | 2 | car:3, motorcycle:2 | 3 | 3 | 0.85 | 0.66 | 1.15 | 1.74 | 51.5 |
| ablation: shadow off | s4b_two_pedestrians | 2 | ped_a:1, ped_b:1 | 0 | 0 | 1.00 | 0.93 | 0.21 | 0.08 | 44.1 |
| ablation: shadow off | s5_occlusion | 1 | cyclist:2 | 1 | 0 | 1.00 | 0.73 | 0.50 | 0.64 | 36.5 |
| ablation: shadow off | s6_multi | 4 | cyclist:1, pedestrian:1, car_east:2, car_west:2 | 2 | 6 | 0.92 | 0.82 | 1.14 | 1.37 | 46.9 |
| ablation: shadow off | s7_background_change | 2 | parking_car:3, new_box:1 | 2 | 2 | 0.86 | 0.21 | 0.99 | 1.56 | 44.4 |
| ablation: shadow off | s9_shadow | 1 | bus:3 | 4 | 4 | 0.54 | 0.87 | 2.10 | 1.12 | 33.5 |
| ablation: shadow off | s8_moving_sensor | 1 | car:1 | 0 | 3 | 0.61 | 1.00 | 1.06 | 1.15 | 104.0 |
| ablation: position-only clustering | s1_static | 0 | – | 0 | 0 | – | – | – | – | 29.6 |
| ablation: position-only clustering | s2_constant_speed | 1 | car:1 | 0 | 3 | 0.79 | 1.00 | 1.17 | 1.04 | 42.3 |
| ablation: position-only clustering | s3_turning | 1 | car:3 | 4 | 3 | 0.79 | 0.73 | 1.74 | 1.75 | 61.9 |
| ablation: position-only clustering | s4_two_vehicles | 2 | car:2, motorcycle:2 | 2 | 1 | 0.87 | 0.66 | 1.24 | 1.34 | 59.0 |
| ablation: position-only clustering | s4b_two_pedestrians | 2 | ped_a:2, ped_b:1 | 2 | 0 | 1.00 | 0.81 | 0.44 | 0.43 | 58.0 |
| ablation: position-only clustering | s5_occlusion | 1 | cyclist:2 | 1 | 0 | 1.00 | 0.73 | 0.50 | 0.64 | 42.3 |
| ablation: position-only clustering | s6_multi | 4 | cyclist:1, pedestrian:1, car_east:3, car_west:1 | 3 | 6 | 0.92 | 0.81 | 1.16 | 1.46 | 76.0 |
| ablation: position-only clustering | s7_background_change | 2 | parking_car:3, new_box:1 | 2 | 2 | 0.86 | 0.21 | 0.99 | 1.56 | 36.5 |
| ablation: position-only clustering | s9_shadow | 1 | bus:3 | 4 | 4 | 0.54 | 0.87 | 2.10 | 1.12 | 45.7 |
| ablation: position-only clustering | s8_moving_sensor | 1 | car:1 | 0 | 3 | 0.61 | 1.00 | 1.06 | 1.15 | 98.9 |
| ablation: raw-count binarisation | s1_static | 0 | – | 0 | 0 | – | – | – | – | 41.4 |
| ablation: raw-count binarisation | s2_constant_speed | 1 | car:2 | 1 | 2 | 0.78 | 1.00 | 1.18 | 1.49 | 54.2 |
| ablation: raw-count binarisation | s3_turning | 1 | car:3 | 2 | 2 | 0.79 | 0.84 | 1.27 | 1.18 | 56.9 |
| ablation: raw-count binarisation | s4_two_vehicles | 2 | car:2, motorcycle:1 | 1 | 1 | 0.86 | 0.70 | 1.03 | 1.38 | 52.5 |
| ablation: raw-count binarisation | s4b_two_pedestrians | 2 | ped_a:1, ped_b:1 | 0 | 0 | 1.00 | 0.93 | 0.20 | 0.09 | 42.3 |
| ablation: raw-count binarisation | s5_occlusion | 1 | cyclist:2 | 1 | 0 | 1.00 | 0.73 | 0.54 | 0.69 | 42.4 |
| ablation: raw-count binarisation | s6_multi | 4 | cyclist:1, pedestrian:1, car_east:2, car_west:2 | 2 | 3 | 0.91 | 0.83 | 0.99 | 1.12 | 51.1 |
| ablation: raw-count binarisation | s7_background_change | 2 | parking_car:3, new_box:1 | 2 | 2 | 0.86 | 0.22 | 0.95 | 1.53 | 21.7 |
| ablation: raw-count binarisation | s9_shadow | 1 | bus:3 | 2 | 4 | 0.53 | 0.87 | 2.09 | 1.12 | 37.1 |
| ablation: raw-count binarisation | s8_moving_sensor | 1 | car:1 | 0 | 2 | 0.74 | 1.00 | 1.01 | 1.08 | 59.7 |

Scenarios:
* `s1_static`: Static scene, no moving objects
* `s2_constant_speed`: One car at 10 m/s on a straight lane (y = 8 m)
* `s3_turning`: One car: straight at 7 m/s, then a right turn (omega = -0.3 rad/s, R = 23 m)
* `s4_two_vehicles`: Car (8 m/s) and motorcycle (12 m/s) on adjacent lanes, overtaking
* `s4b_two_pedestrians`: Two pedestrians passing each other 0.8 m apart (+-1.4 m/s)
* `s5_occlusion`: Cyclist (5 m/s, y = 10 m) passes behind a parked van (static occluder at y = 4 m)
* `s6_multi`: Four road users crossing the field of view at different times
* `s7_background_change`: Car drives in and parks; a new static box appears; static clutter
* `s9_shadow`: Slow bus (12 m, 3 m/s) on the near lane occludes the facade for > (1 - tau_bck) W frames
* `s8_moving_sensor`: Sensor walks at 0.8 m/s while turning (0.08 rad/s); one car at 9 m/s; odom given
