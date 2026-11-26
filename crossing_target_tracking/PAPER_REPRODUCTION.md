# Paper reproduction map

Reference: R. Pieroni, M. Corno, F. Parravicini, S. M. Savaresi, *Design of an automated street
crossing management module for a delivery robot*, Control Engineering Practice 153 (2024) 106095,
**Section 3 (Target tracking)**. Read from the paper PDF (not included in this repository for copyright reasons).
Sections 4–6 (crossing logic, FSM, vehicle classification) are out of scope.

**Important:** the paper publishes **no numeric value** for any target-tracking parameter
(gamma, tau_bin, W, tau_bck, kernels, area threshold, Q, R, tau_G, M/N, clustering thresholds).
Tables 2–3 contain crossing-logic parameters only. Every number in
`config/crossing_tracking.yaml` is therefore an engineering default (tag `[eng]`).

## 1. Equation / section → code → tests

| Paper | Content | Implementation | Tests |
|---|---|---|---|
| 3.1.1 | RANSAC ground removal | `core/ground.py` `GroundRemover` (batched hypotheses, tilt constraint, SVD refinement, fallbacks, height band) | `test_ground.py` (tilted plane, wall rejection, sparse fallback, height band) |
| 3.1.2 | Scan registration into a world-fixed frame; LiDAR odometry with CPD (Myronenko & Song 2010) | `core/registration.py`: `Registrar` modes `stationary / odom / cpd / auto`, `rigid_cpd`, `OdomBuffer` (interpolation, clock offset), quality gates | `test_registration.py` (transform direction, clock offset, interpolation, CPD recovers known motion, stationary check, odom jump), integration `test_8*` |
| Eq. (1)–(3) | point set P, grid B (N×M cells of size gamma, origin at the grid centre), cell counting | `core/bev_grid.py` `GridSpec.index`, `quantize` (0-based i = n−1, j = m−1; array `[j, i]` = OccupancyGrid layout) | `test_bev_grid.py::test_eq3_indexing_and_layout` |
| Eq. (4)–(5) | b̂ = b / b̄, b̄ = α(n,m)·δ | `SensorGeometry` (exact α from the 4 cell corners, 2π for the sensor cell), `binarize`; δ from the cloud layout (`points_per_ring/2π`; bag: 290.3 pts/rad) | `test_alpha_angular_coverage`, `test_normalisation_makes_detection_range_independent` |
| Eq. (6) | b̃ = 1 if b̂ > τ_bin | `binarize` | same |
| Eq. (7)–(8) | moving average of B̃ over W frames, threshold τ_bck | `core/dynamic.py` `DynamicExtractor` (O(1) ring-buffer running sums) | `test_dynamic.py::test_static_becomes_background_and_moving_is_dynamic`, `test_time_gap_resets_history` |
| 3.2.2 | dilate background before subtraction | `DynamicExtractor.process` (`k_dil`) | `test_dilated_background_removes_edge_noise` |
| Eq. (9) | f = b̃ ∧ ¬g̃ | `DynamicExtractor.process` | `test_static_becomes_background_and_moving_is_dynamic` |
| 3.2.2 | closing, area filter, connected components | `DynamicExtractor._components` (`cv2.connectedComponentsWithStats`) → `Blob` (id, centroid, area, bbox, pixels, stamp) | `test_square_closing_bridges_sparse_trace_and_area_filter` |
| 3.2.3 | shadow cones of dynamic objects; shadowed pixels of B̃ kept constant until visible | `core/shadow.py` `shadow_mask`, hold logic in `DynamicExtractor` (`self.last`) | `test_shadow_management_preserves_occluded_background`, `test_shadow_cone_geometry` |
| 3.3 | every blob is a target; its geometric centre is the measurement | `core/pipeline.py` (blob centroid → low-level tracker) | integration tests |
| Eq. (10)–(12) | CT state and transition | `core/ct_ekf.py` `ct_transition`, analytic `ct_jacobian` (Taylor branch for ω→0) | `test_ct_ekf.py` (finite-difference Jacobian incl. ω = 0, 1e‑9, 5e‑5; explicit Eq. 12; CV limit; circle; PSD) |
| Eq. (13) | H = [I₂ 0] | `H_POS` | `test_ct_ekf.py` |
| Eq. (14)–(15) | d_M = (y−ŷ)ᵀ S⁻¹ (y−ŷ), S = HPHᵀ+R; gate d_M ≤ τ_G | `batch_mahalanobis2`, `core/association.py` | `test_process_noise_psd_and_batch_mahalanobis`, `test_association.py` |
| 3.3.2 | GNN with Munkres/Hungarian, out-of-gate pairs = infinite cost | `gnn_associate` (`scipy.optimize.linear_sum_assignment`) | `test_hungarian_is_global_not_greedy`, `test_gate_rejects_and_reports_unmatched` |
| 3.3.3 | M_c/N_c confirmation, M_e/N_e elimination | `core/tracker.py` `MultiTargetTracker` | `test_track_manager.py` |
| Eq. (16)–(17) | δ_d = min distance, δ_v = max velocity difference | `core/clustering.py` `cluster_tracks` | `test_clustering.py` |
| 3.4 steps 1–4 | random seeds, assignment if both distances below thresholds | `cluster_tracks` (seeded `np.random.Generator`) | `test_every_track_in_exactly_one_cluster_and_reproducible`, `test_fixpoint_vs_single_pass` |
| 3.4 | cluster state = average of member states | `Cluster.position/velocity/omega` | `test_blobs_of_one_vehicle_merge_and_average` |
| 3.5, Eq. (18) | high-level CT-EKF, z = [x y vx vy], H₂; GNN; M/N elimination only, no confirmation | `pipeline.py` (`self.high`, `H_POS_VEL`, `use_confirmation=False`) | `test_position_velocity_update_eq18`, `test_no_confirmation_mode_high_level`, integration tests |
| 6.1 / Figs. 15–18 | trajectory, vx/vy, speed and numerically differentiated acceleration | `tools/evaluate.py` | plots in `docs/results/...` |

## 2. Deviations and interpretations (all switchable or documented)

| # | Item | Paper | Here | Why |
|---|---|---|---|---|
| D1 | Eq. (7) index range | Σ_{T=0}^{W} B̃(t−T) / W (W+1 terms) | W terms (T = 0..W−1) | Read as an index typo; keeps G ∈ [0, 1] so that "occupied in at least τ_bck·W frames" (paper's own reading) holds exactly |
| D2 | Warm-up | not described | per-cell: a cell produces dynamic pixels only after `min_observed_frames` observations; G is averaged over the frames available | Without it every pixel is "dynamic" during the first W frames; with a moving sensor, newly visible static structure would also be reported |
| D3 | Out-of-range cells | not described | cells beyond `roi_max_range` are treated like shadowed cells (value held) | Consistent with the shadow rule "kept constant until visible" |
| D4 | Shadow cone geometry | "shadow cones … for each dynamic object" | angular interval of the blob pixels (each pixel's own angular width + margin), range beyond the closest pixel, blob pixels excluded | Exact geometry not given |
| D5 | Shadow timing | not described | cones from the blobs of the current frame (detected with the background including the current, not yet corrected, frame); the corrected frame is then pushed into the history | Causal single pass; the current frame contributes 1/W |
| D6 | Closing element | "closure filter" | square k×k | An elliptical element cannot bridge gaps in 1-pixel-wide LiDAR traces (`test_square_closing…`) |
| D7 | Clustering step 2 | one sweep over unclustered tracks | repeated until no track joins (`growth: fixpoint`); `single_pass` available | Order independence along chains of blobs of one long vehicle |
| D8 | Clustering input | "only confirmed tracks" | confirmed **and** updated at this instant (`include_predicted: false`) | Coasting tracks would feed predictions back as measurements; the paper notes clusters are subject to shadowing/misdetection, handled by high-level M/N |
| D9 | Registration | CPD LiDAR odometry; robot standing still in front of the crossing | default `auto` = robot odometry (clock-offset corrected); CPD selectable (`mode: cpd`) and as a diagnostic; `stationary` mode with an odometry check | User choice. Measured on synthetic data: our rigid CPD drifts (see RESULTS.md) |
| D10 | Ground removal | RANSAC | RANSAC + candidate band around a height prior + tilt constraint + temporal fallback + height band [thr, 3 m] | Robustness (walls / car sides must not win the plane vote); canopies above 3 m do not create occupancy |
| D11 | Process / measurement noise, initialisation | not given | white-acceleration CT noise (σ_a, σ_ω), R = σ_r² I; new tracks start with zero velocity and large σ_v; high-level R₂ = diag(σ_p², σ_p², σ_v², σ_v²) | Standard choices |
| D12 | Instants for M/N | "instants of time" | processed scans; dropped scans (time gap > 1.5·nominal dt) and invalid-registration frames count as instants without measurement | Keeps M/N tied to time |
| D13 | ω clamp | – | |ω| ≤ 2 rad/s | numerical safeguard |

Ablation baselines (not the paper, labelled as such): `grid.normalization: raw_count`, `shadow.enabled: false`,
`clustering.use_velocity: false` (position-only).

## 3. Assumptions that are NOT verified

* Ground truth is not available for the Go2 bag: no accuracy, precision/recall or ID-switch numbers are given for real data. (Synthetic data have exact ground truth.)
* The extrinsic base_link → velodyne_go2 (0.055, 0, 0.39), no rotation, is hand-measured (from `~/go2_bag_ws/launch/go2_tf_offline.launch.py`); a rotation error would bias the registered positions while walking.
* The odometry clock offset is estimated from receive times (median, ≈ −2.73 s); its residual error (few ms) is not measured independently.
* Odometry accuracy (`/utlidar/robot_odom`) during walking is unknown; CPD-vs-odom disagreement is reported only as a diagnostic.
* Per-point motion during a 0.1 s sweep (deskew) is neglected (the `time` field exists but is not used); at Go2 walking speed (< 1 m/s) the error is < 0.1 m.
* Eq. (5) uses one layer, so b̄ underestimates the true maximum near the sensor (paper acknowledges); τ_bin was chosen with that in mind.
* Default parameters were chosen on synthetic scenes; their optimality for the real crossing is not established.
* Semantics: none by design. "Pedestrian"/"car" words in docs refer to synthetic ground truth only.
