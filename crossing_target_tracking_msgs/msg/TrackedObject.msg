# Track of the low-level (blob) or high-level (physical obstacle) CT-EKF.
# No semantic class: the method is class-agnostic by design.
uint8 LEVEL_LOW = 0
uint8 LEVEL_HIGH = 1
uint8 level

uint8 STATUS_TENTATIVE = 0      # low level only: not yet M_c/N_c confirmed
uint8 STATUS_CONFIRMED = 1      # updated with a measurement at this stamp
uint8 STATUS_PREDICTED = 2      # no measurement at this stamp (coasting until M_e/N_e deletes it)
uint8 status

uint32 id                       # persistent id (independent counters per level)
geometry_msgs/Point position    # [m] fixed frame, z = 0
geometry_msgs/Vector3 velocity  # [m/s]
float64 speed                   # |v| [m/s]
float64 heading                 # atan2(vy, vx) [rad]; valid only if heading_valid
bool heading_valid
float64 turn_rate               # omega [rad/s]
float64[25] covariance          # row-major 5x5 for state [px py vx vy omega]
float64 age                     # [s] since creation
uint32 hits
uint32 consecutive_misses
builtin_interfaces/Time first_stamp
builtin_interfaces/Time last_observed_stamp
uint32[] member_ids             # high level: low-level track ids of the last associated cluster
