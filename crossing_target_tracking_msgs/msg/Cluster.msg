# Position-velocity cluster of confirmed low-level tracks (paper Sec. 3.4).
uint32 id                       # index inside this frame (not persistent)
geometry_msgs/Point position    # mean member position [m]
geometry_msgs/Vector3 velocity  # mean member velocity [m/s]
float64 turn_rate               # mean member omega [rad/s]
uint32[] member_track_ids       # low-level track ids
uint32 n_members
