# Connected component of the dynamic BEV image F (paper Sec. 3.2.2).
# All metric quantities are in the fixed (registration) frame of the array header.
uint32 id                       # index of the blob inside this frame (not persistent)
geometry_msgs/Point centroid    # geometric centre of the blob pixels [m], z = 0
float64 area                    # [m^2]
float64 x_min                   # axis-aligned bounding box in the fixed frame [m]
float64 y_min
float64 x_max
float64 y_max
uint32 n_pixels
