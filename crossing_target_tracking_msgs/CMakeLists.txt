cmake_minimum_required(VERSION 3.8)
project(crossing_target_tracking_msgs)

find_package(ament_cmake REQUIRED)
find_package(std_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(builtin_interfaces REQUIRED)
find_package(rosidl_default_generators REQUIRED)

rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/Blob.msg"
  "msg/BlobArray.msg"
  "msg/Cluster.msg"
  "msg/ClusterArray.msg"
  "msg/TrackedObject.msg"
  "msg/TrackedObjectArray.msg"
  DEPENDENCIES std_msgs geometry_msgs builtin_interfaces
)

ament_export_dependencies(rosidl_default_runtime)
ament_package()
