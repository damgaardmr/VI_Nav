# An array of predicted future poses that represents the path a robot is planning to follow.

# Indicates the frame_id of the path.
std_msgs/Header header

# Radius of robot
float32 radius

# Array of poses to follow.
geometry_msgs/PoseWithCovarianceStamped[] path