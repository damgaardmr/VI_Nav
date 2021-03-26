# An array of predicted future poses that represents the path a robot is planning to follow.

# Indicates the frame_id of the path.
std_msgs/Header header

# Radius of robot
float32 radius

float64 planning_time
float64 time_left_from_predicted_pose

geometry_msgs/PoseWithCovarianceStamped pose

# Array of action distribution parameters to follow.
float64[] a_current
float64[] a_alpha
float64[] a_beta