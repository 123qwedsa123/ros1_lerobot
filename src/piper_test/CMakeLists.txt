cmake_minimum_required(VERSION 3.0.2)
project(piper_test)
find_package(catkin REQUIRED COMPONENTS rospy std_msgs sensor_msgs geometry_msgs tf2_ros tf2_geometry_msgs)
catkin_package()

catkin_install_python(PROGRAMS
  #scripts/act_client_2arm3cam_deploy.py
  #scripts/gaze_slave3_joint_client.py
  scripts/realsense_publisher2.py
  scripts/record_waypoints_joint.py
  #scripts/teleop_raw_system.py
  scripts/teleop_force_feedback_node.py
  scripts/teleop_gravity_compensation_node.py
  scripts/teleop_raw_system_piperros_ff.py
  #scripts/teleop_raw_system_2m3s_3cam_gaze.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

install(DIRECTORY rviz/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/rviz
)
