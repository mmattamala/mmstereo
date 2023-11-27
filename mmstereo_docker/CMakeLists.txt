cmake_minimum_required(VERSION 3.1)
project(mmstereo_docker)


set(PACKAGE_DEPENDENCIES
  mmstereo_ros
)

find_package(catkin REQUIRED 
	COMPONENTS
	${PACKAGE_DEPENDENCIES}
)

catkin_package(
  CATKIN_DEPENDS
    ${PACKAGE_DEPENDENCIES}
)

# # Python Setup
# catkin_python_setup()
# catkin_install_python(PROGRAMS scripts/mmstereo_node.py
#                       DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )
