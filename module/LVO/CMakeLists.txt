cmake_minimum_required( VERSION 2.8 )
project( CNSLAM )

# 添加c++ 11标准支持
set( CMAKE_CXX_FLAGS "-std=c++11" )

set(OpenCV_DIR "/data/Downloads/opencv/opencv/build/")
# 寻找OpenCV库
find_package( OpenCV )

file(GLOB SRC *.cpp)
# 添加头文件
include_directories( ${OpenCV_INCLUDE_DIRS} )

add_executable( LVO ${SRC} )
# 链接OpenCV库
target_link_libraries( LVO ${OpenCV_LIBS} )