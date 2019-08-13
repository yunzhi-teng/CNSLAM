cmake_minimum_required( VERSION 2.8 )
project( CNSLAM )
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
set( CMAKE_CXX_FLAGS "-std=c++11" )

SET(CMAKE_BUILD_TYPE "Debug") 
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

set(OpenCV_DIR "/data/Downloads/opencv/opencv/build/")


find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package( Ceres REQUIRED )
include_directories( ${CERES_INCLUDE_DIRS} )

include_directories( "/usr/include/eigen3" )

file(GLOB SRC src.cpp)


add_executable( LVO ${SRC} )
# add_executable(test test.cpp )

target_link_libraries( LVO ${CERES_LIBRARIES} ${OpenCV_LIBS} pthread )
# target_link_libraries( test ${CERES_LIBRARIES} ${OpenCV_LIBS} pthread )