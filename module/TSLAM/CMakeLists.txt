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

find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})



include_directories( "/usr/include/eigen3" )

file(GLOB SRC src.cpp Tslam.cpp)


add_executable( TSLAM ${SRC} )

target_link_libraries( TSLAM ${CERES_LIBRARIES} ${OpenCV_LIBS} pthread ${PCL_LIBRARIES} )