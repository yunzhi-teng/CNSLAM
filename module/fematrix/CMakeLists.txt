cmake_minimum_required( VERSION 2.8 )
project( CNSLAM )
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
set( CMAKE_CXX_FLAGS "-std=c++11" )

SET(CMAKE_BUILD_TYPE "Debug") 
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

set(OpenCV_DIR "/data/Downloads/opencv/opencv/build/")


find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )





include_directories( "/usr/include/eigen3" )

file(GLOB SRC *.cpp )


add_executable( EFmatrix ${SRC} )

target_link_libraries( EFmatrix  ${OpenCV_LIBS} )