cmake_minimum_required( VERSION 2.8 )
project( CNSLAM )
# add_definitions(-D_GLIBCXX_USE_CXX11_ABI=1)
set( CMAKE_CXX_FLAGS "-std=c++11" )

SET(CMAKE_BUILD_TYPE "Debug") 
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")




include_directories( "/usr/include/eigen3" )

file(GLOB SRC *.cpp)


add_executable( QR_test ${SRC} )

target_link_libraries( QR_test )