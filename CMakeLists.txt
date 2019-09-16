cmake_minimum_required( VERSION 2.8 )
SET(PROJECT_NAME CNSLAM)
PROJECT(${PROJECT_NAME})

set( CMAKE_CXX_FLAGS "-std=c++11" )

SET(CMAKE_BUILD_TYPE "Debug") 
SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")

set(OpenCV_DIR "/data/Downloads/opencv/opencv/build/")
find_package( OpenCV )
include_directories( ${OpenCV_INCLUDE_DIRS} )

find_package(PCL 1.3 REQUIRED COMPONENTS common io)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

include_directories( "/usr/include/eigen3" )


include_directories(${PROJECT_SOURCE_DIR})


SET (MODULES fematrix
            image_rectification
            pnp
            qr
            stereomatch
            triangulation
            )
FOREACH(module ${MODULES})
    LIST(APPEND SOURCES ./module/${module}/${module}.cpp)
ENDFOREACH(module)

LINK_LIBRARIES ( ${OpenCV_LIBS} ${PCL_LIBRARIES} )

ADD_LIBRARY(${PROJECT_NAME} SHARED ${SOURCES})
# file(GLOB SRC *.cpp ../stereomatch/*.cpp ../triangulation/*.cpp)

# add_executable( imageBasics imageBasics.cpp )
# 链接OpenCV库
# target_link_libraries( imageBasics ${OpenCV_LIBS} )