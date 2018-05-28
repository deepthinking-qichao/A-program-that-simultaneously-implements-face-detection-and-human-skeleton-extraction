cmake_minimum_required(VERSION 3.5)
project(face_detection_ex)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DDLIB_JPEG_SUPPORT -DCMAKE_BUILD_TYPE=Release")

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake/Modules")

find_package(OpenCV REQUIRED)

find_package(GFlags)
find_package(Glog)
find_package(OpenPose REQUIRED)

include_directories(${CMAKE_SOURCE_DIR})
include_directories(/home/scut214/dlib/) #dlib.h....

include_directories(${OpenPose_INCLUDE_DIRS} ${GFLAGS_INCLUDE_DIR} ${GLOG_INCLUDE_DIR} ${OpenCV_INCLUDE_DIRS})

link_directories(/usr/lib/x86_64-linux-gnu/) #jpeg.so
link_directories(/home/scut214/dlib/build/dlib/) #dlib.so
link_directories(/home/scut214/openpose/3rdparty/caffe/build/lib) 

set(SOURCE_FILES face_detection_ex.cpp)
add_executable(face_detection_ex ${SOURCE_FILES})


target_link_libraries(face_detection_ex ${OpenPose_LIBS} ${GFLAGS_LIBRARY} ${GLOG_LIBRARY} ${OpenCV_LIBS} dlib.so jpeg.so)


