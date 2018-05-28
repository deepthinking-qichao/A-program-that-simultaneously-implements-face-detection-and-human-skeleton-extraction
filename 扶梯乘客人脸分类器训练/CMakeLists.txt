cmake_minimum_required(VERSION 3.5)
project(fhog_object_detector_ex)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DDLIB_JPEG_SUPPORT -DCMAKE_BUILD_TYPE=Release")
include_directories(${CMAKE_SOURCE_DIR})
include_directories(/home/scut214/dlib/) #dlib.h....
link_directories(/usr/lib/x86_64-linux-gnu/) #jpeg.so
link_directories(/home/scut214/dlib/build/dlib/) #dlib.so

set(SOURCE_FILES fhog_object_detector_ex.cpp)
add_executable(fhog_object_detector_ex ${SOURCE_FILES})
find_package(OpenCV REQUIRED)
target_link_libraries(fhog_object_detector_ex ${OpenCV_LIBS} dlib.so jpeg.so)


