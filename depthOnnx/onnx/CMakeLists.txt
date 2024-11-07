cmake_minimum_required (VERSION 3.8.0)
project(main)

#set(CMAKE_CXX_STANDARD 14)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE "Debug")

include_directories("/home/zhike/公共的/x30032372/onnxruntime-linux-x64-gpu-1.16.3/include")
link_directories("/home/zhike/公共的/x30032372/onnxruntime-linux-x64-gpu-1.16.3/lib")
link_libraries(onnxruntime)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES})

add_executable(${PROJECT_NAME} main.cpp)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})