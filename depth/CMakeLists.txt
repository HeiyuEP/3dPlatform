cmake_minimum_required (VERSION 3.8.0)
project(main)

set(CMAKE_BUILD_TYPE "Debug")

#libtorch
#set(CMAKE_PREFIX_PATH /home/zhike/公共的/x30032372/libtorch)
set(Torch_DIR /home/zhike/公共的/x30032372/libtorch/share/cmake/Torch)
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGE "${CMAKE_CXX_FLAGE} ${TORCH_CXX_FLAGS}")

#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
link_directories(${OpenCV_LIBRARIES})

#depth
#依赖的头文件
include_directories(${PROJECT_SOURCE_DIR}/include)
#源文件搜索
file(GLOB SRC "${PROJECT_SOURCE_DIR}/src/*.cpp")
#设置生成库文件的路径
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)
MESSAGE("PROJECT_NAME: ${PROJECT_NAME}")
MESSAGE("SRC: ${SRC}")
MESSAGE("LIBRARY_OUTPUT_PATH: ${LIBRARY_OUTPUT_PATH}")
#生成动态库
add_library(depth SHARED ${SRC})
target_link_libraries(depth "${TORCH_LIBRARIES}")
#包含动态库路径
link_directories(${PROJECT_SOURCE_DIR}/lib)

add_executable(${PROJECT_NAME} main.cpp)
#链接动态库文件
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBRARIES})
target_link_libraries(${PROJECT_NAME} ${TORCH_LIBRARIES})
target_link_libraries(${PROJECT_NAME} depth)