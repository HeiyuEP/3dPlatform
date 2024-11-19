latestCodePath=/home/zhike/公共的/c30041999/code/sunshine
cleanCodePath=/home/zhike/公共的/x30032372/sunshine-codehub/sunshine-clean



cp -r ${latestCodePath}/CMakeLists.txt  ${cleanCodePath}/
cp -r ${latestCodePath}/cmake/compile_definitions/linux.cmake ${cleanCodePath}/cmake/compile_definitions/
cp -r ${latestCodePath}/cmake/targets/common.cmake ${cleanCodePath}/cmake/targets/
cp -r ${latestCodePath}/scripts/linux_build.sh ${cleanCodePath}/scripts/
cp -r ${latestCodePath}/src/platform/linux/cuda.cpp ${latestCodePath}/src/platform/linux/cuda.cu ${latestCodePath}/src/platform/linux/cuda.h ${cleanCodePath}/src/platform/linux/
cp -r ${latestCodePath}/src/video.cpp ${cleanCodePath}/src/
