patchDstPath=/home/zhike/公共的/x30032372/sunshine-codehub/sunshine-patch

cd sunshine-clean
git diff CMakeLists.txt                   > ${patchDstPath}/CMakeLists.patch
git diff cmake/compile_definitions/       > ${patchDstPath}/cmake_compile_definitions.patch
git diff cmake/targets/                   > ${patchDstPath}/cmake_targets.patch
git diff scripts/                         > ${patchDstPath}/scripts.patch
git diff src/platform/linux/              > ${patchDstPath}/src_platform_linux.patch
git diff src/video.cpp                    > ${patchDstPath}/src_video.patch