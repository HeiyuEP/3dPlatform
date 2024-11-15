cp sunshine-patch/* verify/sunshine/
cd verify/sunshine/
patch -p1 < cmake_compile_definitions.patch
patch -p1 < cmake_targets.patch
patch -p1 < CMakeLists.patch
patch -p1 < scripts.patch
patch -p1 < src_platform_linux.patch
patch -p1 < src_video.patch