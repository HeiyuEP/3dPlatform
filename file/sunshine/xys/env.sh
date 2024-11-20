# 编译时的环境依赖
# Securec root
export SECUREC_DIR=/home/zhike/3DPlatformDep/libboundscheck

# Torch root
export TORCH_ROOT=/home/zhike/3DPlatformDep/libtorch

# Depth root
export DEPTH_ROOT=/home/zhike/3DPlatformDep/depth

# dibr root
export DIBR_ROOT=/home/zhike/3DPlatformDep/openDibr

# 执行时的环境依赖
# Securec lib
export LD_LIBRARY_PATH=/home/zhike/3DPlatformDep/libboundscheck/lib:$LD_LIBRARY_PATH

# DepthV2 lib
export LD_LIBRARY_PATH=/home/zhike/3DPlatformDep/depth/lib:$LD_LIBRARY_PATH

# DIBR lib
export LD_LIBRARY_PATH=/home/zhike/3DPlatformDep/openDibr/lib:$LD_LIBRARY_PATH

# DepthV2 Env
export DEPTH_MODEL_PATH=/home/zhike/3DPlatformDep/AIModel/vits_1080P.pt
export GPU_ID_NUM=1

# DIBR Env
export DIBR_BIN=/home/zhike/3DPlatformDep/openDibr/bin/RealtimeDIBR
export SUNSHINE_ASSET_DIR=/home/zhike/3DPlatformDep/openDibr/asset/
export DIBR_1K_OUPUT_JSON_PATH=/home/zhike/3DPlatformDep/openDibr/asset/example_opengl_1k.json
export DIBR_2K_OUPUT_JSON_PATH=/home/zhike/3DPlatformDep/openDibr/asset/example_opengl_2k.json
export DIBR_LEFT_OMAF_1K_INPUT_JSON=/home/zhike/3DPlatformDep/openDibr/asset/left_omaf_1k.json
export DIBR_LEFT_OMAF_2K_INPUT_JSON=/home/zhike/3DPlatformDep/openDibr/asset/left_omaf_2k.json
export DIBR_RIGHT_OMAF_1K_INPUT_JSON=/home/zhike/3DPlatformDep/openDibr/asset/right_omaf_1k.json
export DIBR_RIGHT_OMAF_2K_INPUT_JSON=/home/zhike/3DPlatformDep/openDibr/asset/right_omaf_2k.json
