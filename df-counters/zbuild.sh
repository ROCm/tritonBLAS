#!/usr/bin/env bash

# This script MUST be palced in the base directory of rocm-systems
# e.g., rocm-systems/zbuild.sh

set -e
# rm -rf build


BASE_DIR="$(dirname $(readlink -f "${BASH_SOURCE[0]}"))"
BUILD_TYPE=Release

BUILD_DIR=$BASE_DIR/build
# INSTALL_PREFIX=$BUILD_DIR/install
INSTALL_PREFIX=/opt/rocm # alternatively, $BASE_DIR/build, useful mostly only when building ONE project

PREFIX_PATH="/opt/rocm;/opt/rocm/share/rocm/cmake;/opt/rocm/lib/cmake;/opt/rocm/llvm/lib/cmake"

echo ""
echo "Base Dir:    $BASE_DIR"
echo "Build Dir:   $BUILD_DIR"
echo "Install Dir: $INSTALL_PREFIX"
echo "Prefix Path: $PREFIX_PATH"
echo "GPU devices: ${GPU_DEVICES[*]}"
echo ""

# ignore build, to prevent it from polluting git status
mkdir -p $BUILD_DIR
echo "*" > $BUILD_DIR/.gitignore

echo "------------------------------------"
echo "------- Building HSA Runtime -------"
echo "------------------------------------"
cmake       \
  -G Ninja  \
  -B $BUILD_DIR/rocr                              \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON              \
  -DHSA_DEP_ROCPROFILER_REGISTER=ON               \
  -DHSA_ENABLE_DEBUG=ON                           \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX          \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE                  \
  -DCMAKE_PREFIX_PATH=$PREFIX_PATH                \
  -DCMAKE_C_FLAGS="-w"                            \
  -DCMAKE_CXX_FLAGS="-w"                          \
  -S "$BASE_DIR/projects/rocr-runtime"

# set up clangd
ln -sf $BUILD_DIR/rocr/compile_commands.json $BASE_DIR/projects/rocr-runtime/compile_commands.json

cmake --build   $BUILD_DIR/rocr --parallel
sudo cmake --install $BUILD_DIR/rocr

echo "------------------------------------"
echo "--------- Building HIP/CLR ---------"
echo "------------------------------------"

cmake       \
  -G Ninja  \
  -B $BUILD_DIR/clr                                 \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON                \
  -DCMAKE_PREFIX_PATH=$PREFIX_PATH                  \
  -DCLR_BUILD_HIP=ON                                \
  -DHIP_COMMON_DIR="$BASE_DIR/projects/hip"         \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX            \
  -DCMAKE_BUILD_TYPE=$BUILD_TYPE                    \
  -DCMAKE_C_FLAGS="-w"                              \
  -DCMAKE_CXX_FLAGS="-w"                            \
  -S "$BASE_DIR/projects/clr"

# set up clangd
ln -sf $BUILD_DIR/clr/compile_commands.json $BASE_DIR/projects/clr/compile_commands.json

cmake --build   $BUILD_DIR/clr --parallel
sudo cmake --install $BUILD_DIR/clr

echo "------------------------------------"
echo "------- Building aqlprofile --------"
echo "------------------------------------"
cmake       \
  -G Ninja  \
  -B $BUILD_DIR/aqlprofile                          \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON                \
  -DCMAKE_PREFIX_PATH=$PREFIX_PATH                  \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX            \
  -DDEBUG_TRACE=2                                   \
  -DCMAKE_BUILD_TYPE=Debug                          \
  -S "$BASE_DIR/projects/aqlprofile"

# set up clangd
ln -sf $BUILD_DIR/aqlprofile/compile_commands.json $BASE_DIR/projects/aqlprofile/compile_commands.json

cmake --build   $BUILD_DIR/aqlprofile --parallel
sudo cmake --install $BUILD_DIR/aqlprofile

echo "------------------------------------"
echo "----- Building rocprofiler-sdk -----"
echo "------------------------------------"
cmake       \
  -G Ninja  \
  -B $BUILD_DIR/rocprofiler-sdk                     \
  -DCMAKE_EXPORT_COMPILE_COMMANDS=ON                \
  -DCMAKE_PREFIX_PATH=$PREFIX_PATH                  \
  -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX            \
  -DCMAKE_BUILD_TYPE=Debug                          \
  -DROCPROFILER_BUILD_TESTS=TRUE                    \
  -S "$BASE_DIR/projects/rocprofiler-sdk"

# set up clangd
ln -sf $BUILD_DIR/rocprofiler-sdk/compile_commands.json $BASE_DIR/projects/rocprofiler-sdk/compile_commands.json

cmake --build   $BUILD_DIR/rocprofiler-sdk --parallel
sudo cmake --install $BUILD_DIR/rocprofiler-sdk
