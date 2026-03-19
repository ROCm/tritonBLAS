#!/usr/bin/env bash
set -e

# Remove gfx1310 blit_kernel references
sed -i -z 's/  } else if (sname == "gfx1310") {\n    \*blit_code_object = ocl_blit_object_gfx1310;//g' projects/rocr-runtime/runtime/hsa-runtime/image/blit_kernel.cpp
sed -i 's/extern uint8_t ocl_blit_object_gfx1310\[\];//g' projects/rocr-runtime/runtime/hsa-runtime/image/blit_kernel.cpp

# Remove gfx1370 blit_kernel references
sed -i -z 's/  } else if (sname == "gfx1370") {\n    \*blit_code_object = ocl_blit_object_gfx1370;//g' projects/rocr-runtime/runtime/hsa-runtime/image/blit_kernel.cpp
sed -i 's/extern uint8_t ocl_blit_object_gfx1370\[\];//g' projects/rocr-runtime/runtime/hsa-runtime/image/blit_kernel.cpp

# rocrtst CMakeLists
sed -i 's/"gfx1260;gfx1310;gfx1370"//g'  projects/rocr-runtime/rocrtst/suites/test_common/CMakeLists.txt

# blit_shaders CMakeLists
sed -i 's/;gfx1260;gfx1310//g'      projects/rocr-runtime/runtime/hsa-runtime/core/runtime/blit_shaders/CMakeLists.txt
sed -i 's/;1260;13//g'              projects/rocr-runtime/runtime/hsa-runtime/core/runtime/blit_shaders/CMakeLists.txt

# trap_handler CMakeLists
sed -i 's/;gfx1260;gfx1310//g'      projects/rocr-runtime/runtime/hsa-runtime/core/runtime/trap_handler/CMakeLists.txt
sed -i 's/;1260;13//g'              projects/rocr-runtime/runtime/hsa-runtime/core/runtime/trap_handler/CMakeLists.txt

# blit_src CMakeLists - remove both gfx1310 and gfx1370
sed -i 's/;gfx1370//g'              projects/rocr-runtime/runtime/hsa-runtime/image/blit_src/CMakeLists.txt
sed -i 's/gfx1310//g'               projects/rocr-runtime/runtime/hsa-runtime/image/blit_src/CMakeLists.txt

# amd_gpu_agent.cpp - remove gfx13 kCode references
sed -i 's/.*kCode.*\/\/ gfx13//g'   projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp
sed -i 's/.*kCode.*\/\/ gfx1260//g' projects/rocr-runtime/runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp
