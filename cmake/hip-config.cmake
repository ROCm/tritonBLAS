# Fallback hip-config.cmake for environments without a full ROCm install
# (e.g. simulation environments without a full ROCm SDK).
#
# Discovers HIP headers from the triton source tree and libamdhip64.so
# from standard library search paths.  Intended to be used via
#   CMAKE_PREFIX_PATH=<tritonBLAS>/cmake
# so that origami's `find_package(hip REQUIRED)` resolves here when
# /opt/rocm is absent.
#
# Discovery order for headers:
#   1. TRITON_HIP_INCLUDE_DIR env/cmake var  (explicit override)
#   2. Triton source tree at TRITON_ROOT/third_party/amd/backend/include
#   3. Common ROCm install paths (/opt/rocm/include)
#
# Discovery order for runtime library:
#   1. HIP_LIB_DIR env/cmake var  (explicit override)
#   2. Standard library search (LD_LIBRARY_PATH, ldconfig cache, /opt/rocm/lib)

if(TARGET hip::host)
  return()
endif()

set(hip_FOUND TRUE)
set(HIP_FOUND TRUE)

# ── Discover HIP include directory ────────────────────────────────────

# Allow explicit override
if(DEFINED ENV{TRITON_HIP_INCLUDE_DIR})
  set(_HIP_INCLUDE_DIR "$ENV{TRITON_HIP_INCLUDE_DIR}")
elseif(DEFINED TRITON_HIP_INCLUDE_DIR)
  set(_HIP_INCLUDE_DIR "${TRITON_HIP_INCLUDE_DIR}")
else()
  # Try TRITON_ROOT (set by setup.py or env)
  if(DEFINED ENV{TRITON_ROOT})
    set(_TRITON_ROOT "$ENV{TRITON_ROOT}")
  elseif(DEFINED TRITON_ROOT)
    set(_TRITON_ROOT "${TRITON_ROOT}")
  else()
    set(_TRITON_ROOT "")
  endif()

  set(_HIP_INCLUDE_DIR "")

  # Search triton source tree
  if(_TRITON_ROOT AND EXISTS "${_TRITON_ROOT}/third_party/amd/backend/include/hip/hip_runtime.h")
    set(_HIP_INCLUDE_DIR "${_TRITON_ROOT}/third_party/amd/backend/include")
  else()
    # Search common paths
    find_path(_HIP_INCLUDE_DIR
      NAMES hip/hip_runtime.h
      PATHS
        /opt/rocm/include
        /usr/local/include
      NO_DEFAULT_PATH
    )
  endif()

  if(NOT _HIP_INCLUDE_DIR)
    message(FATAL_ERROR
      "hip-config.cmake: Cannot find HIP headers (hip/hip_runtime.h).\n"
      "Set TRITON_ROOT to your triton source tree or TRITON_HIP_INCLUDE_DIR "
      "to the directory containing hip/hip_runtime.h.")
  endif()
endif()

# ── Discover libamdhip64.so ───────────────────────────────────────────

if(DEFINED ENV{HIP_LIB_DIR})
  set(_HIP_LIB_DIR "$ENV{HIP_LIB_DIR}")
  set(_HIP_LIB "${_HIP_LIB_DIR}/libamdhip64.so")
elseif(DEFINED HIP_LIB_DIR)
  set(_HIP_LIB "${HIP_LIB_DIR}/libamdhip64.so")
else()
  find_library(_HIP_LIB
    NAMES amdhip64
    HINTS ENV LD_LIBRARY_PATH ENV LIBRARY_PATH
    PATHS /opt/rocm/lib /usr/local/lib
  )
  if(NOT _HIP_LIB)
    message(FATAL_ERROR
      "hip-config.cmake: Cannot find libamdhip64.so.\n"
      "Set HIP_LIB_DIR or ensure it is on LD_LIBRARY_PATH.")
  endif()
endif()

# ── Create imported targets ───────────────────────────────────────────

add_library(hip::host INTERFACE IMPORTED)
set_target_properties(hip::host PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_HIP_INCLUDE_DIR}"
  INTERFACE_COMPILE_DEFINITIONS "__HIP_PLATFORM_AMD__"
  INTERFACE_LINK_LIBRARIES "${_HIP_LIB}"
)

add_library(hip::device INTERFACE IMPORTED)
set_target_properties(hip::device PROPERTIES
  INTERFACE_INCLUDE_DIRECTORIES "${_HIP_INCLUDE_DIR}"
  INTERFACE_LINK_LIBRARIES "${_HIP_LIB}"
)

message(STATUS "hip-config.cmake (fallback): headers=${_HIP_INCLUDE_DIR}, lib=${_HIP_LIB}")
