if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

include_directories(${CMAKE_SOURCE_DIR}/inc ${CUDA_INCLUDE_DIRS})
add_library(utility
    noiseSource.cpp
    cellListGPU.cpp
    hilbert_curve.cpp
    )
cuda_add_library(utilityGPU
    noiseSource.cu
    cellListGPU.cu
    utilities.cu
    )
