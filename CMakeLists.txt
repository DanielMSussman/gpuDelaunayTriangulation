cmake_minimum_required(VERSION 3.0.0)
set(CMAKE_C_COMPILER /usr/bin/gcc-6)
set(CMAKE_CXX_COMPILER /usr/bin/g++-6)

project(gpuDelaunayTriangulation)

add_definitions(-DENABLE_CUDA)


set(CMAKE_CC_FLAGS "${CMAKE_CC_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -std=c++14 -DBOOST_MATH_DISABLE_FLOAT128 -fopenmp")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr
                                        -arch=sm_30
                                        -gencode=arch=compute_30,code=sm_30
                                        -gencode=arch=compute_50,code=sm_50
                                        -gencode=arch=compute_52,code=sm_52
                                        -Wno-deprecated-gpu-targets
                                        ")
                                        #")#-gencode arch=compute_30,code=sm_30 -Wno-deprecated-gpu-targets ")#-Xptxas -O3 -Xcompiler -O3")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")

find_package(CUDA REQUIRED)
find_package(CGAL REQUIRED)

if(${CMAKE_BUILD_TYPE} MATCHES "Debug")
    add_definitions(-DDEBUGFLAGUP)
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g -lineinfo -Xptxas --generate-line-info")
endif(${CMAKE_BUILD_TYPE} MATCHES "Debug")

add_subdirectory(src)

include_directories(${CUDA_INCLUDE_DIRS} inc src src/utilities src/model)
link_directories(${CUDA_LIBRARY_DIRS})

set(myLibs
    model
    modelGPU
    utility
    utilityGPU
    )

message("libraries = " "${myLibs}")
message("cuda include = " "${CUDA_INCLUDE_DIRS}")

set(SOURCES )
set(HEADERS
    )

# list the names of cpp files corresponding to linked executables you'd like...NON-GUI
foreach(ARG triangulation computationalTimeScaling)
add_executable("${ARG}.out" "${ARG}.cpp" )
target_link_libraries("${ARG}.out" ${myLibs} 
    ${CUDA_LIBRARIES}
    CGAL::CGAL
    )
endforeach()
