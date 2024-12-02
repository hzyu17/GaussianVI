# ----------------------- duplicated contents from the higher level CMakeLists ------------------------
cmake_minimum_required(VERSION 3.0)
enable_testing()
project(GaussianVI CXX C CUDA)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

set(gvi_subdirs 
    quadrature 
    ngd
    proxgd
    gvibase
)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_compile_definitions("SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}")
# add_compile_definitions(GVI_SUBDUR_ENV=0)

# Set the policy CMP0104 to NEW
if(POLICY CMP0104)
  cmake_policy(SET CMP0104 NEW)
endif()

# Set the CUDA architectures you want to target
set(CMAKE_CUDA_ARCHITECTURES "52;60;61;70;75;80")

# Cuda
find_package(CUDA REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
set(CUDA_ARCH "sm_86") #Change it based on the GPU
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}; -arch=${CUDA_ARCH}; -std=c++17; --expt-extended-lambda)

if (CMAKE_CUDA_COMPILER)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
endif()


# Specify the CUDA standard
set(CMAKE_CUDA_STANDARD 14)

# Eigen
find_package(Eigen3 3.4.0 REQUIRED)
add_definitions(-DEIGEN_DONT_VECTORIZE)

# # Boost
# find_package(Boost 1.70 REQUIRED)
# find_package(Boost COMPONENTS serialization REQUIRED)
# include_directories(${Boost_INCLUDE_DIR})
# link_directories(${Boost_LIBRARY_DIRS})

# OpenMP
find_package(OpenMP)

# Cereal
set(CEREAL_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/include/cereal/include")
include_directories(${CEREAL_INCLUDE_DIR})

if(DEFINED libSpGH_LIBRARY AND NOT "${libSpGH_LIBRARY}" STREQUAL "")
    message("Variable libSpGH_LIBRARY is defined and not empty.")
    # Your code for the case when the variable is defined and not empty
else()
    message("Variable libSpGH_LIBRARY is either NOT defined or empty.")
    # Find Matlab
    message("Finding Matlab Library")
    # set(MATLAB_ROOT_DIR "~/.local/MATLAB/R2020b")
    set(MATLAB_ROOT_DIR "/usr/local/MATLAB/R2020b")
    set(Matlab_INCLUDE_DIRS "${MATLAB_ROOT_DIR}/extern/include")
    set(Matlab_MEX_LIBRARY "${MATLAB_ROOT_DIR}/bin/glnxa64/libmex.so")  # Adjust the path accordingly
    set(MCR_LIBRARY_PATH "${MATLAB_ROOT_DIR}/runtime/glnxa64") 
    # find_library(Matlab_MX_LIBRARY NAMES libmx.so PATHS "{MATLAB_ROOT_DIR}/bin/glnxa64")
    set(libSpGH_LIBRARY "${CMAKE_CURRENT_SOURCE_DIR}/quadrature/libSpGH/for_testing/libSpGH.so")


    # Your code for the case when the variable is either NOT defined or empty
endif()

include_directories(${EIGEN3_INCLUDE_DIRS})
include_directories("${CMAKE_CURRENT_BINARY_DIR}/quadrature/libSpGH/for_testing")

# Add all the source files in src/gvi as a separate executable
file(GLOB_RECURSE gvi_SOURCES src/*.cpp)
foreach(gvi_source ${gvi_SOURCES})
    get_filename_component(gvi_name ${gvi_source} NAME_WE)
    add_executable(${gvi_name} ${gvi_source})
    set_target_properties(${gvi_name} PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(${gvi_name} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src"
                            )   
    target_link_libraries(${gvi_name} ${EIGEN3_LIBRARIES} ${Matlab_MEX_LIBRARY} ${CUDA_LIBRARIES} OpenMP::OpenMP_CXX 
                        #   ${Matlab_MX_LIBRARY} 
                          ${libSpGH_LIBRARY} ${MCR_LIBRARY_PATH}/libmwmclmcrrt.so.9.9)
    target_include_directories(${gvi_name} PRIVATE ${Matlab_INCLUDE_DIRS})
endforeach()

# include(FetchContent)
# FetchContent_Declare(
#   googletest
#   URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip
# )
# # For Windows: Prevent overriding the parent project's compiler/linker settings
# set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
# FetchContent_MakeAvailable(googletest)
# enable_testing()

# include(GoogleTest)

# file(GLOB_RECURSE test_SOURCES tests/*.cpp)
# foreach(test_source ${test_SOURCES})
#     get_filename_component(test_name ${test_source} NAME_WE)
#     add_executable(${test_name} ${test_source} helpers/MatrixMultiplication.cu)
#     set_target_properties(${test_name} PROPERTIES
#                             RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/tests"
#                             )  
#     target_link_libraries( ${test_name} GTest::gtest_main ${Matlab_MEX_LIBRARY} 
#     # ${Matlab_MX_LIBRARY} 
#                            ${libSpGH_LIBRARY} ${MCR_LIBRARY_PATH}/libmwmclmcrrt.so.9.9 Boost::serialization)
#     gtest_discover_tests(${test_name})
#     target_include_directories(${test_name} PRIVATE ${Matlab_INCLUDE_DIRS})

# endforeach()
