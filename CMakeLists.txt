# ----------------------- duplicated contents from the higher level CMakeLists ------------------------
cmake_minimum_required(VERSION 3.0)
enable_testing()
project(GaussianVI CXX C)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w")

set(gvi_subdirs 
    quadrature
    pgcsmp
    helpers 
    instances
    robots
    dynamics
)

include_directories("${CMAKE_CURRENT_SOURCE_DIR}")

add_compile_definitions("SOURCE_ROOT=${CMAKE_CURRENT_SOURCE_DIR}")

# Eigen
find_package(Eigen3 3.4.0 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIRS})

# Add all the source files in src/gvi as a separate executable
file(GLOB_RECURSE gvi_SOURCES src/*.cpp)
foreach(gvi_source ${gvi_SOURCES})
    get_filename_component(gvi_name ${gvi_source} NAME_WE)
    add_executable(${gvi_name} ${gvi_source})
    set_target_properties(${gvi_name} PROPERTIES LINKER_LANGUAGE CXX)
    set_target_properties(${gvi_name} PROPERTIES
                            RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/src"
                            )   
    target_link_libraries(${gvi_name} ${EIGEN3_LIBRARIES})
endforeach()