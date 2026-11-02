cmake_minimum_required(VERSION 3.20)
project(chroma-chord VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Find LibTorch                                                
find_package(Torch REQUIRED)                                             
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")             

if(MSVC)
    add_compile_options(/W4 /permissive-)
else()
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# Include directories
include_directories(include)

# Source files
file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS "src/*.cpp")

# Main executable
add_executable(${PROJECT_NAME} ${SOURCES})
target_include_directories(${PROJECT_NAME} PUBLIC include)

# Link LibTorch libraries
target_link_libraries(${PROJECT_NAME} PRIVATE "${TORCH_LIBRARIES}")
