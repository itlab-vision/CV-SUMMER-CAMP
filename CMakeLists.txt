cmake_minimum_required(VERSION 3.4)

set_property(GLOBAL PROPERTY USE_FOLDERS ON)
set_property(GLOBAL PROPERTY PREDEFINED_TARGETS_FOLDER "CMake")

# Create new Visual Studio solution
set(PROJECT_NAME CV_SUMMER_CAMP)
project(${PROJECT_NAME})
set(LIBRARY "${PROJECT_NAME}_lib")

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)



#Include dairectories with sources
include_directories(src)

# Move executable files 
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)

# Find OpenCV installation folder
find_package(OpenCV 4.0 REQUIRED)
set(LIBRARY_DEPS ${OpenCV_LIBS})

include_directories("${CMAKE_CURRENT_SOURCE_DIR}/include" ${OpenCV_INCLUDE_DIRS})

# Add projects in solution
add_subdirectory(src)

add_subdirectory(samples)