cmake_minimum_required(VERSION 3.4)

# Create new Visual Studio solution
project(CV_SUMMER_CAMP C CXX)

# Enable C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

# Find OpenCV installation folder
find_package(OpenCV 4.0 REQUIRED)
set(LIBRARY_DEPS ${OpenCV_LIBS})

# Move executable files 
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_RELEASE ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY_DEBUG ${CMAKE_BINARY_DIR}/bin)

# Add projects in solution
add_subdirectory(src)