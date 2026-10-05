cmake_minimum_required(VERSION 3.15)
project(RailPerceptionLite)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS} include)

add_executable(rail_perception src/main.cpp)
target_link_libraries(rail_perception ${OpenCV_LIBS})
