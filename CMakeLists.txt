cmake_minimum_required(VERSION 2.8)
project (class-tracker)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -std=c++11")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} ${CMAKE_CXX_FLAGS}")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} ${CMAKE_CXX_FLAGS} -O3")

#include_directories( BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include /usr/include/python2.7 )
include_directories( BEFORE ${CMAKE_CURRENT_SOURCE_DIR}/include )

find_package(pybind11 REQUIRED)

file(GLOB class-tracker-SRC "src/*.cpp")
# set(Python_LIBRARY /usr/lib/aarch64-linux-gnu/libpython3.6m.so CACHE FILEPATH "")
# set(Python_INCLUDE_DIR /usr/include/python3.6m CACHE FILEPATH "")
# find_package(Python 3.6 REQUIRED COMPONENTS Interpreter Development)
# include_directories(BEFORE /usr/include/python3.6)
# find_package(PythonLibs 3)
include_directories(BEFORE ${PYTHON_INCLUDE_DIRS})
add_library(class-tracker SHARED ${class-tracker-SRC})
pybind11_add_module(track SHARED ${class-tracker-SRC})
target_link_libraries(class-tracker ${class-tracker-LIBS} ${PYTHON_LIBRARIES})
target_link_libraries(track PUBLIC ${class-tracker-LIBS} ${PYTHON_LIBRARIES})

#-------------------------------------------------------------------------------
# Build executables
#-------------------------------------------------------------------------------
add_executable(tracker demo/main.cpp)
target_link_libraries(tracker class-tracker)
