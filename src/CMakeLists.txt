cmake_minimum_required(VERSION 2.8 FATAL_ERROR)

project(nonrigid_descriptor)

set(OpenCV_DIR /usr/local/share/OpenCV)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS "-w") #"-Wall -Wextra")
#set(CMAKE_CXX_FLAGS_DEBUG "-g")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_CXX_FLAGS_RELEASE "-O3")
#set(CMAKE_BUILD_TYPE Debug)
find_package(PCL 1.8 REQUIRED)
find_package(OpenCV REQUIRED)

if("${OpenCV_VERSION_MAJOR}" STREQUAL "4")
  message("Using opencv 4")
  add_definitions(-DCV4)
endif()


include_directories(${PCL_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/libgeodesic/include)

# libgeodesic
include_directories("/usr/local/include/")
include_directories("/usr/include/suitesparse/")
include_directories("/usr/include")

link_directories(${PCL_LIBRARY_DIRS})

#find_package(MATLAB REQUIRED)
#link_directories("C:/Program Files/MATLAB/R2017a/extern/lib/win64/microsoft")
#include_directories("C:/Program Files/MATLAB/R2017a/extern/include")

# libgeodesic
# link_directories("/usr/lib/x86_64-linux-gnu")

include_directories(${PCL_INCLUDE_DIRS})

add_definitions(${PCL_DEFINITIONS})

include_directories(${PROJECT_SOURCE_DIR}/libgeodesic/include)

file(GLOB GEODESIC_SRC_FILES "${PROJECT_SOURCE_DIR}/libgeodesic/src/*.cpp")


file(GLOB SOURCES "${PROJECT_SOURCE_DIR}/*.cpp")

add_executable (nonrigid_descriptor ${SOURCES} ${GEODESIC_SRC_FILES})
target_link_libraries (nonrigid_descriptor 
${PCL_LIBRARIES}
${OpenCV_LIBS}
#${Matlab_LIBRARIES}
spqr 
umfpack 
cholmod 
colamd 
ccolamd 
camd 
amd 
m 
suitesparseconfig
)
