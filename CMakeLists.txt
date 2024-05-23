cmake_minimum_required(VERSION 2.8.3)
project(iBoW3D)

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake_modules)

set(CMAKE_BUILD_TYPE "release")
set(CMAKE_CXX_FLAGS "-std=c++17")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")


find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Open3D REQUIRED)


include_directories(
  ${PROJECT_SOURCE_DIR}/include
  ${Open3D_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIR}
  ${catkin_INCLUDE_DIRS}  
  ${CHOLMOD_INCLUDE_DIR}
)
    

add_executable(ibow3d 
      src/Data_IO.cpp       
      src/DataBase.cpp
      src/FeatureContainer.cpp
      src/iBoW3D.cpp 
      main.cpp
      )

target_link_libraries(ibow3d ${catkin_LIBRARIES} ${CERES_LIBRARIES} ${OpenCV_LIBS} ${Open3D_LIBRARIES} ${G2O_LIBS} ${Sophus_LIBRARIES})

target_link_libraries(ibow3d
      /usr/local/lib/libtbb.so
      /usr/local/lib/libtbbmalloc.so
      /usr/local/lib/libtbbmalloc_proxy.so
      ${third_party_libs}
      )


