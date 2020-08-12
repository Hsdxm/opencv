cmake_minimum_required(VERSION 2.8)
project(opencv_test)

add_definitions(-std=c++11)
find_package(OpenCV REQUIRED)

add_subdirectory(dnn)

