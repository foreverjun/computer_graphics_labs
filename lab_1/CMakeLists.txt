cmake_minimum_required(VERSION 3.27)
project(lab1)


set(CMAKE_CXX_STANDARD 17)
set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -static-libstdc++ -static-libgcc")
add_compile_options(-O3)

add_executable(lab1 main.cpp
        DSU.cpp
        DSU.h
        maze.cpp
        maze.h)
