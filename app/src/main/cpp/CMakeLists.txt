# For more information about how to modify this file, see https://developer.android.com/studio/build/native-dependencies#cmake
cmake_minimum_required(VERSION 3.18.1)

project("triangle_renderer")

# 添加对C++14的支持
set(CMAKE_CXX_STANDARD 14)

add_library(
    triangle_renderer
    SHARED
    triangle_renderer.cpp)

# 查找预构建库
find_library(
    log-lib
    log)

find_library(
    GLESv3-lib
    GLESv3)

# 链接库
target_link_libraries(
    triangle_renderer
    ${log-lib}
    ${GLESv3-lib}
    android)