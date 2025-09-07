# For more information about how to modify this file, see https://developer.android.com/studio/build/native-dependencies#cmake
cmake_minimum_required(VERSION 3.18.1)

project("triangle_renderer")

# 添加对C++14的支持
set(CMAKE_CXX_STANDARD 14)

# stb_image是单头文件库，不需要查找外部依赖
# 只需要确保stb_image.h在包含路径中

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

# 设置包含目录
target_include_directories(triangle_renderer PRIVATE 
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# 链接基础库（stb_image不需要链接）
target_link_libraries(
    triangle_renderer
    ${log-lib}
    ${GLESv3-lib}
    android)
