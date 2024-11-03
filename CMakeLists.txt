cmake_minimum_required(VERSION 3.10)
project(face_system)

# 设置C++标准
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 添加编译选项
add_compile_options(-O3)  # 优化级别
if(CMAKE_COMPILER_IS_GNUCXX)
    add_compile_options(-march=native)  # 启用CPU特定优化
endif()

# OpenMP支持
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    add_compile_options(${OpenMP_CXX_FLAGS})
endif()

# 查找必要的包
find_package(OpenCV REQUIRED)
find_package(SQLite3 REQUIRED)

# NCNN路径配置
set(NCNN_DIR "${CMAKE_SOURCE_DIR}/3rdparty/ncnn" CACHE PATH "Path to NCNN installation")
find_package(ncnn REQUIRED PATHS ${NCNN_DIR})

# 包含目录
include_directories(
    ${CMAKE_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${SQLite3_INCLUDE_DIRS}
    ${NCNN_INCLUDE_DIRS}
)

# 源文件
file(GLOB SOURCES 
    "src/*.cpp"
)

# 生成可执行文件
add_executable(face_system ${SOURCES})

# 链接库
target_link_libraries(face_system
    ${OpenCV_LIBS}
    SQLite::SQLite3
    ncnn
    ${OpenMP_CXX_LIBRARIES}
)

# 安装规则
install(TARGETS face_system 
    RUNTIME DESTINATION bin
)

install(DIRECTORY models/
    DESTINATION share/face_system/models
)

# 复制模型文件到构建目录
file(COPY models/ DESTINATION ${CMAKE_BINARY_DIR}/models)
