cmake_minimum_required(VERSION 3.7)

project(tinydl)

cmake_policy(SET CMP0079 NEW)
set(CMAKE_CXX_STANDARD 11)
enable_language(CUDA)

set(LIBRARY_NAME "_tinydl")

add_library(
    ${LIBRARY_NAME}
    SHARED
    # ${SRCS}
    )

add_subdirectory(third_party/pybind11)
add_subdirectory(src)
