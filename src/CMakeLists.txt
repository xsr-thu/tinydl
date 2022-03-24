add_subdirectory(opr)

find_package(CUDAToolkit REQUIRED)

find_package(CUDAToolkit REQUIRED)

file(GLOB SRCS "*.cpp")

message("srcs: ${SRCS}")

target_sources(${LIBRARY_NAME} PRIVATE ${SRCS})

target_link_libraries(
    ${LIBRARY_NAME}
    PRIVATE
    CUDA::cudart)

target_link_libraries(
    ${LIBRARY_NAME}
    PRIVATE
    pybind11::module)

pybind11_extension(${LIBRARY_NAME})

target_include_directories(
    ${LIBRARY_NAME}
    PRIVATE
    "${PROJECT_SOURCE_DIR}/src")

message("CUDA ${CUDA_LIBRARIES}")

target_link_libraries(${LIBRARY_NAME}
    PRIVATE
    # ${CUDA_LIBRARIES}
    cudnn)
