file(GLOB OPR_SRCS "*.cu")
file(GLOB OPR_CUDNN_SRCS "cudnn/*.cu")

target_sources(${LIBRARY_NAME} PRIVATE ${OPR_SRCS} ${OPR_CUDNN_SRCS})
target_include_directories(${LIBRARY_NAME}
    PRIVATE
    "${PROJECT_SOURCE_DIR}/src/opr")
