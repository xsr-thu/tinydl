

file(GLOB OPR_SRCS "*.cu")

target_sources(${LIBRARY_NAME} PRIVATE ${OPR_SRCS})
target_include_directories(${LIBRARY_NAME}
    PRIVATE
    "${PROJECT_SOURCE_DIR}/src/opr")
