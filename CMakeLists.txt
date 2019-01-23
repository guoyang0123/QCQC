cmake_minimum_required(VERSION 0.0.0)
project(qcqc)

SET(SOURCE_DIR "src/qcqc")
# Tell cmake that headers are in alse in source_dir
include_directories(${SOURCE_DIR})
SET(SOURCES ${SOURCES} 
    "${SOURCE_DIR}/math.cpp"
    "${SOURCE_DIR}/cints.cpp"
    "${SOURCE_DIR}/libint.cpp")

#link_directories("../../lib/libint/lib")
#link_libraries(int)

find_library(LIBINT NAMES int PATHS "./lib/libint/lib")
link_libraries(${LIBINT})

SET(TEST_DIR "tests")

# Generate python module
add_subdirectory(lib/pybind11)
pybind11_add_module(qcqc ${SOURCES} "${SOURCE_DIR}/bindings.cpp")
