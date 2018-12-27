cmake_minimum_required(VERSION 0.0.0)
project(qcqc)

SET(SOURCE_DIR "src/qcqc")
# Tell cmake that headers are in alse in source_dir
include_directories(${SOURCE_DIR})
SET(SOURCES ${SOURCES} 
    "${SOURCE_DIR}/math.cpp"
    "${SOURCE_DIR}/crys.cpp")

SET(TEST_DIR "tests")
SET(TESTS ${SOURCES}
    "${TEST_DIR}/test_main.cpp"
    "${TEST_DIR}/test_math.cpp")

# Generate a test executable
include_directories(lib/catch/include)
add_executable("${PROJECT_NAME}_test" ${TESTS})

# Generate python module
add_subdirectory(lib/pybind11)
pybind11_add_module(qcqc ${SOURCES} "${SOURCE_DIR}/bindings.cpp")
