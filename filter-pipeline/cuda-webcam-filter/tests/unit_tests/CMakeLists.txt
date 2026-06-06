# Unit tests
add_executable(webcam_filter_tests
    test_convolution.cpp
    test_utils.cpp
    # Include implementation files needed by tests
    ${CMAKE_SOURCE_DIR}/src/utils/filter_utils.cpp
    ${CMAKE_SOURCE_DIR}/src/kernels/convolution_kernels.cu
)

# Set CUDA language for test executable if using CUDA
if(CUDA_SUPPORT)
    set_target_properties(webcam_filter_tests PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
endif()

target_include_directories(webcam_filter_tests PRIVATE
    ${CMAKE_SOURCE_DIR}/src/kernels/
    ${CMAKE_SOURCE_DIR}/src/utils/
    ${CMAKE_SOURCE_DIR}/src/input_args_parser/
    ${CMAKE_CURRENT_BINARY_DIR}/src/utils/
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(webcam_filter_tests
    GTest::gtest
    GTest::gtest_main
    ${OpenCV_LIBS}
    plog::plog
)

# Register tests
include(GoogleTest)
gtest_discover_tests(webcam_filter_tests)

# Add custom target to run the tests
add_custom_target(run_unit_tests
    COMMAND webcam_filter_tests
    DEPENDS webcam_filter_tests)

enable_testing()
