cmake_minimum_required(VERSION 3.9)
project(librealesrgan-ncnn-vulkan)

# Set policies
cmake_policy(SET CMP0091 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0091 NEW)
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")

# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Ensure position-independent code for shared libraries
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Set build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Options
option(USE_SYSTEM_NCNN "Build with system libncnn" OFF)
option(USE_SYSTEM_WEBP "Build with system libwebp" OFF)
option(USE_STATIC_MOLTENVK "Link MoltenVK static library" OFF)

# Find packages
find_package(Threads)
find_package(OpenMP)
find_package(Vulkan REQUIRED)

# Find glslangValidator
find_program(GLSLANGVALIDATOR_EXECUTABLE NAMES glslangValidator PATHS $ENV{VULKAN_SDK}/bin NO_CMAKE_FIND_ROOT_PATH)
if(NOT GLSLANGVALIDATOR_EXECUTABLE)
    message(FATAL_ERROR "glslangValidator not found. Please install Vulkan SDK.")
endif()
message(STATUS "Found glslangValidator: ${GLSLANGVALIDATOR_EXECUTABLE}")

# Shader compilation macro (unchanged)
macro(compile_shader SHADER_SRC)
    set(SHADER_SRC_FULLPATH ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER_SRC})

    get_filename_component(SHADER_SRC_NAME_WE ${SHADER_SRC} NAME_WE)
    set(SHADER_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -V -s -x -o ${SHADER_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_SPV_HEX_FILE})

    # fp16 storage
    set(SHADER_fp16s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_fp16s")

    set(SHADER_fp16s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_fp16s_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_fp16s_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -DNCNN_fp16_storage=1 -V -s -x -o ${SHADER_fp16s_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_fp16s_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_fp16s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_fp16s_SPV_HEX_FILE})

    # int8 storage
    set(SHADER_int8s_SRC_NAME_WE "${SHADER_SRC_NAME_WE}_int8s")

    set(SHADER_int8s_SPV_HEX_FILE ${CMAKE_CURRENT_BINARY_DIR}/${SHADER_int8s_SRC_NAME_WE}.spv.hex.h)
    add_custom_command(
        OUTPUT ${SHADER_int8s_SPV_HEX_FILE}
        COMMAND ${GLSLANGVALIDATOR_EXECUTABLE}
        ARGS -DNCNN_fp16_storage=1 -DNCNN_int8_storage=1 -V -s -x -o ${SHADER_int8s_SPV_HEX_FILE} ${SHADER_SRC_FULLPATH}
        DEPENDS ${SHADER_SRC_FULLPATH}
        COMMENT "Building SPIR-V module ${SHADER_int8s_SRC_NAME_WE}.spv"
        VERBATIM
    )
    set_source_files_properties(${SHADER_int8s_SPV_HEX_FILE} PROPERTIES GENERATED TRUE)
    list(APPEND SHADER_SPV_HEX_FILES ${SHADER_int8s_SPV_HEX_FILE})
endmacro()

include_directories(${CMAKE_CURRENT_BINARY_DIR})

if(OPENMP_FOUND)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

# Enable global link time optimization
cmake_policy(SET CMP0069 NEW)
set(CMAKE_POLICY_DEFAULT_CMP0069 NEW)
include(CheckIPOSupported)
check_ipo_supported(RESULT ipo_supported OUTPUT ipo_supported_output)
if(ipo_supported)
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
else()
    message(WARNING "IPO is not supported: ${ipo_supported_output}")
endif()

# NCNN library
find_package(ncnn REQUIRED)

# WebP library
find_package(WebP REQUIRED)

# Compile shaders
set(SHADER_SPV_HEX_FILES)
compile_shader(realesrgan_preproc.comp)
compile_shader(realesrgan_postproc.comp)
compile_shader(realesrgan_preproc_tta.comp)
compile_shader(realesrgan_postproc_tta.comp)
add_custom_target(generate-spirv DEPENDS ${SHADER_SPV_HEX_FILES})

# Add the shared library target
add_library(realesrgan-ncnn-vulkan SHARED realesrgan.cpp)
set_target_properties(realesrgan-ncnn-vulkan PROPERTIES OUTPUT_NAME realesrgan-ncnn-vulkan)

add_dependencies(realesrgan-ncnn-vulkan generate-spirv)

# Link libraries
target_link_libraries(realesrgan-ncnn-vulkan PUBLIC ncnn webp ${Vulkan_LIBRARY})

if(OPENMP_FOUND)
    target_link_libraries(realesrgan-ncnn-vulkan PUBLIC OpenMP::OpenMP_CXX)
endif()

if(USE_STATIC_MOLTENVK)
    find_library(CoreFoundation NAMES CoreFoundation)
    find_library(Foundation NAMES Foundation)
    find_library(Metal NAMES Metal)
    find_library(QuartzCore NAMES QuartzCore)
    find_library(CoreGraphics NAMES CoreGraphics)
    find_library(Cocoa NAMES Cocoa)
    find_library(IOKit NAMES IOKit)
    find_library(IOSurface NAMES IOSurface)

    target_link_libraries(realesrgan-ncnn-vulkan PUBLIC
        ${Metal}
        ${QuartzCore}
        ${CoreGraphics}
        ${Cocoa}
        ${IOKit}
        ${IOSurface}
        ${Foundation}
        ${CoreFoundation}
    )
endif()

# Include directories
target_include_directories(realesrgan-ncnn-vulkan PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}>
    $<INSTALL_INTERFACE:include>
)

# Ensure that the shared library uses at least C++17
set_target_properties(realesrgan-ncnn-vulkan PROPERTIES CXX_STANDARD 17 CXX_STANDARD_REQUIRED YES)

# Ensure position-independent code is used
set_target_properties(realesrgan-ncnn-vulkan PROPERTIES POSITION_INDEPENDENT_CODE ON)

# ===================== Install Rules ===================== #

# Install the library
install(TARGETS realesrgan-ncnn-vulkan
    EXPORT realesrganTargets
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    RUNTIME DESTINATION bin
)

# Install the header file
install(FILES realesrgan.h DESTINATION include)

# Install the generated shader headers
install(FILES ${SHADER_SPV_HEX_FILES} DESTINATION include)

# Install export targets
install(EXPORT realesrganTargets
    FILE realesrganTargets.cmake
    NAMESPACE realesrgan::
    DESTINATION lib/cmake/realesrgan
)

# Create and install the package configuration file
include(CMakePackageConfigHelpers)
write_basic_package_version_file(
    "${CMAKE_CURRENT_BINARY_DIR}/realesrganConfigVersion.cmake"
    VERSION 1.0.0
    COMPATIBILITY AnyNewerVersion
)

configure_package_config_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/realesrganConfig.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/realesrganConfig.cmake"
    INSTALL_DESTINATION lib/cmake/realesrgan
)

install(FILES
    "${CMAKE_CURRENT_BINARY_DIR}/realesrganConfig.cmake"
    "${CMAKE_CURRENT_BINARY_DIR}/realesrganConfigVersion.cmake"
    DESTINATION lib/cmake/realesrgan
)

