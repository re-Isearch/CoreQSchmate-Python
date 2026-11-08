cmake_minimum_required(VERSION 3.24)
project(CoreQSchmatePython VERSION 0.1.0 LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(Python 3.10 REQUIRED COMPONENTS Interpreter Development.Module)
find_package(nanobind CONFIG REQUIRED)

set(COREQSCHMATE_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../Schmate" CACHE PATH "Current re-Isearch/Schmate checkout")
set(COREQSCHMATE_BERT_SOURCE_DIR "${COREQSCHMATE_SOURCE_DIR}/bert.cpp" CACHE PATH "Current re-Isearch/bert.cpp checkout")
set(COREQSCHMATE_GGML_SOURCE_DIR "${COREQSCHMATE_BERT_SOURCE_DIR}/ggml" CACHE PATH "ggml-org/ggml checkout")
foreach(_dependency IN ITEMS COREQSCHMATE_SOURCE_DIR COREQSCHMATE_BERT_SOURCE_DIR COREQSCHMATE_GGML_SOURCE_DIR)
    get_filename_component(${_dependency} "${${_dependency}}" REALPATH)
endforeach()
if(NOT EXISTS "${COREQSCHMATE_SOURCE_DIR}/include/BertIndexManager.hpp")
    message(FATAL_ERROR "COREQSCHMATE_SOURCE_DIR must name the standalone Schmate checkout")
endif()
if(NOT EXISTS "${COREQSCHMATE_BERT_SOURCE_DIR}/bert.cpp")
    message(FATAL_ERROR "Set COREQSCHMATE_BERT_SOURCE_DIR to re-Isearch/bert.cpp")
endif()
if(NOT EXISTS "${COREQSCHMATE_GGML_SOURCE_DIR}/include/ggml.h")
    message(FATAL_ERROR "Set COREQSCHMATE_GGML_SOURCE_DIR to ggml-org/ggml")
endif()

# Build the standalone core only. Schmate's top-level project also creates the
# optional IB bridge and command-line programs; neither belongs in this module.
set(BUILD_SHARED_LIBS OFF CACHE BOOL "Statically link native dependencies")
set(GGML_BUILD_TESTS OFF CACHE BOOL "Do not build GGML tests")
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "Do not build GGML examples")
set(GGML_NATIVE OFF CACHE BOOL "Disable host-specific march=native")
set(GGML_BACKEND_DL OFF CACHE BOOL "Link GGML backends into the extension")
option(GGML_METAL "Enable Apple's Metal backend" ${APPLE})
option(GGML_ACCELERATE "Enable Apple's Accelerate backend" ${APPLE})
add_subdirectory("${COREQSCHMATE_GGML_SOURCE_DIR}" "ggml" EXCLUDE_FROM_ALL)

add_library(schmate_bert STATIC "${COREQSCHMATE_BERT_SOURCE_DIR}/bert.cpp")
target_include_directories(schmate_bert PUBLIC "${COREQSCHMATE_BERT_SOURCE_DIR}")
target_link_libraries(schmate_bert PUBLIC ggml)
foreach(_backend IN ITEMS CUDA METAL VULKAN)
    if(GGML_${_backend})
        target_compile_definitions(schmate_bert PUBLIC GGML_USE_${_backend}=1)
    endif()
endforeach()
if(APPLE AND GGML_ACCELERATE)
    target_compile_definitions(schmate_bert PUBLIC GGML_USE_ACCELERATE=1)
endif()

set(_core_sources BertIndex HnswConfig ShardedIndex BertIndexManager Util OffsetFile
    Logger FileLock SBertGGML hf_mapper StringEmbedding FileStreamCache unified_hnsw)
set(_core_paths "")
foreach(_source IN LISTS _core_sources)
    if(_source MATCHES "^(Logger|ShardedIndex|BertIndex|OffsetFile)$")
        list(APPEND _core_paths "${CMAKE_CURRENT_BINARY_DIR}/native-src/${_source}.cpp")
    else()
        list(APPEND _core_paths "${COREQSCHMATE_SOURCE_DIR}/src/${_source}.cpp")
    endif()
endforeach()
file(GLOB_RECURSE _native_headers CONFIGURE_DEPENDS "${COREQSCHMATE_SOURCE_DIR}/include/*")
set_property(DIRECTORY APPEND PROPERTY CMAKE_CONFIGURE_DEPENDS ${_native_headers}
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare_native.py"
    "${CMAKE_CURRENT_SOURCE_DIR}/bindings/merge_serial.inc"
    "${COREQSCHMATE_SOURCE_DIR}/src/Logger.cpp"
    "${COREQSCHMATE_SOURCE_DIR}/src/ShardedIndex.cpp"
    "${COREQSCHMATE_SOURCE_DIR}/src/BertIndex.cpp"
    "${COREQSCHMATE_SOURCE_DIR}/src/OffsetFile.cpp")
execute_process(COMMAND "${Python_EXECUTABLE}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/prepare_native.py"
    "${COREQSCHMATE_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}/native-include"
    COMMAND_ERROR_IS_FATAL ANY)
add_library(schmate_standalone STATIC ${_core_paths} bindings/standalone.cxx)
target_include_directories(schmate_standalone PUBLIC "${CMAKE_CURRENT_BINARY_DIR}/native-include")
target_compile_options(schmate_standalone PUBLIC
    "-include" "${CMAKE_CURRENT_SOURCE_DIR}/bindings/native_compat.hpp")
target_link_libraries(schmate_standalone PUBLIC schmate_bert)
find_package(Threads REQUIRED)
target_link_libraries(schmate_standalone PUBLIC Threads::Threads)
if(APPLE AND GGML_ACCELERATE)
    target_link_libraries(schmate_standalone PUBLIC "-framework Accelerate")
endif()

nanobind_add_module(_core NB_STATIC bindings/core.cxx)
target_link_libraries(_core PRIVATE schmate_standalone)
foreach(_pair IN ITEMS "schmate;${COREQSCHMATE_SOURCE_DIR}" "bert;${COREQSCHMATE_BERT_SOURCE_DIR}" "ggml;${COREQSCHMATE_GGML_SOURCE_DIR}")
    list(GET _pair 0 _name)
    list(GET _pair 1 _directory)
    execute_process(COMMAND git -C "${_directory}" rev-parse HEAD
        OUTPUT_VARIABLE _revision OUTPUT_STRIP_TRAILING_WHITESPACE ERROR_QUIET)
    target_compile_definitions(_core PRIVATE "COREQSCHMATE_${_name}_REVISION=\"${_revision}\"")
endforeach()
install(FILES LICENSE DESTINATION schmate/licenses RENAME CoreQSchmate-Python.txt COMPONENT python)
install(FILES "${COREQSCHMATE_SOURCE_DIR}/include/hnswlib/LICENSE"
    DESTINATION schmate/licenses RENAME hnswlib.txt COMPONENT python)
install(DIRECTORY licenses/ DESTINATION schmate/licenses COMPONENT python)

set(_package "${CMAKE_CURRENT_BINARY_DIR}/schmate")
set_target_properties(_core PROPERTIES LIBRARY_OUTPUT_DIRECTORY "${_package}" RUNTIME_OUTPUT_DIRECTORY "${_package}")
foreach(_config IN LISTS CMAKE_CONFIGURATION_TYPES)
    string(TOUPPER "${_config}" _upper)
    set_target_properties(_core PROPERTIES "LIBRARY_OUTPUT_DIRECTORY_${_upper}" "${_package}" "RUNTIME_OUTPUT_DIRECTORY_${_upper}" "${_package}")
endforeach()
add_custom_target(schmate_python_sources ALL
    COMMAND "${CMAKE_COMMAND}" -E copy_directory "${CMAKE_CURRENT_SOURCE_DIR}/src/schmate" "${_package}")
add_dependencies(_core schmate_python_sources)
install(TARGETS _core LIBRARY DESTINATION schmate COMPONENT python RUNTIME DESTINATION schmate COMPONENT python)
install(DIRECTORY src/schmate/ DESTINATION schmate COMPONENT python
    FILES_MATCHING PATTERN "*.py" PATTERN "*.pyi" PATTERN "py.typed" PATTERN "__pycache__" EXCLUDE)
foreach(_pair IN ITEMS "Schmate;${COREQSCHMATE_SOURCE_DIR}" "bert.cpp;${COREQSCHMATE_BERT_SOURCE_DIR}" "ggml;${COREQSCHMATE_GGML_SOURCE_DIR}")
    list(GET _pair 0 _name)
    list(GET _pair 1 _directory)
    if(EXISTS "${_directory}/LICENSE")
        install(FILES "${_directory}/LICENSE" DESTINATION schmate/licenses RENAME "${_name}.txt" COMPONENT python)
    endif()
endforeach()
