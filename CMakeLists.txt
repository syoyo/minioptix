# 3.13 = target_link_directories
cmake_minimum_required(VERSION 3.13)

set(MINIOPTIX_BUILD_TARGET "minioptix")

project(${MINIOPTIX_BUILD_TARGET} C CXX)

# options
option(MINIOPTIX_USE_CCACHE "Use ccache for faster recompile." ON)
option(MINIOPTIX_WITH_CUEW "Use cuew(no CUDA SDK required, but need to manually generate PTX)" ON)
set(MINIOPTIX_CUSTOM_LINKER "" CACHE STRING "Set custom linker name(e.g. 'lld'). Assume the compiler supports `-fuse=ld=` linker flags.")

#set(OptiX_INSTALL_DIR "${CMAKE_SOURCE_DIR}/../" CACHE PATH "Path to OptiX installed location.")

# cmake modules
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake/sanitizers)
find_package(Sanitizers) # Address sanitizer (-DSANITIZE_ADDRESS=ON)

# Assume looking for `cmake/FindOptiX.cmake`
find_package(OptiX REQUIRED)


# Require strict C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# TODO(LTE): Only enable PIC when compiling python module(dll)?.
set(CMAKE_POSITION_INDEPENDENT_CODE On)

if (MINIOPTIX_WITH_PYBIND11)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11)
endif (MINIOPTIX_WITH_PYBIND11)

# [ccache]
if(MINIOPTIX_USE_CCACHE)
  if(MSVC)
    # No ccache support
  else()
    find_program(CCACHE_EXE ccache)
    if(CCACHE_EXE)

      message(STATUS "Use ccache : " ${CCACHE_EXE})
      # CMAKE_C_COMPILER_LAUNCHER = available from cmake 3.4+)
      if(CMAKE_C_COMPILER_LAUNCHER)
        set(CMAKE_C_COMPILER_LAUNCHER "${CMAKE_C_COMPILER_LAUNCHER}"
                                      "${CCACHE_EXE}")
      else()
        set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_EXE}")
      endif()

      if(CMAKE_CXX_COMPILER_LAUNCHER)
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CMAKE_CXX_COMPILER_LAUNCHER}"
                                        "${CCACHE_EXE}")
      else()
        set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_EXE}")
      endif()

    endif(CCACHE_EXE)
  endif()
endif()

# [cppdap]
if(MINIOPTIX_WITH_CPPDAP)

  # set(CPPDAP_THIRD_PARTY_DIR <third-party-root-directory>) # defaults to
  # ${CPPDAP_DIR}/third_party set(CPPDAP_JSON_DIR        <path-to-nlohmann-
  # json>)      # defaults to ${CPPDAP_THIRD_PARTY_DIR}/json
  # set(CPPDAP_GOOGLETEST_DIR  <path-to-googletest>)         # defaults to
  # ${CPPDAP_THIRD_PARTY_DIR}/googletest

  set(CPPDAP_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cppdap)
  add_subdirectory(${CPPDAP_DIR})

endif(MINIOPTIX_WITH_CPPDAP)


if(MINIOPTIX_WITH_OPTIX)
  # Use cuew for CUDA API
  list(APPEND MINIOPTIX_DEP_SOURCES  ${CMAKE_CURRENT_SOURCE_DIR}/deps/cuew/src/cuew.c)
  find_package(OptiX REQUIRED)
endif(MINIOPTIX_WITH_OPTIX)


set(MINIOPTIX_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/src/main.cc
    )



if(MINIOPTIX_WITH_GUI)
endif(MINIOPTIX_WITH_GUI)

add_executable(${MINIOPTIX_BUILD_TARGET} ${MINIOPTIX_SOURCES} ${MINIOPTIX_DEP_SOURCES} ${MINIOPTIX_GUI_SOURCES})
add_sanitizers(${MINIOPTIX_BUILD_TARGET})


# custom linker
if( MINIOPTIX_CUSTOM_LINKER )

  message(STATUS "Use custom linker: " ${MINIOPTIX_CUSTOM_LINKER})

  # TODO(LTE): Check if `-fuse-ld=` argument is supported on the compiler.
  set_property(TARGET ${MINIOPTIX_BUILD_TARGET} APPEND_STRING PROPERTY LINK_FLAGS " -fuse-ld=${MINIOPTIX_CUSTOM_LINKER}")

endif()

if(MINIOPTIX_WITH_CPPDAP)
  target_link_libraries(${MINIOPTIX_BUILD_TARGET} cppdap)
  target_compile_definitions(${MINIOPTIX_BUILD_TARGET} PUBLIC MINIOPTIX_USE_CPPDAP)
  target_include_directories(${MINIOPTIX_BUILD_TARGET} PUBLIC "${CPPDAP_DIR}/include")

endif(MINIOPTIX_WITH_CPPDAP)

target_include_directories(${MINIOPTIX_BUILD_TARGET} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src)

set_target_properties(${MINIOPTIX_BUILD_TARGET}
                      PROPERTIES RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR})

if(MINIOPTIX_WITH_CUEW)
  target_include_directories(
    ${MINIOPTIX_BUILD_TARGET}
    PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/deps/cuew/include" ${OptiX_INCLUDE})

  target_compile_definitions(
    ${MINIOPTIX_BUILD_TARGET} PUBLIC "MINIOPTIX_USE_CUEW")
endif()



if (WIN32)
  target_link_libraries(${MINIOPTIX_BUILD_TARGET} PUBLIC advapi32)
endif (WIN32)

if(MINIOPTIX_WITH_GUI)

  if (WIN32)
    target_link_libraries(${MINIOPTIX_BUILD_TARGET} PUBLIC shell32 gdi32 ole32)
  endif (WIN32)


  # [imgui]
  target_include_directories(${MINIOPTIX_BUILD_TARGET}
                             PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/deps/")

  # [glad]
  target_compile_definitions(${MINIOPTIX_BUILD_TARGET} PRIVATE
                             IMGUI_IMPL_OPENGL_LOADER_GLAD)
  target_include_directories(${MINIOPTIX_BUILD_TARGET}
                             PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/deps/glad/include")

  target_include_directories(${MINIOPTIX_BUILD_TARGET}
                             PRIVATE ${glfw_INCLUDE_DIRS})
  list(APPEND MINIOPTIX_EXT_GUI_LIBRARIES
              glfw
              ${OPENGL_LIBRARIES}
              ${CMAKE_DL_LIBS})
endif(MINIOPTIX_WITH_GUI)

target_link_libraries(${MINIOPTIX_BUILD_TARGET} PUBLIC ${MINIOPTIX_EXT_LIBRARIES} ${CMAKE_DL_LIBS})
target_link_libraries(${MINIOPTIX_BUILD_TARGET} PRIVATE ${MINIOPTIX_EXT_GUI_LIBRARIES})

# Increase warning level for clang.
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")

  include(CheckCXXCompilerFlag)

  # -Wno-delete-non-abstract-non-virtual-dtor
  # -Wno-delete-non-virtual-dtor
  #  workaround for pbrtParser https://github.com/ingowald/pbrt-parser/issues/39
  check_cxx_compiler_flag("-Wno-delete-non-abstract-non-virtual-dtor" HAS_NO_DELETE_NON_ABSTRACT_NON_VIRTUAL_DTOR)
  if (HAS_NO_DELETE_NON_ABSTRACT_NON_VIRTUAL_DTOR)
    string(APPEND OPTIONAL_CXX_FLAGS " -Wno-delete-non-abstract-non-virtual-dtor ")
  endif ()

  check_cxx_compiler_flag("-Wno-delete-non-virtual-dtor" HAS_NO_DELETE_NON_VIRTUAL_DTOR)
  if (HAS_NO_DELETE_NON_VIRTUAL_DTOR)
    string(APPEND OPTIONAL_CXX_FLAGS " -Wno-delete-non-virtual-dtor ")
  endif ()

  set(CLANG_COMPILE_FLAGS "-Weverything -Werror -Wno-padded -Wno-c++98-compat-pedantic -Wno-documentation -Wno-documentation-unknown-command -Wno-reserved-id-macro -Wno-unused-variable  -Wno-exit-time-destructors -Wno-global-constructors ")

  string(APPEND CLANG_COMPILE_FLAGS ${OPTIONAL_CXX_FLAGS})

  if(MINGW)
    # TODO(LTE): increase warning level for llvm-mingw.
  else()
    set_source_files_properties(
      ${MINIOPTIX_SOURCES}
      PROPERTIES
      COMPILE_FLAGS
      ${CLANG_COMPILE_FLAGS}
      )
  endif()
endif()

if (MINIOPTIX_BUILD_TESTS)
  add_subdirectory(tests)
endif (MINIOPTIX_BUILD_TESTS)

# postbuild
if (WIN32)
  if (LLVM_MINGW_ROOT_ABSPATH)

    # Copy libunwind.dll and libc++.dll
    add_custom_command(TARGET ${MINIOPTIX_BUILD_TARGET} POST_BUILD        # Adds a post-build event to MyTest
        COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
            "${LLVM_MINGW_ROOT_ABSPATH}/x86_64-w64-mingw32/bin/libc++.dll"      # src
            $<TARGET_FILE_DIR:${MINIOPTIX_BUILD_TARGET}>)           # dest

    add_custom_command(TARGET ${MINIOPTIX_BUILD_TARGET} POST_BUILD        # Adds a post-build event to MyTest
        COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
            "${LLVM_MINGW_ROOT_ABSPATH}/x86_64-w64-mingw32/bin/libunwind.dll"      # src
            $<TARGET_FILE_DIR:${MINIOPTIX_BUILD_TARGET}>)           # dest

    # TODO(LTE): Copy asan dll only when the app is build with asan support
    add_custom_command(TARGET ${MINIOPTIX_BUILD_TARGET} POST_BUILD        # Adds a post-build event to MyTest
        COMMAND ${CMAKE_COMMAND} -E copy_if_different  # which executes "cmake - E copy_if_different..."
            "${LLVM_MINGW_ROOT_ABSPATH}/x86_64-w64-mingw32/bin/libclang_rt.asan_dynamic-x86_64.dll"      # src
            $<TARGET_FILE_DIR:${MINIOPTIX_BUILD_TARGET}>)           # dest

  endif (LLVM_MINGW_ROOT_ABSPATH)

endif (WIN32)

# [VisualStudio]
if(WIN32)
  # Set ${MINIOPTIX_BUILD_TARGET} as a startup project for VS IDE
  set_property(DIRECTORY PROPERTY VS_STARTUP_PROJECT ${MINIOPTIX_BUILD_TARGET})

  # For easier debugging in VS IDE(cmake 3.8.0 or later required) Set working
  # directory to ${MINIOPTIX_BUILD_TARGET} git repo root.
  if(CMAKE_VERSION VERSION_GREATER 3.8.0)
    set_target_properties(${MINIOPTIX_BUILD_TARGET}
                          PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY
                                     "${CMAKE_CURRENT_SOURCE_DIR}")
  endif()
endif()
