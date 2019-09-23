KOKKOS_CFG_DEPENDS(CXX_STD COMPILER_ID)

FUNCTION(kokkos_set_cxx_standard_feature standard)
  SET(EXTENSION_NAME CMAKE_CXX${standard}_EXTENSION_COMPILE_OPTION)
  SET(STANDARD_NAME  CMAKE_CXX${standard}_STANDARD_COMPILE_OPTION)
  SET(FEATURE_NAME   cxx_std_${standard})
  #CMake's way of telling us that the standard (or extension)
  #flags are supported is the extension/standard variables
  IF (NOT DEFINED CMAKE_CXX_EXTENSIONS)
    IF(KOKKOS_DONT_ALLOW_EXTENSIONS)
      GLOBAL_SET(KOKKOS_USE_CXX_EXTENSIONS OFF)
    ELSE()
      GLOBAL_SET(KOKKOS_USE_CXX_EXTENSIONS ON)
    ENDIF()
  ELSEIF(CMAKE_CXX_EXTENSIONS)
    IF(KOKKOS_DONT_ALLOW_EXTENSIONS)
      MESSAGE(FATAL_ERROR "The chosen configuration does not support CXX extensions flags: ${KOKKOS_DONT_ALLOW_EXTENSIONS}. Must set CMAKE_CXX_EXTENSIONS=OFF to continue") 
    ELSE()
      GLOBAL_SET(KOKKOS_USE_CXX_EXTENSIONS ON)
    ENDIF()
  ELSE()
    #For trilinos, we need to make sure downstream projects 
    GLOBAL_SET(KOKKOS_USE_CXX_EXTENSIONS OFF)
  ENDIF()

  IF (KOKKOS_USE_CXX_EXTENSIONS AND ${EXTENSION_NAME})
    MESSAGE(STATUS "Using ${${EXTENSION_NAME}} for C++${standard} extensions as feature")
    GLOBAL_SET(KOKKOS_CXX_STANDARD_FEATURE ${FEATURE_NAME})
  ELSEIF(NOT KOKKOS_USE_CXX_EXTENSIONS AND ${STANDARD_NAME})
    MESSAGE(STATUS "Using ${${STANDARD_NAME}} for C++${standard} standard as feature")
    GLOBAL_SET(KOKKOS_CXX_STANDARD_FEATURE ${FEATURE_NAME})
  ELSE()
    #nope, we can't do anything here
    MESSAGE(STATUS "C++${standard} is not supported as a compiler feature - choosing custom flags")
    GLOBAL_SET(KOKKOS_CXX_STANDARD_FEATURE "")
  ENDIF()

  IF(NOT ${FEATURE_NAME} IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    MESSAGE(FATAL_ERROR "Compiler ${KOKKOS_CXX_COMPILER_ID} should support ${FEATURE_NAME}, but CMake reports feature not supported")
  ENDIF()
ENDFUNCTION()


IF (KOKKOS_CXX_STANDARD AND CMAKE_CXX_STANDARD)
  #make sure these are consistent
  IF (NOT KOKKOS_CXX_STANDARD STREQUAL CMAKE_CXX_STANDARD)
    MESSAGE(WARNING "Specified both CMAKE_CXX_STANDARD=${CMAKE_CXX_STANDARD} and KOKKOS_CXX_STANDARD=${KOKKOS_CXX_STANDARD}, but they don't match")
    SET(CMAKE_CXX_STANDARD ${KOKKOS_CXX_STANDARD} CACHE STRING "C++ standard" FORCE)
  ENDIF()
ENDIF()


IF (KOKKOS_CXX_STANDARD STREQUAL "11" )
  kokkos_set_cxx_standard_feature(11)
  GLOBAL_SET(KOKKOS_ENABLE_CXX11 ON)
ELSEIF(KOKKOS_CXX_STANDARD STREQUAL "14")
  kokkos_set_cxx_standard_feature(14)
  GLOBAL_SET(KOKKOS_ENABLE_CXX14 ON)
ELSEIF(KOKKOS_CXX_STANDARD STREQUAL "17")
  kokkos_set_cxx_standard_feature(17)
  GLOBAL_SET(KOKKOS_ENABLE_CXX17 ON)
ELSEIF(KOKKOS_CXX_STANDARD STREQUAL "98")
  MESSAGE(FATAL_ERROR "Kokkos requires C++11 or newer!")
ELSE()
  #set to empty
  GLOBAL_SET(KOKKOS_CXX_STANDARD_FEATURE "")
  IF (KOKKOS_CXX_COMPILER_ID STREQUAL "NVIDIA")
    MESSAGE(FATAL_ERROR "nvcc_wrapper does not support intermediate standards (1Y,1Z,2A) - must use 11, 14, or 17")
  ENDIF()
  #okay, this is funky - kill this variable
  #this value is not really valid as a cmake variable
  UNSET(CMAKE_CXX_STANDARD)
  UNSET(CMAKE_CXX_STANDARD CACHE)
  IF     (KOKKOS_CXX_STANDARD STREQUAL "1Y")
    GLOBAL_SET(KOKKOS_ENABLE_CXX14 ON)
  ELSEIF (KOKKOS_CXX_STANDARD STREQUAL "1Z")
    GLOBAL_SET(KOKKOS_ENABLE_CXX17 ON)
  ELSEIF (KOKKOS_CXX_STANDARD STREQUAL "2A")
    GLOBAL_SET(KOKKOS_ENABLE_CXX20 ON)
  ENDIF()
ENDIF()



# Enforce that extensions are turned off for nvcc_wrapper.
# For compiling CUDA code using nvcc_wrapper, we will use the host compiler's
# flags for turning on C++11.  Since for compiler ID and versioning purposes
# CMake recognizes the host compiler when calling nvcc_wrapper, this just
# works.  Both NVCC and nvcc_wrapper only recognize '-std=c++11' which means
# that we can only use host compilers for CUDA builds that use those flags.
# It also means that extensions (gnu++11) can't be turned on for CUDA builds.

IF(KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
  IF(NOT DEFINED CMAKE_CXX_EXTENSIONS)
    SET(CMAKE_CXX_EXTENSIONS OFF)
  ELSEIF(CMAKE_CXX_EXTENSIONS)
    MESSAGE(FATAL_ERROR "NVCC doesn't support C++ extensions.  Set -DCMAKE_CXX_EXTENSIONS=OFF")
  ENDIF()
ENDIF()

IF(KOKKOS_ENABLE_CUDA)
  # ENFORCE that the compiler can compile CUDA code.
  IF(KOKKOS_CXX_COMPILER_ID STREQUAL Clang)
    IF(KOKKOS_CXX_COMPILER_VERSION VERSION_LESS 4.0.0)
      MESSAGE(FATAL_ERROR "Compiling CUDA code directly with Clang requires version 4.0.0 or higher.")
    ENDIF()
    IF(NOT DEFINED CMAKE_CXX_EXTENSIONS)
      SET(CMAKE_CXX_EXTENSIONS OFF)
    ELSEIF(CMAKE_CXX_EXTENSIONS)
      MESSAGE(FATAL_ERROR "Compiling CUDA code with clang doesn't support C++ extensions.  Set -DCMAKE_CXX_EXTENSIONS=OFF")
    ENDIF()
  ELSEIF(NOT KOKKOS_CXX_COMPILER_ID STREQUAL NVIDIA)
    MESSAGE(FATAL_ERROR "Invalid compiler for CUDA.  The compiler must be nvcc_wrapper or Clang, but compiler ID was ${KOKKOS_CXX_COMPILER_ID}")
  ENDIF()
ENDIF()

IF (NOT KOKKOS_CXX_STANDARD_FEATURE)
  #we need to pick the C++ flags ourselves
  UNSET(CMAKE_CXX_STANDARD)
  UNSET(CMAKE_CXX_STANDARD CACHE)
  IF(KOKKOS_CXX_COMPILER_ID STREQUAL Cray)
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/cray.cmake)
    kokkos_set_cray_flags(${KOKKOS_CXX_STANDARD})
  ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL PGI)
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/pgi.cmake)
    kokkos_set_pgi_flags(${KOKKOS_CXX_STANDARD})
  ELSEIF(KOKKOS_CXX_COMPILER_ID STREQUAL Intel)
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/intel.cmake)
    kokkos_set_intel_flags(${KOKKOS_CXX_STANDARD})
  ELSE()
    INCLUDE(${KOKKOS_SRC_PATH}/cmake/gnu.cmake)
    kokkos_set_gnu_flags(${KOKKOS_CXX_STANDARD})
  ENDIF()
  #check that the compiler accepts the C++ standard flag
  INCLUDE(CheckCXXCompilerFlag)
  IF (DEFINED CXX_STD_FLAGS_ACCEPTED)
    UNSET(CXX_STD_FLAGS_ACCEPTED CACHE)
  ENDIF()
  CHECK_CXX_COMPILER_FLAG(${KOKKOS_CXX_STANDARD_FLAG} CXX_STD_FLAGS_ACCEPTED)
  IF (NOT CXX_STD_FLAGS_ACCEPTED)
    MESSAGE(FATAL_ERROR "${KOKKOS_CXX_COMPILER_ID} did not accept ${KOKKOS_CXX_STANDARD_FLAG}. You likely need to reduce the level of the C++ standard from ${KOKKOS_CXX_STANDARD}")
  ELSE()
    MESSAGE(STATUS "Compiler features not supported, but ${KOKKOS_CXX_COMPILER_ID} accepts ${KOKKOS_CXX_STANDARD_FLAG}")
  ENDIF()
ENDIF()




