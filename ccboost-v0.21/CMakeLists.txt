project (ContextCueBoosting)

# we need v2.8 for ExternalProject
cmake_minimum_required (VERSION 2.8) 
include(ExternalProject)

## Needs ITK at least v4
find_package(ITK REQUIRED)
include(${ITK_USE_FILE})

if( "${ITK_VERSION_MAJOR}" LESS 4 )
  message(FATAL_ERROR "ITK version 4 required")
endif()

## RANDOMLIB
set(RANDOMLIB_DESTPATH "${CMAKE_BINARY_DIR}/randomlib")
set(RANDOMLIB_LIB "${RANDOMLIB_DESTPATH}/lib/libRandom.a")
set(RANDOMLIB_INCPATH "${RANDOMLIB_DESTPATH}/include")

externalproject_add( 
    randomlibBuild
    GIT_REPOSITORY http://git.code.sf.net/p/randomlib/code
    GIT_TAG "r1.6"
    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${RANDOMLIB_DESTPATH}" "-DCMAKE_BUILD_TYPE=RELEASE" "-DMAINTAINER=OFF" "-DRANDOM_SHARED_LIB=OFF" "-DDISABLE_BOOST=ON" "-DRANDOMLIB_DOCUMENTATION=OFF"
    UPDATE_COMMAND "" )

add_library(randomlib STATIC IMPORTED DEPENDS)
add_dependencies( randomlib randomlibBuild)
set_property(TARGET randomlib PROPERTY IMPORTED_LOCATION ${RANDOMLIB_LIB})

## LIBCONFIG
set(LIBCONFIG_DESTPATH "${CMAKE_BINARY_DIR}/libconfig")
set(LIBCONFIG_LIB "${LIBCONFIG_DESTPATH}/lib/libconfig++.a")
set(LIBCONFIG_INCPATH "${LIBCONFIG_DESTPATH}/include")

externalproject_add( 
    configlibBuild
    GIT_REPOSITORY https://bitbucket.org/cbecker/libconfig.git
    CMAKE_ARGS "-DCMAKE_INSTALL_PREFIX=${LIBCONFIG_DESTPATH}"
    UPDATE_COMMAND "" )

add_library(libconfig STATIC IMPORTED)
add_dependencies( libconfig configlibBuild)
set_property(TARGET libconfig PROPERTY IMPORTED_LOCATION ${LIBCONFIG_LIB})


# Needed for log loss
#set(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH} /home/cjbecker/filer/bitbucket/liblbfgs-1.10/lib/.libs)

include_directories(${LIBCONFIG_INCPATH} ${RANDOMLIB_INCPATH})


# Flags
# static release is pretty generic, possibly not the fastest one though
set(CMAKE_CXX_FLAGS_RELEASE_ALLSTATIC "-fopenmp -O3")
set(CMAKE_EXE_LINKER_FLAGS_RELEASE_ALLSTATIC "-static-libgcc -static-libstdc++ -static")

set(CMAKE_CXX_FLAGS_RELEASE "-fopenmp -O3 -march=native -mtune=native")
set(CMAKE_CXX_FLAGS_DEBUG "-msse2 -msse3 -mssse3 -march=native -g")

if (NOT CMAKE_BUILD_TYPE)
	message(STATUS "No build type selected, default to Release")
	set(CMAKE_BUILD_TYPE "Release")
endif()

include_directories( ${ContextCueBoosting_SOURCE_DIR} )
include_directories( ${ContextCueBoosting_SOURCE_DIR}/extra )
include_directories( ${ContextCueBoosting_SOURCE_DIR}/eigen )

add_executable( ccboost main.cpp ConfigData.cpp slic/LKM.cpp slic/utils.cpp )
target_link_libraries( ccboost ${ITK_LIBRARIES} libconfig randomlib rt )

## Purely ITK-based executables
add_executable( GaussianImageFilter auxItk/GaussianImageFilter.cxx )
target_link_libraries( GaussianImageFilter ${ITK_LIBRARIES} )

add_executable( GradientMagnitudeImageFilter auxItk/GradientMagnitudeImageFilter.cxx )
target_link_libraries( GradientMagnitudeImageFilter ${ITK_LIBRARIES} )

add_executable( LoGImageFilter auxItk/LoGImageFilter.cxx )
target_link_libraries( LoGImageFilter ${ITK_LIBRARIES} )

add_executable( EigenOfHessianImageFilter auxItk/EigenOfHessianImageFilter.cxx )
target_link_libraries( EigenOfHessianImageFilter ${ITK_LIBRARIES} )

add_executable( SingleEigenVectorOfHessian auxItk/SingleEigenVectorOfHessian.cxx )
target_link_libraries( SingleEigenVectorOfHessian ${ITK_LIBRARIES} )

add_executable( EigenOfStructureTensorImageFilter auxItk/EigenOfStructureTensorImageFilter.cxx )
target_link_libraries( EigenOfStructureTensorImageFilter ${ITK_LIBRARIES} )

add_executable( RepolarizeYVersorWithGradient auxItk/RepolarizeYVersorWithGradient.cxx )
target_link_libraries( RepolarizeYVersorWithGradient ${ITK_LIBRARIES} )

add_executable( AllEigenVectorsOfHessian auxItk/AllEigenVectorsOfHessian.cxx )
target_link_libraries( AllEigenVectorsOfHessian ${ITK_LIBRARIES} )

add_executable( BinaryErodeDilateSynapseImageFilter auxItk/BinaryErodeDilateSynapseImageFilter.cxx )
target_link_libraries( BinaryErodeDilateSynapseImageFilter ${ITK_LIBRARIES} )
