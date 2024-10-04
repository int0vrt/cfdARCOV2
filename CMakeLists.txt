#cfdARCO - high-level framework for solving systems of PDEs on multi-GPUs system
#Copyright (C) 2024 cfdARCHO
#
#This program is free software: you can redistribute it and/or modify
#it under the terms of the GNU General Public License as published by
#the Free Software Foundation, either version 3 of the License, or
#(at your option) any later version.
#
#This program is distributed in the hope that it will be useful,
#but WITHOUT ANY WARRANTY; without even the implied warranty of
#MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#GNU General Public License for more details.
#
#You should have received a copy of the GNU General Public License
#along with this program.  If not, see <http://www.gnu.org/licenses/>.

cmake_minimum_required(VERSION 3.22)
project(cfdARCO LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

#option(USE_CUDA "USE_CUDA" OFF)
#option(USE_HIP "USE_HIP" ON)

set(USE_CUDA 1)
set(USE_HIP 0)
set(USE_HIP_NVIDIA 0)

if(USE_CUDA OR USE_HIP)
    set(USE_GPU 1)
else()
    set(USE_GPU FALSE)
endif()

include(FetchContent)

find_package (OpenMP REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (indicators REQUIRED)
find_package(MPI REQUIRED)
if (USE_HIP)
    find_package(FMT REQUIRED)
endif()
find_package(VTK COMPONENTS
        CommonCore
        CommonDataModel
        IOXML
        REQUIRED
)
#find_package(CGAL REQUIRED)


set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG v3.11.2
        )
FetchContent_MakeAvailable(json)
FetchContent_Declare(argparse
        GIT_REPOSITORY https://github.com/p-ranav/argparse.git
        GIT_TAG v2.9
        )
FetchContent_MakeAvailable(argparse)


set(CFDARCHO_BASE_SOURCES
        include/abstract_mesh.hpp
#        include/mesh2d.hpp
#        src/mesh2d.cpp
#        include/fvm.hpp
#        src/fvm.cpp
#        src/cfdarcho_main.cpp
#        include/cfdarcho_main.hpp
        src/cfdarcho_main_3d.cpp
        include/cfdarcho_main_3d.hpp
        include/decls.hpp
#        include/distribution_algo.hpp
#        src/distribution_algo.cpp
        include/cuda_operators.hpp
#        include/io_operators.hpp
#        src/io_operators.cpp
#        src/utils.cpp
#        src/val_utils.cpp
        include/eigen_holder.hpp
        include/mesh3d.hpp
        src/fvm3d.cpp
        src/mesh3d.cpp
        )

set(CFDARCHO_SOURCES_SOD_SHOCK ${CFDARCHO_BASE_SOURCES}
        examples/main_3d_sod_shock.cpp
        )

set(CFDARCHO_SOURCES_GAUSS_PULSE ${CFDARCHO_BASE_SOURCES}
        examples/main_3d_wave_gauss_pulse.cu
)

set(CFDARCHO_SOURCES_EULER ${CFDARCHO_BASE_SOURCES}
        examples/main_3d.cpp
)

set(CFDARCHO_SOURCES_WAVE ${CFDARCHO_BASE_SOURCES}
        examples/main_3d_wave.cu
)

set(CFDARCHO_SOURCES_HEAT ${CFDARCHO_BASE_SOURCES}
        examples/main_3d_heat.cpp
)

include(CheckLanguage)

if(USE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(CFDARCHO_SOURCES_SOD_SHOCK ${CFDARCHO_SOURCES_SOD_SHOCK}
                src/cuda_operators.cu
                include/pool_allocator.hpp
                src/pool_allocator.cpp
        )
        set(CFDARCHO_SOURCES_GAUSS_PULSE ${CFDARCHO_SOURCES_GAUSS_PULSE}
                src/cuda_operators.cu
                include/pool_allocator.hpp
                src/pool_allocator.cpp
        )
        set(CFDARCHO_SOURCES_EULER ${CFDARCHO_SOURCES_EULER}
                src/cuda_operators.cu
                include/pool_allocator.hpp
                src/pool_allocator.cpp
                )
        set(CFDARCHO_SOURCES_WAVE ${CFDARCHO_SOURCES_WAVE}
                src/cuda_operators.cu
                include/pool_allocator.hpp
                src/pool_allocator.cpp
        )
        set(CFDARCHO_SOURCES_HEAT ${CFDARCHO_SOURCES_HEAT}
                src/cuda_operators.cu
                include/pool_allocator.hpp
                src/pool_allocator.cpp
        )
        enable_language(CUDA)
        add_compile_definitions(CFDARCHO_CUDA_ENABLE)

        find_package(OCCA REQUIRED)

        FetchContent_Declare(rmm
                GIT_REPOSITORY https://github.com/rapidsai/rmm.git
                GIT_TAG v23.04.00
        )
        FetchContent_MakeAvailable(rmm)
    endif()
endif()



if(USE_HIP)
    check_language(HIP)
    if(CMAKE_HIP_COMPILER)
        enable_language(HIP)
        add_compile_definitions(CFDARCHO_HIP_ENABLE)
        find_package(OCCA REQUIRED)
    endif()
endif()

if(USE_HIP AND USE_HIP_NVIDIA AND CMAKE_HIP_COMPILER)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__HIP_PLATFORM_NVIDIA__ -Wno-unused-result")
    enable_language(CUDA)
endif()

add_executable(cfdARCO_sod_shock ${CFDARCHO_SOURCES_SOD_SHOCK})
add_executable(cfdARCO_gauss_pulse ${CFDARCHO_SOURCES_GAUSS_PULSE})
add_executable(cfdARCO_euler ${CFDARCHO_SOURCES_EULER})
add_executable(cfdARCO_wave ${CFDARCHO_SOURCES_WAVE})
add_executable(cfdARCO_heat ${CFDARCHO_SOURCES_HEAT})

set(CFDARCO_JIT_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/include") # Not configurable by user
configure_file(include/jit_config.h.in jit_config.h)

function(set_cfdarcho_links TARGET_NAME)
    target_include_directories(${TARGET_NAME} PRIVATE include)
    target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CURRENT_BINARY_DIR})
    if(USE_HIP AND USE_HIP_NVIDIA AND CMAKE_HIP_COMPILER)
        target_include_directories(${TARGET_NAME} PRIVATE ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
    endif()
    target_link_libraries (${TARGET_NAME} Eigen3::Eigen)
    target_link_libraries (${TARGET_NAME} OpenMP::OpenMP_CXX)
    target_link_libraries (${TARGET_NAME} indicators::indicators)
#    if (NOT USE_GPU)
        target_link_libraries (${TARGET_NAME} fmt::fmt-header-only)
#    endif()
    target_link_libraries(${TARGET_NAME} MPI::MPI_CXX)
    target_link_libraries(${TARGET_NAME} nlohmann_json::nlohmann_json)
    target_link_libraries(${TARGET_NAME} argparse)
    target_link_libraries(${TARGET_NAME} ${VTK_LIBRARIES})
    vtk_module_autoinit(
            TARGETS ${TARGET_NAME}
            MODULES ${VTK_LIBRARIES}
    )

    if(USE_CUDA AND CMAKE_CUDA_COMPILER)
        target_link_libraries(${TARGET_NAME} rmm::rmm)
        set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
        target_link_libraries(${TARGET_NAME} OCCA::libocca)
    endif()
    if(USE_HIP AND CMAKE_HIP_COMPILER)
        target_link_libraries(${TARGET_NAME} OCCA::libocca)
        target_link_libraries(${TARGET_NAME} ${CUDA_LIBRARIES})
        target_include_directories(${TARGET_NAME} PRIVATE "${HIP_ROOT_DIR}/include")
    endif()
endfunction()

set_cfdarcho_links(cfdARCO_sod_shock)
set_cfdarcho_links(cfdARCO_gauss_pulse)
set_cfdarcho_links(cfdARCO_euler)
set_cfdarcho_links(cfdARCO_wave)
set_cfdarcho_links(cfdARCO_heat)


