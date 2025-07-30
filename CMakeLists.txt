#cfdARCO - high-level framework for solving systems of PDEs on multi-GPUs system
#Copyright (C) 2025 cfdARCO team
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


set(USE_CUDA 1)
set(USE_HIP 0)
set(USE_HIP_NVIDIA 0)
set(BUILD_NO_KERNELBUILDER_EXAMPLES 0)

if(USE_CUDA OR USE_HIP)
    set(USE_GPU 1)
else()
    set(USE_GPU FALSE)
endif()

include(FetchContent)

find_package (OpenMP REQUIRED)
find_package (Eigen3 3.3 REQUIRED NO_MODULE)
find_package (indicators REQUIRED)
find_package(Threads REQUIRED)
#find_package (MPI REQUIRED)
find_package(TBB REQUIRED)
set(JSON_BuildTests OFF CACHE INTERNAL "")
FetchContent_Declare(json GIT_REPOSITORY https://github.com/nlohmann/json.git)
FetchContent_MakeAvailable(json)
FetchContent_Declare(argparse GIT_REPOSITORY https://github.com/p-ranav/argparse.git GIT_TAG v2.9)
FetchContent_MakeAvailable(argparse)
if (USE_HIP)
    find_package(FMT REQUIRED)
endif()
find_package(VTK COMPONENTS
        CommonCore
        CommonDataModel
        IOXML
        REQUIRED
)

set(CFDARCO_JIT_INCLUDE_PATH "${CMAKE_SOURCE_DIR}/include") # Not configurable by user
configure_file(include/jit_config.h.in jit_config.h)

set(CFDARCHO_BASE_SOURCES
        src/cfdarcho_main_3d.cpp
        src/equation.cpp
        src/mesh3d.cpp

        src/variables/dissipation.cpp
        src/variables/dt_update.cpp
        src/variables/interpolation.cpp
        src/variables/spatial_first_derivative.cpp
        src/variables/spatial_second_derivative.cpp
        src/variables/sub_variables.cpp
        src/variables/time_first_derivative.cpp
        src/variables/time_second_derivative.cpp
        src/variables/point_sourse.cpp
        src/variables/variable.cpp
        )

set(CFDARCHO_BASE_INCLUDES
        include/abstract_mesh.hpp
        include/cfdarcho_main_3d.hpp
        include/decls.hpp
        include/cuda_operators.hpp
        include/eigen_holder.hpp
        include/mesh3d.hpp
        include/equation.hpp

        include/variables/dissipation.hpp
        include/variables/dt_update.hpp
        include/variables/interpolation.hpp
        include/variables/spatial_first_derivative.hpp
        include/variables/spatial_second_derivative.hpp
        include/variables/sub_variables.hpp
        include/variables/time_first_derivative.hpp
        include/variables/time_second_derivative.hpp
        include/variables/variable.hpp
)

include(CheckLanguage)

if(USE_CUDA)
    check_language(CUDA)
    if(CMAKE_CUDA_COMPILER)
        set(CFDARCHO_BASE_SOURCES ${CFDARCHO_BASE_SOURCES}
                src/cuda_operators.cu
                src/pool_allocator.cpp
        )
        set(CFDARCHO_BASE_INCLUDES ${CFDARCHO_BASE_INCLUDES}
                include/pool_allocator.hpp
        )
        enable_language(CUDA)
        set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")
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


add_library(cfdARCHOlib SHARED ${CFDARCHO_BASE_SOURCES} ${CFDARCHO_BASE_INCLUDES})

target_include_directories(cfdARCHOlib PUBLIC include)
target_include_directories(cfdARCHOlib PUBLIC ${CMAKE_CURRENT_BINARY_DIR})
if(USE_HIP AND USE_HIP_NVIDIA AND CMAKE_HIP_COMPILER)
    target_include_directories(cfdARCHOlib PUBLIC ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
endif()
target_link_libraries (cfdARCHOlib Eigen3::Eigen)
target_link_libraries (cfdARCHOlib OpenMP::OpenMP_CXX)
target_link_libraries (cfdARCHOlib indicators::indicators)
target_link_libraries (cfdARCHOlib fmt::fmt-header-only)
#target_link_libraries (cfdARCHOlib MPI::MPI_CXX)
target_link_libraries (cfdARCHOlib TBB::tbb)
target_link_libraries(cfdARCHOlib Threads::Threads)
target_link_libraries (cfdARCHOlib nlohmann_json::nlohmann_json)
target_link_libraries (cfdARCHOlib argparse)
target_link_libraries (cfdARCHOlib ${VTK_LIBRARIES})
vtk_module_autoinit(
        TARGETS cfdARCHOlib
        MODULES ${VTK_LIBRARIES}
)

if(USE_CUDA AND CMAKE_CUDA_COMPILER)
    target_link_libraries(cfdARCHOlib rmm::rmm)
    set_target_properties(cfdARCHOlib PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    target_link_libraries(cfdARCHOlib OCCA::libocca)
endif()
if(USE_HIP AND CMAKE_HIP_COMPILER)
    target_link_libraries(cfdARCHOlib OCCA::libocca)
    target_link_libraries(cfdARCHOlib ${CUDA_LIBRARIES})
    target_include_directories(cfdARCHOlib PUBLIC "${HIP_ROOT_DIR}/include")
endif()



function(add_cfdarcho_example TARGET_NAME EXAMPLE_FILE_PATH)
    add_executable(${TARGET_NAME} ${EXAMPLE_FILE_PATH})
    target_link_libraries(${TARGET_NAME} cfdARCHOlib)
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    set_target_properties(${TARGET_NAME} PROPERTIES LINKER_LANGUAGE CXX)
endfunction()


#add_cfdarcho_example(cfdARCO_sod_shock examples/main_3d_sod_shock.cpp)
#add_cfdarcho_example(cfdARCO_gauss_pulse examples/main_3d_wave_gauss_pulse.cu)
add_cfdarcho_example(cfdARCO_euler examples/main_3d_euler.cpp)
add_cfdarcho_example(cfdARCO_wave examples/main_3d_wave.cu)
add_cfdarcho_example(cfdARCO_elastic_wave examples/main_3d_wave_elastic.cu)
add_cfdarcho_example(cfdARCO_heat examples/main_3d_heat.cpp)
add_cfdarcho_example(cfdARCO_burgers examples/main_3d_burgers.cpp)
add_cfdarcho_example(cfdARCO_navier_stokes examples/main_3d_navier_stokes.cpp)
add_cfdarcho_example(cfdARCO_maxwell examples/main_3d_maxwell_equations.cu)
add_cfdarcho_example(cfdARCO_reaction_diffusion examples/main_3d_reaction_diffusion.cpp)
add_cfdarcho_example(cfdARCO_shallow_water_2d examples/main_2d_shallow_water.cpp)
add_cfdarcho_example(cfdARCO_phase_field examples/main_3d_phase_field.cpp)
#add_cfdarcho_example(testing_setup examples/testing_setup.cpp)

if(BUILD_NO_KERNELBUILDER_EXAMPLES)
    add_cfdarcho_example(cfdARCO_euler_no_kernel_builder examples/no_kernel_builder/main_3d_euler.cpp)
    add_cfdarcho_example(cfdARCO_wave_no_kernel_builder examples/no_kernel_builder/main_3d_wave.cu)
    add_cfdarcho_example(cfdARCO_elastic_wave_no_kernel_builder examples/no_kernel_builder/main_3d_wave_elastic.cu)
    add_cfdarcho_example(cfdARCO_heat_no_kernel_builder examples/no_kernel_builder/main_3d_heat.cpp)
    add_cfdarcho_example(cfdARCO_burgers_no_kernel_builder examples/no_kernel_builder/main_3d_burgers.cpp)
    add_cfdarcho_example(cfdARCO_navier_stokes_no_kernel_builder examples/no_kernel_builder/main_3d_navier_stokes.cpp)
    add_cfdarcho_example(cfdARCO_maxwell_no_kernel_builder examples/no_kernel_builder/main_3d_maxwell_equations.cu)
    add_cfdarcho_example(cfdARCO_reaction_diffusion_no_kernel_builder examples/no_kernel_builder/main_3d_reaction_diffusion.cpp)
    add_cfdarcho_example(cfdARCO_phase_field_no_kernel_builder examples/no_kernel_builder/main_3d_phase_field.cpp)
endif()
