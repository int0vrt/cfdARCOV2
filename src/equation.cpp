/*
cfdARCO - high-level framework for solving systems of PDEs on multi-GPUs system
Copyright (C) 2025 cfdARCO team

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/
// This is a personal academic project. Dear PVS-Studio, please check it.
// PVS-Studio Static Code Analyzer for C, C++, C#, and Java: https://pvs-studio.com


#include "equation.hpp"
#include "cuda_operators.hpp"
#include "kernel_builder.hpp"
#include "variables/dt_update.hpp"

#include <indicators/progress_bar.hpp>
#include <iostream>


void EqSolver::solve_dt(Variable *equation, Variable *time_var, Variable *set_var, DT *dt) {
    if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
        auto current_cu = equation->evaluate_cu();
        sync_device();
        auto extracted_cu = time_var->extract_cu(current_cu, dt->_dt);
        set_var->set_current(extracted_cu, 0);
        if (0) {
            std::cout << "At " << set_var->name << " Current sum = " << set_var->current.sum() << " Current max = "
                      << set_var->current.maxCoeff() << std::endl;
        }

    } else {
        Eigen::Matrix<float, -1, 1> current = equation->evaluate();
        auto extracted = time_var->extract(current, dt->_dt);
        set_var->set_current(extracted);

        if (0) {
            std::cout << "At " << set_var->name << " Current sum = " << extracted.sum() << " Current max = "
                      << extracted.maxCoeff() << std::endl;
        }
    }
}

Equation::Equation(size_t timesteps_) : timesteps{timesteps_} {}

void Equation::evaluate(std::vector<Variable *> &all_vars,
                        EquationTemplate &equation_system, DT *dt, bool visualize,
                        std::vector<Variable *> store_vars) const {
    indicators::ProgressBar bar{
            indicators::option::BarWidth{50},
            indicators::option::Start{"["},
            indicators::option::Fill{"="},
            indicators::option::Lead{">"},
            indicators::option::Remainder{" "},
            indicators::option::End{"]"},
            indicators::option::PostfixText{"0.0 %"},
            indicators::option::ShowElapsedTime{true},
            indicators::option::ShowRemainingTime{true},
            indicators::option::ForegroundColor{indicators::Color::green},
            indicators::option::FontStyles{std::vector<indicators::FontStyle>{indicators::FontStyle::bold}}
    };

    const bool use_kernel_builder = USE_GPU && (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled);

#if USE_GPU
    std::vector<KernelBuilder> krns{};

    if (use_kernel_builder) {
        for (auto &equation: equation_system) {
            if (std::get<3>(equation)) {
                krns.emplace_back(std::get<0>(equation), &std::get<2>(equation));
                std::get<0>(equation)->jit_post();
            } else {
                krns.emplace_back(nullptr, nullptr);
            }
        }
        for (int i = 0; i < krns.size(); ++i) {
            if (std::get<3>(equation_system.at(i))) {
                krns.at(i).build(i);
            }
        }
    }
#endif

    for (int t = 0; t < timesteps; ++t) {
        dt->update();

        if (!use_kernel_builder) {
            for (auto &equation: equation_system) {
                auto left_part = std::get<0>(equation);
                auto &right_part = std::get<2>(equation);
                left_part->solve(&right_part, dt);
            }
        } else {
#if USE_GPU
            for (int i = 0; i < krns.size(); ++i) {
                auto &eq_crr = equation_system.at(i);
                if (std::get<3>(eq_crr)) {
                    auto *left_part = krns.at(i).left_val;
                    left_part->set_dt(dt->_dt);
                    krns.at(i).run();
                    left_part->jit_post();
                } else {
                    auto left_part = std::get<0>(equation_system.at(i));
                    auto &right_part = std::get<2>(equation_system.at(i));
                    left_part->solve(&right_part, dt);
                }
                sync_device();
            }
#endif
        }

        for (auto &var: all_vars) {
            if (var->has_boundary_conditions_cu &&
                (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled)) {
                var->set_bound_cu(dt);
            } else {
                if (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled) {
                    var->current = var->current_cu.to_eigen();
                }
                var->set_bound(dt);
            }
        }
        for (auto& var : store_vars) {
            var->add_history();
        }

        if (visualize && CFDArcoGlobalInit::get_rank() == 0) {
            float progress = (static_cast<float>(t) / static_cast<float>(timesteps)) * 100;
            bar.set_progress(static_cast<size_t>(progress));
            bar.set_option(indicators::option::PostfixText{std::to_string(progress) + " %"});
        }

    }

    for (auto var: all_vars) {
        if (var->has_boundary_conditions_cu && (CFDArcoGlobalInit::cuda_enabled || CFDArcoGlobalInit::hip_enabled)) {
            var->current = var->current_cu.to_eigen();
        }
        std::cout << "At " << var->name << " Current sum = " << var->current.sum() << " Current max = "
                  << var->current.maxCoeff() << std::endl;
    }

    if (CFDArcoGlobalInit::get_rank() == 0) std::cout << "Time progress = " << dt->_current_time_dbl << std::endl;
}

#if USE_GPU

void KernelBuilder::build(int kern_id) {
    int pipeline_stages = CFDArcoGlobalInit::pipe_steps;
    const int use_pipe = static_cast<int>(CFDArcoGlobalInit::use_pipe);
    const int blocksize = CFDArcoGlobalInit::blocksize;
    const int launch_sim_blocks = CFDArcoGlobalInit::launch_sim_blocks;

    if (use_pipe) {
        std::cout << "Using pipeline with nsteps = " << pipeline_stages << std::endl;
    } else {
        std::cout << "NOT Using pipeline with nsteps = " << pipeline_stages << std::endl;
        pipeline_stages = 1;
    }

    std::string kernel_template = "#define BLOCKSIZE {} \n"
                                  "#define PIPELINE_SUPPORT {} \n"
                                  "#define pipeline_stages {} \n"
                                  "#define LNCH_SIM_BLK {} \n"
                                  "#include <custom_cuda_functions.hpp>\n"
                                  "extern \"C\" __global__ __launch_bounds__(BLOCKSIZE, LNCH_SIM_BLK) void cfdARCHOKernel({} size_t n) {{\n"    // here goes inputs
                                  "{} \n"  // here goes declataions

                                  "size_t pipe_step = BLOCKSIZE; \n"
                                  "size_t idx = (size_t) blockIdx.x * pipeline_stages * blockDim.x + threadIdx.x; \n"

                                  "#pragma unroll\n"
                                  "for (int pipe = 0; pipe < pipeline_stages; pipe++) {{\n"

                                  "{} \n"  // here goes pipeline initialization

                                  "#if PIPELINE_SUPPORT\n"
                                  "__syncthreads();\n"
                                  "__pipeline_commit();\n"
                                  "#endif\n"

                                  "idx += pipe_step;\n"

                                  "}} \n"

                                  "idx = (size_t) blockIdx.x * pipeline_stages * blockDim.x + threadIdx.x; \n"

                                  "#pragma unroll\n"
                                  "for (int pipe = 0; pipe < pipeline_stages; pipe++) {{ \n"

                                  "#if PIPELINE_SUPPORT\n"
                                  "__pipeline_wait_prior(pipeline_stages - pipe - 1);\n"
                                  "__syncthreads();\n"
                                  "#endif\n"

                                  "{} \n"  // here goes operations
                                  "if (idx < n) {{\n"
                                  "{} \n"  // here goes assign
                                  "}}\n"

                                  "idx += pipe_step;\n"
                                  "}}\n"
                                  "}}";

    auto vec_jit_declarations = expression->get_jit_declarations();
    auto left_jit_declaration = left_val->get_jit_declarations();
    if (left_val->is_dt2) {
        vec_jit_declarations.insert(vec_jit_declarations.end(), left_jit_declaration.begin(),
                                    left_jit_declaration.end());
    }

    auto vec_jit_inputs = expression->get_jit_inputs();

    auto vec_jit_operations = expression->get_jit_operations();

    auto vec_jit_memops = expression->get_jit_memops();
    auto left_jit_memops = left_val->get_jit_memops();
    if (left_val->is_dt2) {
        vec_jit_memops.insert(vec_jit_memops.end(), left_jit_memops.begin(), left_jit_memops.end());
    }

    std::string full_jit_declarations;
    std::vector<std::string> all_decls = {};
    for (const auto &vec_jit_declaration: vec_jit_declarations) {
        if (std::find(all_decls.begin(), all_decls.end(), vec_jit_declaration) == all_decls.end()) {
            full_jit_declarations += vec_jit_declaration + "\n";
            all_decls.push_back(vec_jit_declaration);
        }
    }

    std::string full_jit_inputs;
    std::vector<std::string> all_ins = {};
    std::vector<std::tuple<std::string, void *, TypeEnum>> inputs_in_order = {};
    for (const auto &vec_jit_input: vec_jit_inputs) {
        if (std::find(all_ins.begin(), all_ins.end(), std::get<0>(vec_jit_input)) == all_ins.end()) {
            full_jit_inputs += std::get<0>(vec_jit_input) + ", ";
            all_ins.push_back(std::get<0>(vec_jit_input));
            inputs_in_order.push_back(vec_jit_input);
        }
    }

    auto assign_inputs = left_val->get_jit_inputs();
    for (const auto &vec_jit_input: assign_inputs) {
        if (std::find(all_ins.begin(), all_ins.end(), std::get<0>(vec_jit_input)) == all_ins.end()) {
            full_jit_inputs += std::get<0>(vec_jit_input) + ", ";
            all_ins.push_back(std::get<0>(vec_jit_input));
            inputs_in_order.push_back(vec_jit_input);
        }
    }

    std::string full_jit_operations;
    std::vector<std::string> all_ops = {};
    for (const auto &vec_jit_operation: vec_jit_operations) {
        if (std::find(all_ops.begin(), all_ops.end(), vec_jit_operation) == all_ops.end()) {
            full_jit_operations += vec_jit_operation + "\n";
            all_ops.push_back(vec_jit_operation);
        }
    }

    std::string full_jit_memops;
    std::vector<std::string> all_memops = {};
    for (const auto &vec_jit_memop: vec_jit_memops) {
        if (std::find(all_memops.begin(), all_memops.end(), vec_jit_memop) == all_memops.end()) {
            full_jit_memops += vec_jit_memop + "\n";
            all_memops.push_back(vec_jit_memop);
        }
    }


    int nblocks = std::ceil(
            static_cast<float>(expression->mesh->_num_nodes) / (static_cast<float>(blocksize) * pipeline_stages));

    std::cout << "nblocks = " << nblocks << std::endl;

    auto jit_assigns = left_val->jit_assign;
    auto jit_returns = expression->get_jit_returns();
    std::string full_jit_assign = "";
    for (int i = 0; i < jit_assigns.size(); ++i) {
        auto curr_jit_assign = fmt::format(jit_assigns.at(i), jit_returns.at(i));
        full_jit_assign = full_jit_assign + curr_jit_assign + "\n";
    }

    std::string formed_kernel = fmt::format(kernel_template,
                                            std::to_string(blocksize),
                                            std::to_string(use_pipe),
                                            std::to_string(pipeline_stages),
                                            std::to_string(launch_sim_blocks),
                                            full_jit_inputs,
                                            full_jit_declarations,
                                            full_jit_memops,
                                            full_jit_operations,
                                            full_jit_assign);


    const char *include_path = CFDARCO_JIT_INCLUDE_PATH;
    const char *eigeninclude_path = EIGEN3_INCLUDE_DIRS;
    const char *rmm_include_path = EIGEN3_INCLUDE_DIRS;
    const char *env_name = "OCCA_INCLUDE_PATH";
#if defined(CFDARCHO_CUDA_ENABLE)
    occa::json kernelProps({
                                   {"okl/enabled",    false},
                                   {"compiler_flags", fmt::format(
                                           "-std=c++17 --expt-relaxed-constexpr -I{} -I{} -DCFDARCHO_CUDA_ENABLE -DCFDARCO_SKIP_RMM -diag-suppress 20012",
                                           include_path, eigeninclude_path)}
                           });
#elif defined(CFDARCHO_HIP_ENABLE)
    occa::json kernelProps({
           {"okl/enabled", false},
           {"compiler_flags", fmt::format("-std=c++17 -I{} -I{} -DCFDARCHO_HIP_ENABLE -DCFDARCO_SKIP_RMM -arch=sm_80", include_path, eigeninclude_path)}
   });
#else
    occa::json kernelProps({
           {"okl/enabled", false},
           {"compiler_flags", fmt::format("-std=c++17 -I{} -I{} -DCFDARCO_SKIP_RMM -diag-suppress 20012", include_path, eigeninclude_path)}
   });
#endif

    auto kern_name = fmt::format("cfdARCHOKernel", kern_id);
    compute_kernel = occa::buildKernelFromString(formed_kernel, kern_name, kernelProps);

    compute_kernel.setRunDims(nblocks, blocksize);


    std::cout << "Kernel " << kern_name << " build" << std::endl;
}

void KernelBuilder::run() {
    auto vec_jit_inputs = expression->get_jit_inputs();

    std::string full_jit_inputs;
    std::vector<std::string> all_ins = {};
    std::vector<std::tuple<std::string, void *, TypeEnum>> inputs_in_order = {};
    for (const auto &vec_jit_input: vec_jit_inputs) {
        if (std::find(all_ins.begin(), all_ins.end(), std::get<0>(vec_jit_input)) == all_ins.end()) {
            full_jit_inputs += std::get<0>(vec_jit_input) + ", ";
            all_ins.push_back(std::get<0>(vec_jit_input));
            inputs_in_order.push_back(vec_jit_input);
        }
    }

    auto assign_inputs = left_val->get_jit_inputs();
    for (const auto &vec_jit_input: assign_inputs) {
        if (std::find(all_ins.begin(), all_ins.end(), std::get<0>(vec_jit_input)) == all_ins.end()) {
            full_jit_inputs += std::get<0>(vec_jit_input) + ", ";
            all_ins.push_back(std::get<0>(vec_jit_input));
            inputs_in_order.push_back(vec_jit_input);
        }
    }

    compute_kernel.clearArgs();


    std::vector<occa::memory> mems{};
    for (int i = 0; i < inputs_in_order.size(); ++i) {
        auto &[arg_name, arg_ptr, arg_type] = inputs_in_order.at(i);

        if (arg_type == TypeEnum::ConstDoublePointer) {
            const auto *mtrx_ptr = static_cast<CudaDataMatrixD *>(arg_ptr);
            const auto *raw_ptr = mtrx_ptr->data.get();
            occa::memory o_a;
            size_t ssize_ = mtrx_ptr->_size;
            if (ssize_ < 0 || ssize_ > left_val->mesh->_num_nodes * 10) {
                ssize_ = left_val->mesh->_num_nodes;
                raw_ptr = static_cast<CudaMesh3D *>(left_val->mesh)->_normal_x_cu.data.get();
            }
            o_a = occa::wrapMemory<float>(raw_ptr, ssize_);
            mems.push_back(o_a);
            compute_kernel.pushArg(mems.at(mems.size() - 1));
        }

        if (arg_type == TypeEnum::DoublePointer) {
            auto *mtrx_ptr = static_cast<CudaDataMatrixD *>(arg_ptr);
            auto *raw_ptr = mtrx_ptr->data.get();
            occa::memory o_a;
            size_t ssize_ = mtrx_ptr->_size;
            if (ssize_ < 0 || ssize_ > left_val->mesh->_num_nodes * 10) {
                ssize_ = left_val->mesh->_num_nodes;
                raw_ptr = static_cast<CudaMesh3D *>(left_val->mesh)->_normal_x_cu.data.get();
            }
            o_a = occa::wrapMemory<float>(raw_ptr, ssize_);
            mems.push_back(o_a);
            compute_kernel.pushArg(mems.at(mems.size() - 1));
        }

        if (arg_type == TypeEnum::ConstSizeTPointer) {
            const auto *mtrx_ptr = static_cast<CudaDataMatrix<size_t> *>(arg_ptr);
            const auto *raw_ptr = mtrx_ptr->data.get();
            occa::memory o_a;
            size_t ssize_ = mtrx_ptr->_size;
            if (ssize_ < 0 || ssize_ > left_val->mesh->_num_nodes * 10) {
                ssize_ = left_val->mesh->_num_nodes;
                raw_ptr = static_cast<CudaMesh3D *>(left_val->mesh)->_ids_cu.data.get();
            }
            o_a = occa::wrapMemory<size_t>(raw_ptr, ssize_);
            mems.push_back(o_a);
            compute_kernel.pushArg(mems.at(mems.size() - 1));
        }

        if (arg_type == TypeEnum::ConstUint32TPointer) {
            const auto *mtrx_ptr = static_cast<CudaDataMatrix<uint32_t> *>(arg_ptr);
            const auto *raw_ptr = mtrx_ptr->data.get();
            occa::memory o_a;
            size_t ssize_ = mtrx_ptr->_size;
            if (ssize_ < 0 || ssize_ > left_val->mesh->_num_nodes * 10) {
                ssize_ = left_val->mesh->_num_nodes;
                raw_ptr = static_cast<CudaMesh3D *>(left_val->mesh)->_ids32_cu.data.get();
            }
            o_a = occa::wrapMemory<uint32_t>(raw_ptr, ssize_);
            mems.push_back(o_a);
            compute_kernel.pushArg(mems.at(mems.size() - 1));
        }

        if (arg_type == TypeEnum::SizeT) {
            const auto *mtrx_ptr = static_cast<const size_t *>(arg_ptr);
            const auto raw_val = *mtrx_ptr;
            compute_kernel.pushArg(raw_val);
        }

        if (arg_type == TypeEnum::Uint32T) {
            const auto *mtrx_ptr = static_cast<const uint32_t *>(arg_ptr);
            const auto raw_val = *mtrx_ptr;
            compute_kernel.pushArg(raw_val);
        }

        if (arg_type == TypeEnum::Double) {
            const auto *mtrx_ptr = static_cast<const float *>(arg_ptr);
            const auto raw_val = *mtrx_ptr;
            compute_kernel.pushArg(raw_val);
        }

        if (arg_type == TypeEnum::Boolean) {
            const auto *mtrx_ptr = static_cast<const bool *>(arg_ptr);
            const auto raw_val = *mtrx_ptr;
            compute_kernel.pushArg(raw_val);
        }


    }

    compute_kernel.pushArg(expression->mesh->_num_nodes);

    sync_device();
    compute_kernel.run();
    sync_device();
}

#endif
