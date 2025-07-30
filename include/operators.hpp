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
#pragma once

#include "variables/dissipation.hpp"
#include "variables/dt_update.hpp"
#include "variables/interpolation.hpp"
#include "variables/spatial_first_derivative.hpp"
#include "variables/spatial_second_derivative.hpp"
#include "variables/sub_variables.hpp"
#include "variables/time_first_derivative.hpp"
#include "variables/time_second_derivative.hpp"
#include "variables/variable.hpp"
#include "boundary_conditions.hpp"
#include "cuda_boundary_conditions.hpp"
#include "variables/point_sourse.hpp"
