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

// ============================================================================
// PHASE FIELD PARAMETERS CONFIGURATION
// ============================================================================

namespace PhaseFieldParams {
    
    // Physical parameters for dendritic crystal growth
    struct PhysicalParams {
        // Interface and mobility parameters
        static constexpr float epsilon = 0.015f;        // Interface thickness parameter
        static constexpr float M = 1.0f;                // Mobility parameter
        static constexpr float alpha = 1.0f;            // Double-well potential parameter
        
        // Thermal parameters
        static constexpr float D = 0.05f;               // Thermal diffusivity
        static constexpr float Tm = 0.0f;               // Melting temperature
        static constexpr float L = 1.0f;                // Latent heat
        
        // Source parameters
        static constexpr float source_strength = 0.5f;  // Constant source strength
        static constexpr float source_radius = 0.1f;    // Source region radius
        
        // Initial condition parameters
        static constexpr float seed_radius = 0.1f;      // Initial seed radius
        static constexpr float transition_radius = 0.15f; // Transition region radius
        static constexpr float perturbation_radius = 0.25f; // Perturbation region radius
        static constexpr float perturbation_amplitude = 0.1f; // Perturbation amplitude
        static constexpr float seed_temperature = 0.1f; // Seed temperature (above melting)
        static constexpr float supercooling = -0.5f;    // Supercooled liquid temperature
    };
    
    // Simple phase field parameters (no temperature coupling)
    struct SimpleParams {
        // Interface and mobility parameters
        static constexpr float epsilon = 0.02f;         // Interface thickness parameter
        static constexpr float M = 1.0f;                // Mobility parameter
        
        // Source parameters
        static constexpr float source_strength = 0.3f;  // Constant source strength
        static constexpr float source_radius = 0.1f;    // Source region radius
        
        // Initial condition parameters
        static constexpr float seed_radius = 0.15f;     // Initial seed radius
        static constexpr float transition_radius = 0.2f; // Transition region radius
    };
    
    // Aggressive parameters for fast growth
    struct AggressiveParams {
        // Interface and mobility parameters
        static constexpr float epsilon = 0.01f;         // Sharp interface
        static constexpr float M = 3.0f;                // High mobility for fast growth
        static constexpr float alpha = 1.0f;            // Double-well potential parameter
        
        // Thermal parameters
        static constexpr float D = 0.01f;               // Very low thermal diffusivity
        static constexpr float Tm = 0.0f;               // Melting temperature
        static constexpr float L = 1.0f;                // Latent heat
        
        // Source parameters
        static constexpr float source_strength = 3.0f;  // Strong constant source
        static constexpr float source_radius = 0.15f;    // Source region radius
        
        // Initial condition parameters
        static constexpr float seed_radius = 0.05f;      // Initial seed radius
        static constexpr float transition_radius = 0.05f; // Transition region radius
        static constexpr float perturbation_radius = 0.05f; // Perturbation region radius
        static constexpr float perturbation_amplitude = 0.2f; // Strong perturbations
        static constexpr float seed_temperature = 10.0f; // High seed temperature
        static constexpr float supercooling = -0.1f;    // Supercooled liquid temperature
    };
    
    // Conservative parameters for stability
    struct ConservativeParams {
        // Interface and mobility parameters
        static constexpr float epsilon = 0.025f;        // Thick interface for stability
        static constexpr float M = 0.5f;                // Low mobility for stability
        static constexpr float alpha = 1.0f;            // Double-well potential parameter
        
        // Thermal parameters
        static constexpr float D = 0.1f;                // Higher thermal diffusivity
        static constexpr float Tm = 0.0f;               // Melting temperature
        static constexpr float L = 1.0f;                // Latent heat
        
        // Source parameters
        static constexpr float source_strength = 0.2f;  // Weak constant source
        static constexpr float source_radius = 0.1f;    // Source region radius
        
        // Initial condition parameters
        static constexpr float seed_radius = 0.12f;     // Larger seed for stability
        static constexpr float transition_radius = 0.18f; // Wider transition region
        static constexpr float perturbation_radius = 0.3f; // Larger perturbation region
        static constexpr float perturbation_amplitude = 0.05f; // Small perturbations
        static constexpr float seed_temperature = 0.05f; // Low seed temperature
        static constexpr float supercooling = -0.3f;    // Less supercooling
    };
    
    // Default parameter set (using PhysicalParams)
    using DefaultParams = AggressiveParams;
    
} // namespace PhaseFieldParams 