/*!
 * \file hlld_cuda.cu
 * \author Robert 'Bob' Caddy (rvc@pitt.edu)
 * \brief Contains the implementation of the HLLD solver from Miyoshi & Kusano 2005
 * "A multi-state HLL approximate Riemann solver for ideal magnetohydrodynamics",
 * hereafter referred to as M&K 2005
 *
*/

// External Includes

// Local Includes
#include "../utils/gpu.hpp"
#include "../global/global.h"
#include "../global/global_cuda.h"
#include "../utils/hydro_utilities.h"
#include "../utils/mhd_utilities.h"
#include "../riemann_solvers/hlld_cuda.h"
#include "../utils/cuda_utilities.h"
#include "../utils/math_utilities.h"
#include "../grid/grid_enum.h"

#ifdef DE //PRESSURE_DE
    #include "../utils/hydro_utilities.h"
#endif // DE

#ifdef CUDA

#ifdef MHD
namespace mhd
{
    // =========================================================================
    __global__ void Calculate_HLLD_Fluxes_CUDA(Real *dev_bounds_L,
                                               Real *dev_bounds_R,
                                               Real *dev_magnetic_face,
                                               Real *dev_flux,
                                               int nx,
                                               int ny,
                                               int nz,
                                               int n_ghost,
                                               Real gamma,
                                               int direction,
                                               int n_fields)
    {
        // get a thread index
        int blockId  = blockIdx.x + blockIdx.y*gridDim.x;
        int threadId = threadIdx.x + blockId * blockDim.x;
        int xid, yid, zid;
        cuda_utilities::compute3DIndices(threadId, nx, ny, xid, yid, zid);

        // Number of cells
        int n_cells = nx*ny*nz;

        // Offsets & indices
        int o1, o2, o3;
        if (direction==0) {o1 = 1; o2 = 2; o3 = 3;}
        if (direction==1) {o1 = 2; o2 = 3; o3 = 1;}
        if (direction==2) {o1 = 3; o2 = 1; o3 = 2;}

        // Thread guard to avoid overrun
        if (xid < nx and
            yid < ny and
            zid < nz)
        {
            // ============================
            // Retrieve conserved variables
            // ============================
            // The magnetic field in the X-direction
            Real magneticX = dev_magnetic_face[threadId];

            // Left interface
            Real densityL   = dev_bounds_L[threadId];
            Real momentumXL = dev_bounds_L[threadId + n_cells * o1];
            Real momentumYL = dev_bounds_L[threadId + n_cells * o2];
            Real momentumZL = dev_bounds_L[threadId + n_cells * o3];
            Real energyL    = dev_bounds_L[threadId + n_cells * 4];
            Real magneticYL = dev_bounds_L[threadId + n_cells * (grid_enum::Q_x_magnetic_y)];
            Real magneticZL = dev_bounds_L[threadId + n_cells * (grid_enum::Q_x_magnetic_z)];

            #ifdef SCALAR
                Real scalarConservedL[NSCALARS];
                for (int i=0; i<NSCALARS; i++)
                {
                    scalarConservedL[i] = dev_bounds_L[threadId + n_cells * (5+i)];
                }
            #endif // SCALAR
            #ifdef DE
                Real thermalEnergyConservedL = dev_bounds_L[threadId + n_cells * (n_fields-1)];
            #endif // DE

            // Right interface
            Real densityR   = dev_bounds_R[threadId];
            Real momentumXR = dev_bounds_R[threadId + n_cells * o1];
            Real momentumYR = dev_bounds_R[threadId + n_cells * o2];
            Real momentumZR = dev_bounds_R[threadId + n_cells * o3];
            Real energyR    = dev_bounds_R[threadId + n_cells * 4];
            Real magneticYR = dev_bounds_R[threadId + n_cells * (grid_enum::Q_x_magnetic_y)];
            Real magneticZR = dev_bounds_R[threadId + n_cells * (grid_enum::Q_x_magnetic_z)];

            #ifdef SCALAR
                Real scalarConservedR[NSCALARS];
                for (int i=0; i<NSCALARS; i++)
                {
                    scalarConservedR[i] = dev_bounds_R[threadId + n_cells * (5+i)];
                }
            #endif // SCALAR
            #ifdef DE
                Real thermalEnergyConservedR = dev_bounds_R[threadId + n_cells * (n_fields-1)];
            #endif // DE

            // Check for unphysical values
            densityL = fmax(densityL, (Real) TINY_NUMBER);
            densityR = fmax(densityR, (Real) TINY_NUMBER);
            energyL  = fmax(energyL,  (Real) TINY_NUMBER);
            energyR  = fmax(energyR,  (Real) TINY_NUMBER);

            // ============================
            // Compute primitive variables
            // ============================
            // Left interface
            Real const velocityXL = momentumXL / densityL;
            Real const velocityYL = momentumYL / densityL;
            Real const velocityZL = momentumZL / densityL;

            #ifdef DE //PRESSURE_DE
                Real energyNonThermal =   hydro_utilities::Calc_Kinetic_Energy_From_Velocity(densityL, velocityXL, velocityYL, velocityZL)
                                        + mhd::utils::computeMagneticEnergy(magneticX, magneticYL, magneticZL);

                Real const gasPressureL   = fmax(hydro_utilities::Get_Pressure_From_DE(energyL,
                                                                      energyL - energyNonThermal,
                                                                      thermalEnergyConservedL,
                                                                      gamma),
                                                 (Real) TINY_NUMBER);
            #else
                // Note that this function does the positive pressure check
                // internally
                Real const gasPressureL  = mhd::utils::computeGasPressure(energyL,
                                                                        densityL,
                                                                        momentumXL,
                                                                        momentumYL,
                                                                        momentumZL,
                                                                        magneticX,
                                                                        magneticYL,
                                                                        magneticZL,
                                                                        gamma);
            #endif //PRESSURE_DE

            Real const totalPressureL = mhd::utils::computeTotalPressure(gasPressureL,
                                                                       magneticX,
                                                                       magneticYL,
                                                                       magneticZL);

            // Right interface
            Real const velocityXR = momentumXR / densityR;
            Real const velocityYR = momentumYR / densityR;
            Real const velocityZR = momentumZR / densityR;

            #ifdef DE //PRESSURE_DE
                energyNonThermal =   hydro_utilities::Calc_Kinetic_Energy_From_Velocity(densityR, velocityXR, velocityYR, velocityZR)
                                   + mhd::utils::computeMagneticEnergy(magneticX, magneticYR, magneticZR);

                Real const gasPressureR   = fmax(hydro_utilities::Get_Pressure_From_DE(energyR,
                                                                      energyR - energyNonThermal,
                                                                      thermalEnergyConservedR,
                                                                      gamma),
                                                 (Real) TINY_NUMBER);
            #else
                // Note that this function does the positive pressure check
                // internally
                Real const gasPressureR  = mhd::utils::computeGasPressure(energyR,
                                                                  densityR,
                                                                  momentumXR,
                                                                  momentumYR,
                                                                  momentumZR,
                                                                  magneticX,
                                                                  magneticYR,
                                                                  magneticZR,
                                                                  gamma);
            #endif //PRESSURE_DE

            Real const totalPressureR = mhd::utils::computeTotalPressure(gasPressureR,
                                                                 magneticX,
                                                                 magneticYR,
                                                                 magneticZR);

            // Compute the approximate wave speeds and density in the star
            // regions
            Real speedL, speedR, speedM, speedStarL, speedStarR, densityStarL, densityStarR;
            mhd::_internal::_approximateWaveSpeeds(densityL,
                                                  momentumXL,
                                                  momentumYL,
                                                  momentumZL,
                                                  velocityXL,
                                                  velocityYL,
                                                  velocityZL,
                                                  gasPressureL,
                                                  totalPressureL,
                                                  magneticX,
                                                  magneticYL,
                                                  magneticZL,
                                                  densityR,
                                                  momentumXR,
                                                  momentumYR,
                                                  momentumZR,
                                                  velocityXR,
                                                  velocityYR,
                                                  velocityZR,
                                                  gasPressureR,
                                                  totalPressureR,
                                                  magneticYR,
                                                  magneticZR,
                                                  gamma,
                                                  speedL,
                                                  speedR,
                                                  speedM,
                                                  speedStarL,
                                                  speedStarR,
                                                  densityStarL,
                                                  densityStarR);

            // =================================================================
            // Compute the fluxes in the non-star states
            // =================================================================
            // Left state
            Real densityFluxL, momentumFluxXL, momentumFluxYL, momentumFluxZL,
                 magneticFluxYL, magneticFluxZL, energyFluxL;
            mhd::_internal::_nonStarFluxes(momentumXL,
                                          velocityXL,
                                          velocityYL,
                                          velocityZL,
                                          totalPressureL,
                                          energyL,
                                          magneticX,
                                          magneticYL,
                                          magneticZL,
                                          densityFluxL,
                                          momentumFluxXL,
                                          momentumFluxYL,
                                          momentumFluxZL,
                                          magneticFluxYL,
                                          magneticFluxZL,
                                          energyFluxL);

            // If we're in the L state then assign fluxes and return.
            // In this state the flow is supersonic
            // M&K 2005 equation 66
            if (speedL >= 0.0)
            {
                mhd::_internal::_returnFluxes(threadId, o1, o2, o3, n_cells,
                                             dev_flux,
                                             densityFluxL,
                                             momentumFluxXL, momentumFluxYL, momentumFluxZL,
                                             energyFluxL,
                                             magneticFluxYL, magneticFluxZL);
                #ifdef SCALAR
                    for (int i=0; i<NSCALARS; i++)
                    {
                        dev_flux[(5+i)*n_cells+threadId]  = (scalarConservedL[i] / densityL) * densityFluxL;
                    }
                #endif  // SCALAR
                #ifdef DE
                    dev_flux[(n_fields-1)*n_cells+threadId]  = (thermalEnergyConservedL / densityL) * densityFluxL;
                #endif  // DE
                return;
            }
            // Right state
            Real densityFluxR, momentumFluxXR, momentumFluxYR, momentumFluxZR,
                 magneticFluxYR, magneticFluxZR, energyFluxR;
            mhd::_internal::_nonStarFluxes(momentumXR,
                                          velocityXR,
                                          velocityYR,
                                          velocityZR,
                                          totalPressureR,
                                          energyR,
                                          magneticX,
                                          magneticYR,
                                          magneticZR,
                                          densityFluxR,
                                          momentumFluxXR,
                                          momentumFluxYR,
                                          momentumFluxZR,
                                          magneticFluxYR,
                                          magneticFluxZR,
                                          energyFluxR);

            // If we're in the R state then assign fluxes and return.
            // In this state the flow is supersonic
            // M&K 2005 equation 66
            if (speedR <= 0.0)
            {
                mhd::_internal::_returnFluxes(threadId, o1, o2, o3, n_cells,
                                             dev_flux,
                                             densityFluxR,
                                             momentumFluxXR, momentumFluxYR, momentumFluxZR,
                                             energyFluxR,
                                             magneticFluxYR, magneticFluxZR);
                #ifdef SCALAR
                    for (int i=0; i<NSCALARS; i++)
                    {
                        dev_flux[(5+i)*n_cells+threadId]  = (scalarConservedR[i] / densityR) * densityFluxR;
                    }
                #endif  // SCALAR
                #ifdef DE
                    dev_flux[(n_fields-1)*n_cells+threadId]  = (thermalEnergyConservedR / densityR) * densityFluxR;
                #endif  // DE
                return;
            }

            // =================================================================
            // Compute the fluxes in the star states
            // =================================================================
            // Shared quantity
            // note that velocityStarX = speedM
            // M&K 2005 equation 23, might need to switch to eqn. 41 in the
            // future though they should produce identical results
            Real totalPressureStar = totalPressureL + densityL
                                                      * (speedL - velocityXL)
                                                      * (speedM - velocityXL);

            // Left star state
            Real velocityStarYL, velocityStarZL,
                 energyStarL, magneticStarYL, magneticStarZL,
                 densityStarFluxL,
                 momentumStarFluxXL, momentumStarFluxYL, momentumStarFluxZL,
                 magneticStarFluxYL, magneticStarFluxZL, energyStarFluxL;
            mhd::_internal::_starFluxes(speedM,
                                       speedL,
                                       densityL,
                                       velocityXL,
                                       velocityYL,
                                       velocityZL,
                                       momentumXL,
                                       momentumYL,
                                       momentumZL,
                                       energyL,
                                       totalPressureL,
                                       magneticX,
                                       magneticYL,
                                       magneticZL,
                                       densityStarL,
                                       totalPressureStar,
                                       densityFluxL,
                                       momentumFluxXL,
                                       momentumFluxYL,
                                       momentumFluxZL,
                                       energyFluxL,
                                       magneticFluxYL,
                                       magneticFluxZL,
                                       velocityStarYL,
                                       velocityStarZL,
                                       energyStarL,
                                       magneticStarYL,
                                       magneticStarZL,
                                       densityStarFluxL,
                                       momentumStarFluxXL,
                                       momentumStarFluxYL,
                                       momentumStarFluxZL,
                                       energyStarFluxL,
                                       magneticStarFluxYL,
                                       magneticStarFluxZL);

            // If we're in the L* state then assign fluxes and return.
            // In this state the flow is subsonic
            // M&K 2005 equation 66
            if (speedStarL >= 0.0)
            {
                mhd::_internal::_returnFluxes(threadId, o1, o2, o3, n_cells,
                                             dev_flux,
                                             densityStarFluxL,
                                             momentumStarFluxXL, momentumStarFluxYL, momentumStarFluxZL,
                                             energyStarFluxL,
                                             magneticStarFluxYL, magneticStarFluxZL);
                #ifdef SCALAR
                    for (int i=0; i<NSCALARS; i++)
                    {
                        dev_flux[(5+i)*n_cells+threadId] = (scalarConservedL[i] / densityL) * densityStarFluxL;
                    }
                #endif  // SCALAR
                #ifdef DE
                    dev_flux[(n_fields-1)*n_cells+threadId]  = (thermalEnergyConservedL / densityL) * densityStarFluxL;
                #endif  // DE
                return;
            }

            // Right star state
            Real velocityStarYR, velocityStarZR,
                 energyStarR, magneticStarYR, magneticStarZR,
                 densityStarFluxR,
                 momentumStarFluxXR, momentumStarFluxYR, momentumStarFluxZR,
                 magneticStarFluxYR, magneticStarFluxZR, energyStarFluxR;
            mhd::_internal::_starFluxes(speedM,
                                       speedR,
                                       densityR,
                                       velocityXR,
                                       velocityYR,
                                       velocityZR,
                                       momentumXR,
                                       momentumYR,
                                       momentumZR,
                                       energyR,
                                       totalPressureR,
                                       magneticX,
                                       magneticYR,
                                       magneticZR,
                                       densityStarR,
                                       totalPressureStar,
                                       densityFluxR,
                                       momentumFluxXR,
                                       momentumFluxYR,
                                       momentumFluxZR,
                                       energyFluxR,
                                       magneticFluxYR,
                                       magneticFluxZR,
                                       velocityStarYR,
                                       velocityStarZR,
                                       energyStarR,
                                       magneticStarYR,
                                       magneticStarZR,
                                       densityStarFluxR,
                                       momentumStarFluxXR,
                                       momentumStarFluxYR,
                                       momentumStarFluxZR,
                                       energyStarFluxR,
                                       magneticStarFluxYR,
                                       magneticStarFluxZR);

            // If we're in the R* state then assign fluxes and return.
            // In this state the flow is subsonic
            // M&K 2005 equation 66
            if (speedStarR <= 0.0)
            {
                mhd::_internal::_returnFluxes(threadId, o1, o2, o3, n_cells,
                                             dev_flux,
                                             densityStarFluxR,
                                             momentumStarFluxXR, momentumStarFluxYR, momentumStarFluxZR,
                                             energyStarFluxR,
                                             magneticStarFluxYR, magneticStarFluxZR);
                #ifdef SCALAR
                    for (int i=0; i<NSCALARS; i++)
                    {
                        dev_flux[(5+i)*n_cells+threadId] = (scalarConservedR[i] / densityR) * densityStarFluxR;
                    }
                #endif  // SCALAR
                #ifdef DE
                    dev_flux[(n_fields-1)*n_cells+threadId]  = (thermalEnergyConservedR / densityR) * densityStarFluxR;
                #endif  // DE
                return;
            }

            // =================================================================
            // Compute the fluxes in the double star states
            // =================================================================
            Real velocityDoubleStarY, velocityDoubleStarZ,
                 magneticDoubleStarY, magneticDoubleStarZ,
                 energyDoubleStarL, energyDoubleStarR;
            mhd::_internal::_doubleStarState(speedM,
                                            magneticX,
                                            totalPressureStar,
                                            densityStarL,
                                            velocityStarYL,
                                            velocityStarZL,
                                            energyStarL,
                                            magneticStarYL,
                                            magneticStarZL,
                                            densityStarR,
                                            velocityStarYR,
                                            velocityStarZR,
                                            energyStarR,
                                            magneticStarYR,
                                            magneticStarZR,
                                            velocityDoubleStarY,
                                            velocityDoubleStarZ,
                                            magneticDoubleStarY,
                                            magneticDoubleStarZ,
                                            energyDoubleStarL,
                                            energyDoubleStarR);

            // Compute and return L** fluxes
            // M&K 2005 equation 66
            if (speedM >= 0.0)
            {
                Real momentumDoubleStarFluxX, momentumDoubleStarFluxY, momentumDoubleStarFluxZ,
                     energyDoubleStarFlux,
                     magneticDoubleStarFluxY, magneticDoubleStarFluxZ;
                mhd::_internal::_doubleStarFluxes(speedStarL,
                                                 momentumStarFluxXL,
                                                 momentumStarFluxYL,
                                                 momentumStarFluxZL,
                                                 energyStarFluxL,
                                                 magneticStarFluxYL,
                                                 magneticStarFluxZL,
                                                 densityStarL,
                                                 speedM,
                                                 velocityStarYL,
                                                 velocityStarZL,
                                                 energyStarL,
                                                 magneticStarYL,
                                                 magneticStarZL,
                                                 speedM,
                                                 velocityDoubleStarY,
                                                 velocityDoubleStarZ,
                                                 energyDoubleStarL,
                                                 magneticDoubleStarY,
                                                 magneticDoubleStarZ,
                                                 momentumDoubleStarFluxX,
                                                 momentumDoubleStarFluxY,
                                                 momentumDoubleStarFluxZ,
                                                 energyDoubleStarFlux,
                                                 magneticDoubleStarFluxY,
                                                 magneticDoubleStarFluxZ);

                mhd::_internal::_returnFluxes(threadId, o1, o2, o3, n_cells,
                                             dev_flux,
                                             densityStarFluxL,
                                             momentumDoubleStarFluxX, momentumDoubleStarFluxY, momentumDoubleStarFluxZ,
                                             energyDoubleStarFlux,
                                             magneticDoubleStarFluxY, magneticDoubleStarFluxZ);

                #ifdef SCALAR
                    // Return the passive scalar fluxes
                    for (int i=0; i<NSCALARS; i++)
                    {
                        dev_flux[(5+i)*n_cells+threadId] = (scalarConservedL[i] / densityL) * densityStarFluxL;
                    }
                #endif // SCALAR
                #ifdef DE
                    dev_flux[(n_fields-1)*n_cells+threadId]  = (thermalEnergyConservedL / densityL) * densityStarFluxL;
                #endif  // DE
                return;
            }
            // Compute and return R** fluxes
            // M&K 2005 equation 66
            else if (speedStarR >= 0.0)
            {
                Real momentumDoubleStarFluxX, momentumDoubleStarFluxY, momentumDoubleStarFluxZ,
                     energyDoubleStarFlux,
                     magneticDoubleStarFluxY, magneticDoubleStarFluxZ;
                mhd::_internal::_doubleStarFluxes(speedStarR,
                                                 momentumStarFluxXR,
                                                 momentumStarFluxYR,
                                                 momentumStarFluxZR,
                                                 energyStarFluxR,
                                                 magneticStarFluxYR,
                                                 magneticStarFluxZR,
                                                 densityStarR,
                                                 speedM,
                                                 velocityStarYR,
                                                 velocityStarZR,
                                                 energyStarR,
                                                 magneticStarYR,
                                                 magneticStarZR,
                                                 speedM,
                                                 velocityDoubleStarY,
                                                 velocityDoubleStarZ,
                                                 energyDoubleStarR,
                                                 magneticDoubleStarY,
                                                 magneticDoubleStarZ,
                                                 momentumDoubleStarFluxX,
                                                 momentumDoubleStarFluxY,
                                                 momentumDoubleStarFluxZ,
                                                 energyDoubleStarFlux,
                                                 magneticDoubleStarFluxY,
                                                 magneticDoubleStarFluxZ);

                mhd::_internal::_returnFluxes(threadId, o1, o2, o3, n_cells,
                                             dev_flux,
                                             densityStarFluxR,
                                             momentumDoubleStarFluxX, momentumDoubleStarFluxY, momentumDoubleStarFluxZ,
                                             energyDoubleStarFlux,
                                             magneticDoubleStarFluxY, magneticDoubleStarFluxZ);

                #ifdef SCALAR
                    // Return the passive scalar fluxes
                    for (int i=0; i<NSCALARS; i++)
                    {
                        dev_flux[(5+i)*n_cells+threadId] = (scalarConservedR[i] / densityR) * densityStarFluxR;
                    }
                #endif // SCALAR
                #ifdef DE
                    dev_flux[(n_fields-1)*n_cells+threadId]  = (thermalEnergyConservedR / densityR) * densityStarFluxR;
                #endif  // DE
                return;
            }
        } // End thread guard
    };
    // =========================================================================

    namespace _internal
    {
        // =====================================================================
        __device__ __host__ void _approximateWaveSpeeds(Real const &densityL,
                                                        Real const &momentumXL,
                                                        Real const &momentumYL,
                                                        Real const &momentumZL,
                                                        Real const &velocityXL,
                                                        Real const &velocityYL,
                                                        Real const &velocityZL,
                                                        Real const &gasPressureL,
                                                        Real const &totalPressureL,
                                                        Real const &magneticX,
                                                        Real const &magneticYL,
                                                        Real const &magneticZL,
                                                        Real const &densityR,
                                                        Real const &momentumXR,
                                                        Real const &momentumYR,
                                                        Real const &momentumZR,
                                                        Real const &velocityXR,
                                                        Real const &velocityYR,
                                                        Real const &velocityZR,
                                                        Real const &gasPressureR,
                                                        Real const &totalPressureR,
                                                        Real const &magneticYR,
                                                        Real const &magneticZR,
                                                        Real const &gamma,
                                                        Real &speedL,
                                                        Real &speedR,
                                                        Real &speedM,
                                                        Real &speedStarL,
                                                        Real &speedStarR,
                                                        Real &densityStarL,
                                                        Real &densityStarR)
        {
            // Get the fast magnetosonic wave speeds
            Real magSonicL = mhd::utils::fastMagnetosonicSpeed(densityL,
                                                             gasPressureL,
                                                             magneticX,
                                                             magneticYL,
                                                             magneticZL,
                                                             gamma);
            Real magSonicR = mhd::utils::fastMagnetosonicSpeed(densityR,
                                                             gasPressureR,
                                                             magneticX,
                                                             magneticYR,
                                                             magneticZR,
                                                             gamma);

            // Compute the S_L and S_R wave speeds.
            // Version suggested by Miyoshi & Kusano 2005 and used in Athena
            // M&K 2005 equation 67
            Real magSonicMax = fmax(magSonicL, magSonicR);
            speedL = fmin(velocityXL, velocityXR) - magSonicMax;
            speedR = fmax(velocityXL, velocityXR) + magSonicMax;

            // Compute the S_M wave speed
            // M&K 2005 equation 38
            speedM = // Numerator
                          ( momentumXR * (speedR - velocityXR)
                          - momentumXL * (speedL - velocityXL)
                          + (totalPressureL - totalPressureR))
                          /
                          // Denominator
                          ( densityR * (speedR - velocityXR)
                          - densityL * (speedL - velocityXL));

            // Compute the densities in the star state
            // M&K 2005 equation 43
            densityStarL = densityL * (speedL - velocityXL) / (speedL - speedM);
            densityStarR = densityR * (speedR - velocityXR) / (speedR - speedM);

            // Compute the S_L^* and S_R^* wave speeds
            // M&K 2005 equation 51
            speedStarL = speedM - mhd::utils::alfvenSpeed(magneticX, densityStarL);
            speedStarR = speedM + mhd::utils::alfvenSpeed(magneticX, densityStarR);
        }
        // =====================================================================

        // =====================================================================
        __device__ __host__ void _nonStarFluxes(Real const &momentumX,
                                                Real const &velocityX,
                                                Real const &velocityY,
                                                Real const &velocityZ,
                                                Real const &totalPressure,
                                                Real const &energy,
                                                Real const &magneticX,
                                                Real const &magneticY,
                                                Real const &magneticZ,
                                                Real &densityFlux,
                                                Real &momentumFluxX,
                                                Real &momentumFluxY,
                                                Real &momentumFluxZ,
                                                Real &magneticFluxY,
                                                Real &magneticFluxZ,
                                                Real &energyFlux)
        {
            // M&K 2005 equation 2
            densityFlux   = momentumX;

            momentumFluxX = momentumX * velocityX + totalPressure - magneticX * magneticX;
            momentumFluxY = momentumX * velocityY - magneticX * magneticY;
            momentumFluxZ = momentumX * velocityZ - magneticX * magneticZ;

            magneticFluxY = magneticY * velocityX - magneticX * velocityY;
            magneticFluxZ = magneticZ * velocityX - magneticX * velocityZ;

            // Group transverse terms for FP associative symmetry
            energyFlux    = velocityX * (energy + totalPressure) - magneticX
                            * (velocityX * magneticX
                               + ((velocityY * magneticY)
                               + (velocityZ * magneticZ)));
        }
        // =====================================================================

        // =====================================================================
        __device__ __host__  void _returnFluxes(int const &threadId,
                                                int const &o1,
                                                int const &o2,
                                                int const &o3,
                                                int const &n_cells,
                                                Real *dev_flux,
                                                Real const &densityFlux,
                                                Real const &momentumFluxX,
                                                Real const &momentumFluxY,
                                                Real const &momentumFluxZ,
                                                Real const &energyFlux,
                                                Real const &magneticFluxY,
                                                Real const &magneticFluxZ)
        {
            dev_flux[threadId]                            = densityFlux;
            dev_flux[threadId + n_cells * o1]             = momentumFluxX;
            dev_flux[threadId + n_cells * o2]             = momentumFluxY;
            dev_flux[threadId + n_cells * o3]             = momentumFluxZ;
            dev_flux[threadId + n_cells * 4]              = energyFlux;
            dev_flux[threadId + n_cells * (grid_enum::fluxX_magnetic_z)] = magneticFluxY;
            dev_flux[threadId + n_cells * (grid_enum::fluxX_magnetic_y)] = magneticFluxZ;
        }
        // =====================================================================

        // =====================================================================
        __device__ __host__ void _starFluxes(Real const &speedM,
                                             Real const &speedSide,
                                             Real const &density,
                                             Real const &velocityX,
                                             Real const &velocityY,
                                             Real const &velocityZ,
                                             Real const &momentumX,
                                             Real const &momentumY,
                                             Real const &momentumZ,
                                             Real const &energy,
                                             Real const &totalPressure,
                                             Real const &magneticX,
                                             Real const &magneticY,
                                             Real const &magneticZ,
                                             Real const &densityStar,
                                             Real const &totalPressureStar,
                                             Real const &densityFlux,
                                             Real const &momentumFluxX,
                                             Real const &momentumFluxY,
                                             Real const &momentumFluxZ,
                                             Real const &energyFlux,
                                             Real const &magneticFluxY,
                                             Real const &magneticFluxZ,
                                             Real &velocityStarY,
                                             Real &velocityStarZ,
                                             Real &energyStar,
                                             Real &magneticStarY,
                                             Real &magneticStarZ,
                                             Real &densityStarFlux,
                                             Real &momentumStarFluxX,
                                             Real &momentumStarFluxY,
                                             Real &momentumStarFluxZ,
                                             Real &energyStarFlux,
                                             Real &magneticStarFluxY,
                                             Real &magneticStarFluxZ)
        {
            // Check for and handle the degenerate case
            // Explained at the top of page 326 in M&K 2005
            if (fabs(density * (speedSide - velocityX)
                             * (speedSide - speedM)
                             - (magneticX * magneticX))
                < totalPressureStar * mhd::_internal::_hlldSmallNumber)
            {
                velocityStarY = velocityY;
                velocityStarZ = velocityZ;
                magneticStarY = magneticY;
                magneticStarZ = magneticZ;
            }
            else
            {
                // Denominator for M&K 2005 equations 44-47
                Real const denom = density * (speedSide - velocityX)
                                           * (speedSide - speedM)
                                           - (magneticX * magneticX);

                // Compute the velocity and magnetic field in the star state
                // M&K 2005 equations 44 & 46
                Real coef     = magneticX  * (speedM - velocityX) / denom;
                velocityStarY = velocityY - magneticY * coef;
                velocityStarZ = velocityZ - magneticZ * coef;

                // M&K 2005 equations 45 & 47
                Real tmpPower = (speedSide - velocityX);
                tmpPower      = tmpPower * tmpPower;
                coef          = (density * tmpPower - (magneticX * magneticX)) / denom;
                magneticStarY = magneticY * coef;
                magneticStarZ = magneticZ * coef;
            }

            // M&K 2005 equation 48
            energyStar = ( energy * (speedSide - velocityX)
                        - totalPressure * velocityX
                        + totalPressureStar * speedM
                        + magneticX * (math_utils::dotProduct(velocityX, velocityY, velocityZ, magneticX, magneticY, magneticZ)
                                     - math_utils::dotProduct(speedM, velocityStarY, velocityStarZ, magneticX, magneticStarY, magneticStarZ)))
                        / (speedSide - speedM);

            // Now compute the star state fluxes
            // M&K 2005 equations 64
            densityStarFlux   = densityFlux   + speedSide * (densityStar - density);;
            momentumStarFluxX = momentumFluxX + speedSide * (densityStar * speedM - momentumX);;
            momentumStarFluxY = momentumFluxY + speedSide * (densityStar * velocityStarY - momentumY);;
            momentumStarFluxZ = momentumFluxZ + speedSide * (densityStar * velocityStarZ - momentumZ);;
            energyStarFlux    = energyFlux    + speedSide * (energyStar  - energy);
            magneticStarFluxY = magneticFluxY + speedSide * (magneticStarY - magneticY);
            magneticStarFluxZ = magneticFluxZ + speedSide * (magneticStarZ - magneticZ);
        }
        // =====================================================================

        // =====================================================================
        __device__ __host__ void _doubleStarState(Real const &speedM,
                                                  Real const &magneticX,
                                                  Real const &totalPressureStar,
                                                  Real const &densityStarL,
                                                  Real const &velocityStarYL,
                                                  Real const &velocityStarZL,
                                                  Real const &energyStarL,
                                                  Real const &magneticStarYL,
                                                  Real const &magneticStarZL,
                                                  Real const &densityStarR,
                                                  Real const &velocityStarYR,
                                                  Real const &velocityStarZR,
                                                  Real const &energyStarR,
                                                  Real const &magneticStarYR,
                                                  Real const &magneticStarZR,
                                                  Real &velocityDoubleStarY,
                                                  Real &velocityDoubleStarZ,
                                                  Real &magneticDoubleStarY,
                                                  Real &magneticDoubleStarZ,
                                                  Real &energyDoubleStarL,
                                                  Real &energyDoubleStarR)
        {
            // if Bx is zero then just return the star state
            // Explained at the top of page 328 in M&K 2005. Essentially when
            // magneticX is 0 this reduces to the HLLC solver
            if (magneticX < mhd::_internal::_hlldSmallNumber * totalPressureStar)
            {
                velocityDoubleStarY = velocityStarYL;
                velocityDoubleStarZ = velocityStarZL;
                magneticDoubleStarY = magneticStarYL;
                magneticDoubleStarZ = magneticStarZL;
                energyDoubleStarL   = energyStarL;
                energyDoubleStarR   = energyStarR;
            }
            else
            {
                // Setup some variables we'll need later
                Real sqrtDL = sqrt(densityStarL);
                Real sqrtDR = sqrt(densityStarR);
                Real inverseDensities = 1.0 / (sqrtDL + sqrtDR);
                Real magXSign = copysign(1.0, magneticX);

                // All we need to do now is compute the transverse velocities
                // and magnetic fields along with the energy

                // Double Star velocities
                // M&K 2005 equations 59 & 60
                velocityDoubleStarY = inverseDensities * (sqrtDL * velocityStarYL
                                      + sqrtDR * velocityStarYR
                                      + magXSign * (magneticStarYR - magneticStarYL));
                velocityDoubleStarZ = inverseDensities * (sqrtDL * velocityStarZL
                                      + sqrtDR * velocityStarZR
                                      + magXSign * (magneticStarZR - magneticStarZL));

                // Double star magnetic fields
                // M&K 2005 equations 61 & 62
                magneticDoubleStarY = inverseDensities * (sqrtDL * magneticStarYR
                                      + sqrtDR * magneticStarYL
                                      + magXSign * (sqrtDL * sqrtDR) * (velocityStarYR - velocityStarYL));
                magneticDoubleStarZ = inverseDensities * (sqrtDL * magneticStarZR
                                      + sqrtDR * magneticStarZL
                                      + magXSign * (sqrtDL * sqrtDR) * (velocityStarZR - velocityStarZL));

                // Double star energy
                Real velDblStarDotMagDblStar = math_utils::dotProduct(speedM,
                                                                          velocityDoubleStarY,
                                                                          velocityDoubleStarZ,
                                                                          magneticX,
                                                                          magneticDoubleStarY,
                                                                          magneticDoubleStarZ);
                // M&K 2005 equation 63
                energyDoubleStarL = energyStarL - sqrtDL * magXSign
                    * (math_utils::dotProduct(speedM, velocityStarYL, velocityStarZL, magneticX, magneticStarYL, magneticStarZL)
                    - velDblStarDotMagDblStar);
                energyDoubleStarR = energyStarR + sqrtDR * magXSign
                    * (math_utils::dotProduct(speedM, velocityStarYR, velocityStarZR, magneticX, magneticStarYR, magneticStarZR)
                    - velDblStarDotMagDblStar);
            }
        }
        // =====================================================================

        // =====================================================================
        __device__ __host__ void _doubleStarFluxes(Real const &speedStarSide,
                                                   Real const &momentumStarFluxX,
                                                   Real const &momentumStarFluxY,
                                                   Real const &momentumStarFluxZ,
                                                   Real const &energyStarFlux,
                                                   Real const &magneticStarFluxY,
                                                   Real const &magneticStarFluxZ,
                                                   Real const &densityStar,
                                                   Real const &velocityStarX,
                                                   Real const &velocityStarY,
                                                   Real const &velocityStarZ,
                                                   Real const &energyStar,
                                                   Real const &magneticStarY,
                                                   Real const &magneticStarZ,
                                                   Real const &velocityDoubleStarX,
                                                   Real const &velocityDoubleStarY,
                                                   Real const &velocityDoubleStarZ,
                                                   Real const &energyDoubleStar,
                                                   Real const &magneticDoubleStarY,
                                                   Real const &magneticDoubleStarZ,
                                                   Real &momentumDoubleStarFluxX,
                                                   Real &momentumDoubleStarFluxY,
                                                   Real &momentumDoubleStarFluxZ,
                                                   Real &energyDoubleStarFlux,
                                                   Real &magneticDoubleStarFluxY,
                                                   Real &magneticDoubleStarFluxZ)
        {
            // M&K 2005 equation 65
            momentumDoubleStarFluxX = momentumStarFluxX + speedStarSide * (velocityDoubleStarX - velocityStarX) * densityStar;
            momentumDoubleStarFluxY = momentumStarFluxY + speedStarSide * (velocityDoubleStarY - velocityStarY) * densityStar;
            momentumDoubleStarFluxZ = momentumStarFluxZ + speedStarSide * (velocityDoubleStarZ - velocityStarZ) * densityStar;
            energyDoubleStarFlux    = energyStarFlux    + speedStarSide * (energyDoubleStar    - energyStar);
            magneticDoubleStarFluxY = magneticStarFluxY + speedStarSide * (magneticDoubleStarY - magneticStarY);
            magneticDoubleStarFluxZ = magneticStarFluxZ + speedStarSide * (magneticDoubleStarZ - magneticStarZ);
        }
        // =====================================================================

    } // mhd::_internal namespace
} // end namespace mhd
#endif // MHD
#endif // CUDA
