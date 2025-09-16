# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0

import numpy as np
import pandas as pd


import sys
import time
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from VeraGridEngine.Devices.Aggregation.rms_event import RmsEvent
from VeraGridEngine.Utils.Symbolic.symbolic import Const, Var
from VeraGridEngine.Utils.Symbolic.block_solver import BlockSolver
import VeraGridEngine.api as gce

# TODO: system with only 1 generator, 2 line, 1 load and 3 buses

# ----------------------------------------------------------------------------------------------------------------------
# Power flow
# ----------------------------------------------------------------------------------------------------------------------
# Load system
# TODO: be careful! this is _noshunt, such that the initialization it's easier because we have one device per bus. 
# In scripting_gridcal_hardcoded there are also shunt elements!



# Build system

t = Var("t")

grid = gce.MultiCircuit()
# SIMPLE SYSTEM 0: 1 GEN, 1 TRAFO, 1 LOAD
# # Buses
# bus1 = gce.Bus(name="Bus1", Vnom=20, is_slack=True)
# bus2 = gce.Bus(name="Bus2", Vnom=230)
# # bus3 = gce.Bus(name="Bus3", Vnom=230)
#
# grid.add_bus(bus1)
# grid.add_bus(bus2)
# # grid.add_bus(bus3)
#
# # Line
#
#
# trafo_G1 = grid.add_line(
#     gce.Line(name="trafo 1-2", bus_from=bus2, bus_to=bus1,
#              r=0.00000, x=0.015 * (100.0/900.0), b=0.0, rate=900.0))
#
# # line1 = grid.add_line(
# #     gce.Line(name="line 2-3", bus_from=bus2, bus_to=bus3,
# #              r=0.00000, x=0.015 * (100.0/900.0), b=0.0, rate=900.0))
#
# # load
# load1 = gce.Load(name="load1", P=400.0, Q=80.0, Pl0=-4.0, Ql0=-0.8)
# load1.time = t
# load1_grid = grid.add_load(bus=bus2, api_obj=load1)
# # load1 = grid.add_load(bus=bus7, api_obj=gce.Load(P=967.0, Q=100.0, Pl0=-9.670000000007317, Ql0=-0.9999999999967969))
#
#
# # Generators
# fn_1 = 60.0
# M_1 = 13.0 * 9.0
# D_1 = 10.0 * 9.0
# ra_1 = 0.0
# xd_1 = 0.3 * 100.0 / 900.0
# omega_ref_1 = 1.0
# Kp_1 = 0.0
# Ki_1 = 0.0
#
#
# # Generators
# gen1 = gce.Generator(
#     name="Gen1", P=700.0, vset=1.03, Snom=900.0,
#     x1=xd_1, r1=ra_1, freq=fn_1,
#     # vf=1.0,
#     # tm0=700.0/900.0,   # ≈ 0.7778
#     # tm0=6.999999999999923,
#     tm0=4.000000157249711,
#     vf=1.0646373808567275,
#     # vf0=1.141048034212655,
#     M=M_1, D=D_1,
#     omega_ref=omega_ref_1,
#     Kp=Kp_1, Ki=Ki_1
# )
#
# grid.add_generator(bus=bus1, api_obj=gen1)


# SIMPLE SYSTEM 1: 1 GEN, 1 TRAFO, 1 LINE, 1 LOAD
# Buses
# bus1 = gce.Bus(name="Bus1", Vnom=20, is_slack=True)
# bus2 = gce.Bus(name="Bus2", Vnom=230)
# bus3 = gce.Bus(name="Bus3", Vnom=230)
#
# grid.add_bus(bus1)
# grid.add_bus(bus2)
# grid.add_bus(bus3)
#
# # Line
#
#
# trafo_G1 = grid.add_line(
#     gce.Line(name="trafo 1-2", bus_from=bus1, bus_to=bus2,
#              r=0.00000, x=0.015 * (100.0/900.0), b=0.0, rate=900.0))
#
# line1 = grid.add_line(
#     gce.Line(name="line 2-3", bus_from=bus2, bus_to=bus3,
#              r=0.00000, x=0.015 * (100.0/900.0), b=0.0, rate=900.0))
#
# # load
# load1 = gce.Load(name="load1", P=400.0, Q=80.0, Pl0=-4.0, Ql0=-0.8)
# load1.time = t
# load1_grid = grid.add_load(bus=bus3, api_obj=load1)
# # load1 = grid.add_load(bus=bus7, api_obj=gce.Load(P=967.0, Q=100.0, Pl0=-9.670000000007317, Ql0=-0.9999999999967969))
#
#
# # Generators
# fn_1 = 60.0
# M_1 = 13.0 * 9.0
# D_1 = 10.0 * 9.0
# ra_1 = 0.0
# xd_1 = 0.3 * 100.0 / 900.0
# omega_ref_1 = 1.0
# Kp_1 = 0.0
# Ki_1 = 0.0
#
#
# # Generators
# gen1 = gce.Generator(
#     name="Gen1", P=700.0, vset=1.03, Snom=900.0,
#     x1=xd_1, r1=ra_1, freq=fn_1,
#     # vf=1.0,
#     # tm0=700.0/900.0,   # ≈ 0.7778
#     # tm0=6.999999999999923,
#     tm0=4.000000315333287,
#     vf=1.0654837273526685,
#     # vf0=1.141048034212655,
#     M=M_1, D=D_1,
#     omega_ref=omega_ref_1,
#     Kp=Kp_1, Ki=Ki_1
# )
#
# grid.add_generator(bus=bus1, api_obj=gen1)

# SIMPLE SYSTEM 1.1: 1 GEN, 1 TRAFO, 2 LINE (series), 1 LOAD
# Buses
bus1 = gce.Bus(name="Bus1", Vnom=20, is_slack=True)
bus2 = gce.Bus(name="Bus2", Vnom=230)
bus3 = gce.Bus(name="Bus3", Vnom=230)
bus4 = gce.Bus(name="Bus4", Vnom=230)

grid.add_bus(bus1)
grid.add_bus(bus2)
grid.add_bus(bus3)
grid.add_bus(bus4)

# Line
line_0 = grid.add_line(
    gce.Line(name="trafo 1-2", bus_from=bus1, bus_to=bus2,
             r=0.00000, x=0.015 * (100.0/900.0), b=0.0, rate=900.0))
line1 = grid.add_line(
    gce.Line(name="line 2-3", bus_from=bus2, bus_to=bus3,
             r=0.00000, x=0.015 * (100.0/900.0), b=0.0, rate=900.0))
line2 = grid.add_line(
    gce.Line(name="line 3-4", bus_from=bus3, bus_to=bus4,
             r=0.00000, x=0.015 * (100.0/900.0), b=0.0, rate=900.0))








# load
load1 = gce.Load(name="load1", P=400.0, Q=80.0, Pl0=-4.0, Ql0=-0.8)
load1.time = t
load1_grid = grid.add_load(bus=bus4, api_obj=load1)
# load1 = grid.add_load(bus=bus7, api_obj=gce.Load(P=967.0, Q=100.0, Pl0=-9.670000000007317, Ql0=-0.9999999999967969))


# Generators
fn_1 = 60.0
M_1 = 13.0 * 9.0
D_1 = 10.0 * 9.0
ra_1 = 0.0
xd_1 = 0.3 * 100.0 / 900.0
omega_ref_1 = 1.0
Kp_1 = 0.0
Ki_1 = 0.0


# Generators
gen1 = gce.Generator(
    name="Gen1", P=700.0, vset=1.03, Snom=900.0,
    x1=xd_1, r1=ra_1, freq=fn_1,
    # vf=1.0,
    # tm0=700.0/900.0,   # ≈ 0.7778
    # tm0=6.999999999999923,
    tm0=4.000000474297278,
    vf=1.0663347975892765,
    # vf0=1.141048034212655,
    M=M_1, D=D_1,
    omega_ref=omega_ref_1,
    Kp=Kp_1, Ki=Ki_1
)

grid.add_generator(bus=bus1, api_obj=gen1)


#SIMLE SYSTEM 2: 1 GEN, 1 TRAFO, 2 PARALEL LINES, 1 LOAD

# bus1 = gce.Bus(name="Bus1", Vnom=20, is_slack=True)
# bus2 = gce.Bus(name="Bus2", Vnom=230)
# bus3 = gce.Bus(name="Bus3", Vnom=230)
#
# grid.add_bus(bus1)
# grid.add_bus(bus2)
# grid.add_bus(bus3)
#
# # Line
#
#
# trafo_G1 = grid.add_line(
#     gce.Line(name="trafo 1-2", bus_from=bus2, bus_to=bus1,
#              r=0.00000, x=0.0015 * (100.0/900.0), b=0.0, rate=900.0))
#
# line1 = grid.add_line(
#     gce.Line(name="line 2-3-1", bus_from=bus2, bus_to=bus3,
#              r=0.0000, x=0.00011000 * (100.0/400.0), b=0.00019250, rate=400.0))
# line1 = grid.add_line(
#     gce.Line(name="line 2-3-2", bus_from=bus2, bus_to=bus3,
#              r=0.0000, x=0.00011000 * (100.0/400.0), b=0.00019250, rate=400.0))
# # line1 = grid.add_line(
# #     gce.Line(name="line 2-3-1", bus_from=bus2, bus_to=bus3,
# #              r=0.0000, x=0.0011000, b=0.19250, rate=400.0))
# # line1 = grid.add_line(
# #     gce.Line(name="line 2-3-2", bus_from=bus2, bus_to=bus3,
# #              r=0.0000, x=0.0011000, b=0.19250, rate=400.0))
#
# # load
# load1 = gce.Load(name="load1", P=650.0, Q=250.0, Pl0=-6.50, Ql0=-2.50)
# load1.time = t
# load1_grid = grid.add_load(bus=bus3, api_obj=load1)
# # load1 = grid.add_load(bus=bus7, api_obj=gce.Load(P=967.0, Q=100.0, Pl0=-9.670000000007317, Ql0=-0.9999999999967969))
#
#
#
# # Generators
# fn_1 = 60.0
# M_1 = 13.0 * 9.0
# D_1 = 10.0 * 9.0
# ra_1 = 0.0
# xd_1 = 0.3 * 100.0 / 900.0
# omega_ref_1 = 1.0
# Kp_1 = 0.0
# Ki_1 = 0.0
#
#
# # Generators
# gen1 = gce.Generator(
#     name="Gen1", P=700.0, vset=1.03, Snom=900.0,
#     x1=xd_1, r1=ra_1, freq=fn_1,
#     # vf=1.0,
#     # tm0=700.0/900.0,   # ≈ 0.7778
#     # tm0=6.999999999999923,
#     tm0=6.50000000012592,
#     vf=1.1307812687576402,
#     # vf0=1.141048034212655,
#     M=M_1, D=D_1,
#     omega_ref=omega_ref_1,
#     Kp=Kp_1, Ki=Ki_1
# )
#
# grid.add_generator(bus=bus1, api_obj=gen1)
# ---------------------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------------------

# event1 = RmsEvent(load1, "Pl0", np.array([2.5, 12.5]), np.array([-9.0, -9.01]))
event1 = RmsEvent(load1, "Pl0", np.array([2.5]), np.array([-4.01]))

# event2 = RmsEvent(load1, "Ql0", np.array([16.5]), np.array([-0.8]))


# grid.add_rms_event(event1)
# grid.add_rms_event(event2)

# Run power flow
options = gce.PowerFlowOptions(
    solver_type=gce.SolverType.NR,
    retry_with_other_methods=False,
    verbose=0,
    initialize_with_existing_solution=True,
    tolerance=1e-8,
    max_iter=25,
    control_q=False,
    control_taps_modules=True,
    control_taps_phase=True,
    control_remote_voltage=True,
    orthogonalize_controls=True,
    apply_temperature_correction=True,
    branch_impedance_tolerance_mode=gce.BranchImpedanceMode.Specified,
    distributed_slack=False,
    ignore_single_node_islands=False,
    trust_radius=1.0,
    backtracking_parameter=0.05,
    use_stored_guess=False,
    initialize_angles=False,
    generate_report=False,
    three_phase_unbalanced=False
)
res = gce.power_flow(grid, options=options)

# # Print results
print(res.get_bus_df())
print(res.get_branch_df())
print(f"Converged: {res.converged}")

# ----------------------------------------------------------------------------------------------------------------------
# Time Domain Simulation
# ----------------------------------------------------------------------------------------------------------------------
# TDS initialization
ss, init_guess = gce.initialize_rms(grid, res)
print("init_guess")
print(init_guess)


params_mapping = {}

# TODO: initi_guess in hardcoded was Dict(Var, float), now it's Dict((int(uid), str(name)), float) for debugging. So slv.build_init_vars_vector(init_guess) and slv.sort_vars(init_guess) needs to be addressed
# # Solver
start_building_system = time.time()

slv = BlockSolver(ss, t)

end_building_system = time.time()
print(f"Automatic build system time = {end_building_system-start_building_system:.6f} [s]")

params0 = slv.build_init_params_vector(params_mapping)
x0 = slv.build_init_vars_vector_from_uid(init_guess)
vars_in_order = slv.sort_vars_from_uid(init_guess)

start_simulation = time.time()

t, y = slv.simulate(
    t0=0,
    t_end=20.0,
    h=0.001,
    x0=x0,
    params0=params0,
    time=t,
    method="implicit_euler"
)

end_simulation = time.time()
print(f"Automatic simulation time = {end_simulation-start_simulation:.6f} [s]")


# TODO: check results and implement test once initialize_rms is working!
# # Save to csv
slv.save_simulation_to_csv('simulation_results_Ieee_automatic_init.csv', t, y, csv_saving=True)

# # Plot
# plt.plot(t, y[:, slv.get_var_idx(slv._state_vars[1])], label="ω (pu)")
# plt.plot(t, y[:, slv.get_var_idx(slv._state_vars[4])], label="ω (pu)")
# plt.plot(t, y[:, slv.get_var_idx(slv._state_vars[7])], label="ω (pu)")
# plt.plot(t, y[:, slv.get_var_idx(slv._state_vars[10])], label="ω (pu)")

# plt.legend(loc='upper right', ncol=2)
# plt.xlabel("Time (s)")
# plt.ylabel("Values (pu)")
# plt.xlim([0, 20.0])
# plt.ylim([0.85, 1.15])
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# print("x0:", x0)
# print("size x0:", x0.shape)
# print("params0:", params0)

#stability assessment
start_stability = time.time()

# stab, Eigenvalues, V, W, PF, A = slv.stability_assessment(z=y[1000], params=params0, plot = True)
stab, Eigenvalues, V, W, PF, A = slv.stability_assessment(z=x0, params=params0, plot = True)

end_stability = time.time()
print(f"Time for stability assessment = {end_stability - start_stability:.6f} [s]")

print("State matrix A:", A.toarray())
print("Stability assessment:", stab)
print("Eigenvalues:", Eigenvalues)
#print("Right eivenvectors:", V)
#print("Left eigenvectors:", W)
print("Participation factors:", PF.toarray())





df_Eig = pd.DataFrame(Eigenvalues)
df_Eig.to_csv("Eigenvalues_results.csv", index=False , header = False)
df_A = pd.DataFrame(A.toarray())
df_A.to_csv("A_results.csv", index=False , header = False)