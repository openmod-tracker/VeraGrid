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
# Buses
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
# # bus4 = gce.Bus(name="Bus4", Vnom=230)
#
# grid.add_bus(bus1)
# grid.add_bus(bus2)
# grid.add_bus(bus3)
# # grid.add_bus(bus4)
#
# # Line
# r1 = 0
# x1 = 0.011
# r2 = 0
# x2 = 0.011
#
# z1 = complex(r1,x1)
# z2 = complex(r2,x2)
#
# zeq = (z1*z2)/(z1+z2)
# print("zeq:",zeq)
# req = zeq.real
# xeq = zeq.imag
#
# trafo_G1 = grid.add_line(
#     gce.Line(name="trafo 1-2", bus_from=bus1, bus_to=bus2,
#              r=0.00000, x=0.015 * (100.0/900.0), b=0.0, rate=900.0))
# # line1 = grid.add_line(
# #     gce.Line(name="line 2-3-1", bus_from=bus2, bus_to=bus3,
# #              r=req, x=xeq * (100.0/450.0), b=0.0, rate=450.0))
# line1 = grid.add_line(
#     gce.Line(name="line 2-3-1", bus_from=bus2, bus_to=bus3,
#              r=r1, x=x1 * (100.0/900.0), b=0.00019250, rate=450.0))
# line2 = grid.add_line(
#     gce.Line(name="line 2-3-2", bus_from=bus2, bus_to=bus3,
#              r=r2, x=x2 * (100.0/900.0), b=0.00019250, rate=450.0))
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
#     tm0=4.000000236028486,
#     vf=1.0648290838557057,
#     # vf0=1.141048034212655,
#     M=M_1, D=D_1,
#     omega_ref=omega_ref_1,
#     Kp=Kp_1, Ki=Ki_1
# )
#
# grid.add_generator(bus=bus1, api_obj=gen1)

# SIMPLE SYSTEM 1.1: 1 GEN, 1 TRAFO, 2 LINE (series), 1 LOAD
# Buses
# bus1 = gce.Bus(name="Bus1", Vnom=20, is_slack=True)
# bus2 = gce.Bus(name="Bus2", Vnom=230)
# bus3 = gce.Bus(name="Bus3", Vnom=230)
# bus4 = gce.Bus(name="Bus4", Vnom=230)
#
# grid.add_bus(bus1)
# grid.add_bus(bus2)
# grid.add_bus(bus3)
# grid.add_bus(bus4)
#
# # Line
# line_0 = grid.add_line(
#     gce.Line(name="trafo 1-2", bus_from=bus1, bus_to=bus2,
#              r=0.00000, x=0.015 * (100.0/900.0), b=0.0, rate=900.0))
# line1 = grid.add_line(
#     gce.Line(name="line 2-3", bus_from=bus2, bus_to=bus3,
#              r=0.00000, x=0.015 * (100.0/900.0), b=0.0, rate=900.0))
# line2 = grid.add_line(
#     gce.Line(name="line 3-4", bus_from=bus3, bus_to=bus4,
#              r=0.00000, x=0.015 * (100.0/900.0), b=0.0, rate=900.0))
#
#
# # load
# load1 = gce.Load(name="load1", P=400.0, Q=80.0, Pl0=-4.0, Ql0=-0.8)
# load1.time = t
# load1_grid = grid.add_load(bus=bus4, api_obj=load1)
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
#     tm0=4.000000474297278,
#     vf=1.0663347975892765,
#     # vf0=1.141048034212655,
#     M=M_1, D=D_1,
#     omega_ref=omega_ref_1,
#     Kp=Kp_1, Ki=Ki_1
# )
#
# grid.add_generator(bus=bus1, api_obj=gen1)


#SIMLE SYSTEM 2: 1 GEN, 1 TRAFO, 2 PARALEL LINES, 1 LOAD

# bus1 = gce.Bus(name="Bus1", Vnom=20, is_slack=True)
# bus2 = gce.Bus(name="Bus2", Vnom=230)
# bus3 = gce.Bus(name="Bus3", Vnom=230)
#
# grid.add_bus(bus1)
# grid.add_bus(bus2)
# grid.add_bus(bus3)
# # Line
#
#
# trafo_G1 = grid.add_line(
#     gce.Line(name="trafo 1-2", bus_from=bus1, bus_to=bus2,
#              r=0.00000, x=0.0015 * (100.0/900.0), b=0.0, rate=900.0))
# r1 = 0
# x1 = 0.03
# r2 = 0
# x2 = 0.03
#
# z1 = complex(r1,x1)
# z2 = complex(r2,x2)
#
# zeq = (z1*z2)/(z1+z2)
# req = zeq.real
# xeq = zeq.imag
#
# line1 = grid.add_line(
#     gce.Line(name="line 2-3-1", bus_from=bus2, bus_to=bus3,
#              r=req, x=xeq * (100.0/900.0), b=0.000, rate=900.0))
# # line1 = grid.add_line(
# #     gce.Line(name="line 2-3-1", bus_from=bus2, bus_to=bus3,
# #              r=0.00, x=0.011 * (100.0/500.0), b=0.000, rate=500.0))
# # line2 = grid.add_line(
# #     gce.Line(name="line 2-3-2", bus_from=bus2, bus_to=bus3,
# #              r=0.0000, x=0.011 * (100.0/500.0), b=0.000, rate=500.0))
#
# # line1 = grid.add_line(
# #     gce.Line(name="line 2-3-1", bus_from=bus2, bus_to=bus3,
# #              r=0.0000, x=0.0011000, b=0.19250, rate=400.0))
# # line1 = grid.add_line(
# #     gce.Line(name="line 2-3-2", bus_from=bus2, bus_to=bus3,
# #              r=0.0000, x=0.0011000, b=0.19250, rate=400.0))
#
# # load
# load1 = gce.Load(name="load1", P=400.0, Q=80.0, Pl0=-4.0, Ql0=-0.80)
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
#     tm0=4.000000328289261,
#     vf=1.0762971793681322,
#     # vf0=1.141048034212655,
#     M=M_1, D=D_1,
#     omega_ref=omega_ref_1,
#     Kp=Kp_1, Ki=Ki_1
# )
#
# grid.add_generator(bus=bus1, api_obj=gen1)

# SIMPLE SYSTEM 3: 2 GEN(1 at bus 1, 1 at bus 3 that connects to bus2 by line 3), 1 TRAFO, 1 LINE, 1 LOAD
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
# r1 = 0
# x1 = 0.011
# r2 = 0
# x2 = 0.011
#
# z1 = complex(r1,x1)
# z2 = complex(r2,x2)
#
# zeq = (z1*z2)/(z1+z2)
# print("zeq:",zeq)
# req = zeq.real
# xeq = zeq.imag

trafo_G1 = grid.add_line(
    gce.Line(name="trafo 1-2", bus_from=bus1, bus_to=bus2,
             r=0.00000, x=0.015 * (100.0/900.0), b=0.0, rate=900.0))
# line1 = grid.add_line(
#     gce.Line(name="line 2-3-1", bus_from=bus2, bus_to=bus3,
#              r=req, x=xeq * (100.0/450.0), b=0.0, rate=450.0))
line1 = grid.add_line(
    gce.Line(name="line 2-3", bus_from=bus2, bus_to=bus3,
             r=0.00, x=0.03 * (100.0/900.0), b=0.0, rate=900.0))
line2 = grid.add_line(
    gce.Line(name="line 2-4", bus_from=bus2, bus_to=bus4,
             r=0.00, x=0.03 * (100.0/900.0), b=0.0, rate=900.0))

# load
load1 = gce.Load(name="load1", P=900.0, Q=100.0, Pl0=-9.0, Ql0=-1.0)
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

fn_2 = 60.0
M_2 = 13.0 * 9.0
D_2 = 10.0 * 9.0
ra_2 = 0.0
xd_2 = 0.3 * 100.0 / 900.0
omega_ref_2 = 1.0
Kp_2 = 0.0
Ki_2 = 0.0


# Generators
gen1 = gce.Generator(
    name="Gen1", P=700.0, vset=1.03, Snom=900.0,
    x1=xd_1, r1=ra_1, freq=fn_1,
    # vf=1.0,
    # tm0=700.0/900.0,   # ≈ 0.7778
    # tm0=6.999999999999923,
    tm0=2.0000012904377917,
    vf=1.061039353796422,
    # vf0=1.141048034212655,
    M=M_1, D=D_1,
    omega_ref=omega_ref_1,
    Kp=Kp_1, Ki=Ki_1
)

gen2 = gce.Generator(
    name="Gen2", P=700.0, vset=1.03, Snom=900.0,
    x1=xd_2, r1=ra_2, freq=fn_2,
    # vf=1.0,
    # tm0=700.0/900.0,   # ≈ 0.7778
    # tm0=6.999999999999923,
    tm0=7.0,
    vf=1.0712033002527792,
    # vf0=1.141048034212655,
    M=M_2, D=D_2,
    omega_ref=omega_ref_2,
    Kp=Kp_2, Ki=Ki_2
)

grid.add_generator(bus=bus1, api_obj=gen1)
grid.add_generator(bus=bus3, api_obj=gen2)
# ---------------------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------------------

# event1 = RmsEvent(load1, "Pl0", np.array([2.5, 12.5]), np.array([-9.0, -9.01]))
# event1 = RmsEvent(load1, "Pl0", np.array([2.5]), np.array([-4.01]))

# event2 = RmsEvent(load1, "Ql0", np.array([16.5]), np.array([-0.8]))


# grid.add_rms_event(event1)
# grid.add_rms_event(event2)

# Run power flow
options = gce.PowerFlowOptions(
    solver_type=gce.SolverType.NR,
    retry_with_other_methods=False,
    verbose=0,
    initialize_with_existing_solution=True,
    tolerance=1e-13,
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
    trust_radius=0.8,
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

v_PF = np.abs(res.voltage)
a_PF = np.angle(res.voltage)
p_PF = res.Sbus.real/100
q_PF = res.Sbus.imag/100

v_PFdf = pd.DataFrame([v_PF])  # shape: [T, n_loads]
v_PFdf.columns = [f"v_PF_VeraGrid_Bus_{i + 1}" for i in range(v_PF.shape[0])]
a_PFdf = pd.DataFrame([a_PF])  # shape: [T, n_loads]
a_PFdf.columns = [f"a_PF_VeraGrid_Bus_{i + 1}" for i in range(a_PF.shape[0])]
p_PFdf = pd.DataFrame([p_PF])  # shape: [T, n_loads]
p_PFdf.columns = [f"p_PF_VeraGrid_Bus_{i + 1}" for i in range(p_PF.shape[0])]
q_PFdf = pd.DataFrame([q_PF])  # shape: [T, n_loads]
q_PFdf.columns = [f"q_PF_VeraGrid_Bus_{i + 1}" for i in range(q_PF.shape[0])]

PFdf = pd.concat([v_PFdf, a_PFdf, p_PFdf, q_PFdf], axis=1)
PFdf.to_csv("PowerFlow_VeraGrid_output.csv", index=False)
print('Power Flow results saved in PowerFlow_VeraGrid_output.csv')

# ----------------------------------------------------------------------------------------------------------------------
# Time Domain Simulation
# ----------------------------------------------------------------------------------------------------------------------
# TDS initialization
ss, init_guess = gce.initialize_rms(grid, res)
print("init_guess")
print(init_guess)

#bus
Vm_bus_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'Vm']
Va_bus_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'Va']
#line
Pf_line_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'Pf']
Qf_line_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'Qf']
Pt_line_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'Pt']
Qt_line_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'Qt']
#gencls
delta_gen_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'delta']
omega_gen_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'omega']
Id_gen_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'i_d']
Iq_gen_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'i_q']
vd_gen_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'v_d']
vq_gen_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'v_q']
tm_gen_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'tm']
te_gen_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'te']
Pg_gen_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'P_g']
Qg_gen_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'Q_g']
psid_gen_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'psid']
psiq_gen_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'psiq']
et_gen_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'et']

#load
Pl_load_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'Pl']
Ql_load_init_guess = [v for (k1, k2), v in init_guess.items() if k2 == 'Ql']

#bus
Vm_bus_init_guessdf = pd.DataFrame([Vm_bus_init_guess])  # shape: [T, n_loads]
Vm_bus_init_guessdf.columns = [f"Vm_VeraGrid_Bus_{i + 1}" for i in range(len(Vm_bus_init_guess))]
Va_bus_init_guessdf = pd.DataFrame([Va_bus_init_guess])  # shape: [T, n_loads]
Va_bus_init_guessdf.columns = [f"Va_VeraGrid_Bus_{i + 1}" for i in range(len(Va_bus_init_guess))]
#load
Pf_line_init_guessdf = pd.DataFrame([Pf_line_init_guess])  # shape: [T, n_loads]
Pf_line_init_guessdf.columns = [f"Pf_VeraGrid_line_{i + 1}" for i in range(len(Pf_line_init_guess))]
Qf_line_init_guessdf = pd.DataFrame([Qf_line_init_guess])  # shape: [T, n_loads]
Qf_line_init_guessdf.columns = [f"Qf_VeraGrid_line_{i + 1}" for i in range(len(Qf_line_init_guess))]
Pt_line_init_guessdf = pd.DataFrame([Pt_line_init_guess])  # shape: [T, n_loads]
Pt_line_init_guessdf.columns = [f"Pt_VeraGrid_line_{i + 1}" for i in range(len(Pt_line_init_guess))]
Qt_line_init_guessdf = pd.DataFrame([Qt_line_init_guess])  # shape: [T, n_loads]
Qt_line_init_guessdf.columns = [f"Qt_VeraGrid_line_{i + 1}" for i in range(len(Qt_line_init_guess))]
#gen
delta_gen_init_guessdf = pd.DataFrame([delta_gen_init_guess])  # shape: [T, n_loads]
delta_gen_init_guessdf.columns = [f"delta_VeraGrid_gen_{i + 1}" for i in range(len(delta_gen_init_guess))]
omega_gen_init_guessdf = pd.DataFrame([omega_gen_init_guess])  # shape: [T, n_loads]
omega_gen_init_guessdf.columns = [f"omega_VeraGrid_gen_{i + 1}" for i in range(len(omega_gen_init_guess))]
Id_gen_init_guessdf = pd.DataFrame([Id_gen_init_guess])  # shape: [T, n_loads]
Id_gen_init_guessdf.columns = [f"Id_VeraGrid_gen_{i + 1}" for i in range(len(Id_gen_init_guess))]
Iq_gen_init_guessdf = pd.DataFrame([Iq_gen_init_guess])  # shape: [T, n_loads]
Iq_gen_init_guessdf.columns = [f"Iq_VeraGrid_gen_{i + 1}" for i in range(len(Iq_gen_init_guess))]
vd_gen_init_guessdf = pd.DataFrame([vd_gen_init_guess])  # shape: [T, n_loads]
vd_gen_init_guessdf.columns = [f"vd_VeraGrid_gen_{i + 1}" for i in range(len(vd_gen_init_guess))]
vq_gen_init_guessdf = pd.DataFrame([vq_gen_init_guess])  # shape: [T, n_loads]
vq_gen_init_guessdf.columns = [f"vq_VeraGrid_gen_{i + 1}" for i in range(len(vq_gen_init_guess))]
tm_gen_init_guessdf = pd.DataFrame([tm_gen_init_guess])  # shape: [T, n_loads]
tm_gen_init_guessdf.columns = [f"tm_VeraGrid_gen_{i + 1}" for i in range(len(tm_gen_init_guess))]
te_gen_init_guessdf = pd.DataFrame([te_gen_init_guess])  # shape: [T, n_loads]
te_gen_init_guessdf.columns = [f"te_VeraGrid_gen_{i + 1}" for i in range(len(te_gen_init_guess))]
Pg_gen_init_guessdf = pd.DataFrame([Pg_gen_init_guess])  # shape: [T, n_loads]
Pg_gen_init_guessdf.columns = [f"Pg_VeraGrid_gen_{i + 1}" for i in range(len(Pg_gen_init_guess))]
Qg_gen_init_guessdf = pd.DataFrame([Qg_gen_init_guess])  # shape: [T, n_loads]
Qg_gen_init_guessdf.columns = [f"Qg_VeraGrid_gen_{i + 1}" for i in range(len(Qg_gen_init_guess))]
psid_gen_init_guessdf = pd.DataFrame([psid_gen_init_guess])  # shape: [T, n_loads]
psid_gen_init_guessdf.columns = [f"psid_VeraGrid_gen_{i + 1}" for i in range(len(psid_gen_init_guess))]
psiq_gen_init_guessdf = pd.DataFrame([psiq_gen_init_guess])  # shape: [T, n_loads]
psiq_gen_init_guessdf.columns = [f"psiq_VeraGrid_gen_{i + 1}" for i in range(len(psiq_gen_init_guess))]
et_gen_init_guessdf = pd.DataFrame([et_gen_init_guess])  # shape: [T, n_loads]
et_gen_init_guessdf.columns = [f"et_VeraGrid_gen_{i + 1}" for i in range(len(et_gen_init_guess))]
#load
Pl_load_init_guessdf = pd.DataFrame([Pl_load_init_guess])  # shape: [T, n_loads]
Pl_load_init_guessdf.columns = [f"Pl_VeraGrid_load_{i + 1}" for i in range(len(Pl_load_init_guess))]
Ql_load_init_guessdf = pd.DataFrame([Ql_load_init_guess])  # shape: [T, n_loads]
Ql_load_init_guessdf.columns = [f"Ql_VeraGrid_load_{i + 1}" for i in range(len(Ql_load_init_guess))]

#
init_df = pd.concat([Vm_bus_init_guessdf, Va_bus_init_guessdf, Pf_line_init_guessdf, Qf_line_init_guessdf,
                     Pt_line_init_guessdf, Qt_line_init_guessdf, delta_gen_init_guessdf,omega_gen_init_guessdf,
                     Id_gen_init_guessdf, Iq_gen_init_guessdf,vd_gen_init_guessdf,vq_gen_init_guessdf, tm_gen_init_guessdf,
                     te_gen_init_guessdf, Pg_gen_init_guessdf, Qg_gen_init_guessdf, psid_gen_init_guessdf, psiq_gen_init_guessdf,
                     et_gen_init_guessdf, Pl_load_init_guessdf, Ql_load_init_guessdf], axis=1)
init_df.to_csv("init_guess_VeraGrid_output.csv", index=False)
print('Init guess results saved in init_guess_VeraGrid_output.csv')


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

# stab, Eigenvalues, V, W, PFactors, A = slv.stability_assessment(z=y[1000], params=params0, plot = True)
stab, Eigenvalues, A, V, W, PFactors = slv.stability_assessment(z=y[1], params=params0, plot = True)

end_stability = time.time()
print(f"Time for stability assessment = {end_stability - start_stability:.6f} [s]")

print("State matrix A:", A.toarray())
print("Stability assessment:", stab)
print("Eigenvalues:", Eigenvalues)
#print("Right eivenvectors:", V)
#print("Left eigenvectors:", W)
print("Participation factors:", PFactors.toarray().T)





df_Eig = pd.DataFrame(Eigenvalues)
df_Eig.to_csv("Eigenvalues_results.csv", index=False , header = False)
df_A = pd.DataFrame(A.toarray())
df_A.to_csv("A_results.csv", index=False , header = False)
df_PFactors = pd.DataFrame(PFactors.toarray())
df_PFactors.to_csv("pfactors_results.csv", index=False , header = False)
df_V = pd.DataFrame(V.toarray())
df_V.to_csv("V_results.csv", index=False , header = False)
df_W = pd.DataFrame(W.toarray())
df_W.to_csv("W_results.csv", index=False , header = False)