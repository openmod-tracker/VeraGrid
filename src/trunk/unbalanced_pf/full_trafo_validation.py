import VeraGridEngine.api as gce
from VeraGridEngine import WindingType, ShuntConnectionType, SolverType
import numpy as np

logger = gce.Logger()

grid = gce.MultiCircuit()
grid.fBase = 60

# ----------------------------------------------------------------------------------------------------------------------
# Buses
# ----------------------------------------------------------------------------------------------------------------------
bus_632 = gce.Bus(name='632', Vnom=4.16)
bus_632.is_slack = True
grid.add_bus(obj=bus_632)
gen = gce.Generator()
grid.add_generator(bus=bus_632, api_obj=gen)

# bus_633 = gce.Bus(name='633', Vnom=4.16)
# grid.add_bus(obj=bus_633)

bus_634 = gce.Bus(name='634', Vnom=4.16)
grid.add_bus(obj=bus_634)

# ----------------------------------------------------------------------------------------------------------------------
# Load
# ----------------------------------------------------------------------------------------------------------------------
load_634 = gce.Load(G1=0.140,
                    B1=0.100,
                    G2=0.120,
                    B2=0.090,
                    G3=0.100,
                    B3=0.080)
load_634.conn = ShuntConnectionType.GroundedStar
grid.add_load(bus=bus_634, api_obj=load_634)

# load_634 = gce.Load(Ir1=0.140*2,
#                     Ii1=0.100*2,
#                     Ir2=0.120*2,
#                     Ii2=0.090*2,
#                     Ir3=0.100*2,
#                     Ii3=0.080*2)
# load_634.conn = ShuntConnectionType.GroundedStar
# grid.add_load(bus=bus_634, api_obj=load_634)

# load_634 = gce.Load(P1=0.140,
#                     Q1=0.100,
#                     P2=0.120,
#                     Q2=0.090,
#                     P3=0.100,
#                     Q3=0.080)
# load_634.conn = ShuntConnectionType.GroundedStar
# grid.add_load(bus=bus_634, api_obj=load_634)

# ----------------------------------------------------------------------------------------------------------------------
# Transformer
# ----------------------------------------------------------------------------------------------------------------------
# trafo = gce.Transformer2W(name='XFM-1',
#                           bus_from=bus_632,
#                           bus_to=bus_634,
#                           HV=4.16,
#                           LV=0.48,
#                           nominal_power=0.5,
#                           rate=0.5,
#                           r=1.1*2,
#                           x=2*2)
# trafo.conn_f = WindingType.GroundedStar
# trafo.conn_t = WindingType.Delta
# grid.add_transformer2w(trafo)

# ----------------------------------------------------------------------------------------------------------------------
# Line
# ----------------------------------------------------------------------------------------------------------------------
z_602 = np.array([
    [0.7526 + 1j * 1.1814, 0.1580 + 1j * 0.4236, 0.1560 + 1j * 0.5017],
    [0.1580 + 1j * 0.4236, 0.7475 + 1j * 1.1983, 0.1535 + 1j * 0.3849],
    [0.1560 + 1j * 0.5017, 0.1535 + 1j * 0.3849, 0.7436 + 1j * 1.2112]
], dtype=complex) / 1.60934

y_602 = np.array([
    [1j * 5.6990, 1j * -1.0817, 1j * -1.6905],
    [1j * -1.0817, 1j * 5.1795, 1j * -0.6588],
    [1j * -1.6905, 1j * -0.6588, 1j * 5.4246]
], dtype=complex) / 10**6 / 1.60934

config_602 = gce.create_known_abc_overhead_template(name='Config. 602',
                                                    z_abc=z_602,
                                                    ysh_abc=y_602,
                                                    phases=np.array([1, 2, 3]),
                                                    Vnom=4.16,
                                                    frequency=60)
grid.add_overhead_line(config_602)

line_632_633 = gce.Line(bus_from=bus_632,
                        bus_to=bus_634,
                        length=1000 * 0.0003048)
line_632_633.apply_template(config_602, grid.Sbase, grid.fBase, logger)
grid.add_line(obj=line_632_633)

# ----------------------------------------------------------------------------------------------------------------------
# Run power flow
# ----------------------------------------------------------------------------------------------------------------------
res = gce.power_flow(grid=grid, options=gce.PowerFlowOptions(three_phase_unbalanced=True,
                                                             solver_type=SolverType.HELM))

# ----------------------------------------------------------------------------------------------------------------------
# Show the results
# ----------------------------------------------------------------------------------------------------------------------
print("\n", res.get_voltage_3ph_df())
print("\nConverged? ", res.converged)
print("\nIterations: ", res.iterations)