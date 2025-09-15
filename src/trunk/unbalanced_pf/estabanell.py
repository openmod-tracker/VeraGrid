import VeraGridEngine.api as gce
import pandas as pd
import numpy as np

logger = gce.Logger()
grid = gce.MultiCircuit()
df_buses_lines = pd.read_csv('estabanell_grid/linies_bt_eRoots.csv', sep=";")

# ---------------------------------------------------------------------------------------------------------------------
#   Buses
# ---------------------------------------------------------------------------------------------------------------------
buses = pd.unique(df_buses_lines[["node_start", "node_end"]].values.ravel())
bus_dict = dict()
i = 0
for bus in buses:
    bus = gce.Bus(name=str(bus), Vnom=0.4)
    if i == 0:
        bus.is_slack = True
        gen = gce.Generator()
        grid.add_generator(bus=bus, api_obj=gen)
        i+=1
    grid.add_bus(obj=bus)
    bus_dict[int(float(bus.name))] = bus

# ---------------------------------------------------------------------------------------------------------------------
#   Lines
# ---------------------------------------------------------------------------------------------------------------------
for _, row in df_buses_lines.iterrows():
    line_type = gce.SequenceLineType(
        name=row['tram'],
        Imax=row['intensitat_admisible'] / 1e3,
        Vnom=400,
        R=row['resistencia'],
        X=row['reactancia'],
        R0= 3 * row['resistencia'],
        X0= 3 * row['reactancia']
    )
    grid.add_sequence_line(line_type)

    line = gce.Line(
        bus_from=bus_dict[row['node_start']],
        bus_to=bus_dict[row['node_end']],
        name=row['tram'],
        code=row['num_linia'],
        rate=row['intensitat_admisible'] * 400 / 1e6,
        length=row['longitud_cad']/1000,
        template=line_type
    )
    grid.add_line(obj=line)

# ---------------------------------------------------------------------------------------------------------------------
#   Loads
# ---------------------------------------------------------------------------------------------------------------------
df_meters = pd.read_csv('estabanell_grid/cups_eRoots.csv', sep=";")
df_powers = pd.read_csv('estabanell_grid/Corbes_CUPS_CT-0975.csv', sep=";")

# For now, only the first measurement of each meter
df_first = df_powers.groupby("MeterID").first().reset_index()

for _, row_meters in df_meters.iterrows():
    row_power = df_first[df_first["MeterID"] == row_meters["comptador"]]

    P = float(row_power["TotalActiveEnergyConsumed"].values[0])
    Q = float(row_power["TotalReactiveEnergyConsumed"].values[0])

    load = gce.Load(name=row_meters['comptador'])

    if int(row_meters['tensio']) == 400:
        # Balanced
        load.Pa = P / 3
        load.Pb = P / 3
        load.Pc = P / 3
        load.Qa = Q / 3
        load.Qb = Q / 3
        load.Qc = Q / 3

    elif int(row_meters['tensio']) == 230:
        # Unbalanced
        phase_probs = [1 / 3, 1 / 3, 1 / 3]
        phase = np.random.choice(["A", "B", "C"], p=phase_probs)

        if phase == "A":
            load.Pa = P
            load.Qa = Q
        elif phase == "B":
            load.Pb = P
            load.Qb = Q
        else:
            load.Pc = P
            load.Qc = Q

    else:
        raise Exception("Incorrect load voltage!")

    grid.add_load(bus=bus_dict[row_meters['node']], api_obj=load)

# ----------------------------------------------------------------------------------------------------------------------
# Run power flow
# ----------------------------------------------------------------------------------------------------------------------
res = gce.power_flow(grid=grid, options=gce.PowerFlowOptions(three_phase_unbalanced=True))
print("\n", res.get_voltage_3ph_df())
print("\nConverged? ", res.converged)
print("\nIterations: ", res.iterations)

# ---------------------------------------------------------------------------------------------------------------------
#   Save Grid
# ---------------------------------------------------------------------------------------------------------------------
print()
# gce.save_file(grid=grid, filename='estabanell.veragrid')