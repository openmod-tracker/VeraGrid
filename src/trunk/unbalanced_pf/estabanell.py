import VeraGridEngine.api as gce
import pandas as pd

logger = gce.Logger()
grid = gce.MultiCircuit()
df_buses_lines = pd.read_csv('estabanell_grid/linies_bt_eRoots.csv', sep=";")

# ---------------------------------------------------------------------------------------------------------------------
#   Buses
# ---------------------------------------------------------------------------------------------------------------------
buses = pd.unique(df_buses_lines[["node_start", "node_end"]].values.ravel())
bus_dict = dict()
for bus in buses:
    bus = gce.Bus(name=str(bus), Vnom=0.4)
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
#   Save Grid
# ---------------------------------------------------------------------------------------------------------------------
print()
gce.save_file(grid=grid, filename='estabanell.veragrid')