import VeraGridEngine.api as vge

grid = vge.MultiCircuit()

bus_1 = vge.Bus(name='Bus 1', Vnom=1)
bus_1.is_slack = True
grid.add_bus(obj=bus_1)
gen = vge.Generator()
grid.add_generator(bus=bus_1, api_obj=gen)

bus_2 = vge.Bus(name='Bus 2', Vnom=1)
grid.add_bus(obj=bus_2)

line_1_2 = vge.Line(bus_from=bus_1,
                    bus_to=bus_2,
                    length=1,
                    r=0.01,
                    x=0.1)
grid.add_line(obj=line_1_2)

load_2 = vge.Load(P=0.9 * 100,
                  Q=0.3 * 100)
grid.add_load(bus=bus_2, api_obj=load_2)

# ----------------------------------------------------------------------------------------------------------------------
# Run power flow
# ----------------------------------------------------------------------------------------------------------------------
res = vge.power_flow(grid=grid)

# ----------------------------------------------------------------------------------------------------------------------
# Show the results
# ----------------------------------------------------------------------------------------------------------------------
print(res.get_bus_df())