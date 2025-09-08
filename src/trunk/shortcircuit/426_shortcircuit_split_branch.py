import VeraGridEngine.api as vge

# --------------------------------------------------------------------------------------------------------
# Creating the grid
# --------------------------------------------------------------------------------------------------------
logger = vge.Logger()
grid = vge.MultiCircuit()

bus_1 = vge.Bus(is_slack=True)
grid.add_bus(obj=bus_1)

bus_2 = vge.Bus()
grid.add_bus(obj=bus_2)

line = vge.Line(bus_from=bus_1,
                bus_to=bus_2,
                r=1,
                x=1,
                b=1,
                length=500)
grid.add_line(obj=line)