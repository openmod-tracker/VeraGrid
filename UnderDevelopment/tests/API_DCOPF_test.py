# This file is part of GridCal.
#
# GridCal is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GridCal is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GridCal.  If not, see <http://www.gnu.org/licenses/>.

from GridCal.Engine.CalculationEngine import *
from matplotlib import pyplot as plt


# fname = '/Data/Doctorado/spv_phd/GridCal_project/GridCal/IEEE_300BUS.xls'
# fname = 'Pegasus 89 Bus.xlsx'
# fname = 'Illinois200Bus.xlsx'
# fname = 'IEEE_30_new.xlsx'
# fname = 'lynn5buspq.xlsx'
# fname = '/home/santi/Documentos/GitHub/GridCal/Grids_and_profiles/grids/IEEE_30_new.xlsx'
# fname = '/home/santi/Documentos/GitHub/GridCal/Grids_and_profiles/grids/IEEE39.xlsx'
# fname = '/Data/Doctorado/spv_phd/GridCal_project/GridCal/IEEE_14.xls'
# fname = '/Data/Doctorado/spv_phd/GridCal_project/GridCal/IEEE_39Bus(Islands).xls'
fname = 'D:\\GitHub\\GridCal\\Grids_and_profiles\\grids\\3 node battery opf.xlsx'

grid = MultiCircuit()
grid.load_file(fname)
grid.compile()

opf_options = OptimalPowerFlowOptions(load_shedding=False)
opf = OptimalPowerFlow(grid, opf_options)
opf.run()

# opf.results