# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at https://mozilla.org/MPL/2.0/.
# SPDX-License-Identifier: MPL-2.0
from VeraGridEngine.Devices.multi_circuit import MultiCircuit
from VeraGridEngine.Utils.Symbolic import Var
from VeraGridEngine.Utils.Symbolic.block_solver import BlockSolver
from VeraGridEngine.Simulations.driver_template import DriverTemplate
from VeraGridEngine.Simulations.Rms.rms_options import RmsOptions
from VeraGridEngine.Simulations.Rms.rms_results import RmsResults
from VeraGridEngine.Simulations.Rms.problems.rms_problem import RmsProblem
from VeraGridEngine.Simulations.Rms.numerical.integration_methods import Trapezoid, BackEuler
from VeraGridEngine.enumerations import EngineType, SimulationTypes, DynamicIntegrationMethod
import VeraGridEngine.api as gce


class RmsSimulationDriver(DriverTemplate):
    name = 'Rms Simulation'
    tpe = SimulationTypes.RmsDynamic_run

    """
    Dynamic wrapper to use with Qt
    """

    def __init__(self, grid: MultiCircuit,
                 options: RmsOptions,
                 engine: EngineType = EngineType.VeraGrid):

        """
        DynamicDriver class constructor
        :param grid: MultiCircuit instance
        :param options: RmsOptions instance (optional)
        :param engine: EngineType (i.e., EngineType.VeraGrid) (optional)
        """

        DriverTemplate.__init__(self, grid=grid, engine=engine)

        self.options = options

        self.results = RmsResults()

    def run(self):
        """
        Main function to initialize and run the system simulation.

        This function sets up logging, starts the dynamic simulation, and
        logs the outcome. It handles and logs any exceptions raised during execution.
        :return:
        """
        # Run the dynamic simulation
        self.run_time_simulation()

    def run_time_simulation(self):
        """
        Performs the numerical integration using the chosen method.
        :return:
        """

        # # Get integration method
        # if self.options.integration_method == DynamicIntegrationMethod.Trapezoid:
        #     integrator = Trapezoid()
        # elif self.options.integration_method == DynamicIntegrationMethod.BackEuler:
        #     integrator = BackEuler()
        # else:
        #     raise ValueError(f"integrator not implemented :( {self.options.integration_method}")

        options = gce.PowerFlowOptions(
            solver_type=gce.SolverType.NR,
            retry_with_other_methods=False,
            verbose=0,
            initialize_with_existing_solution=True,
            tolerance=1e-6,
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
        res = gce.power_flow(self.grid, options=options)

        t = Var("t")

        params_mapping = {}

        ss, init_guess = gce.initialize_rms(self.grid, res)

        slv = BlockSolver(ss, t)

        params0 = slv.build_init_params_vector(params_mapping)
        x0 = slv.build_init_vars_vector_from_uid(init_guess)

        t, y = slv.simulate(
            t0=0,
            t_end=20.0,
            h=0.001,
            x0=x0,
            params0=params0,
            glob_time=t,
            method="implicit_euler"
        )
        slv.save_simulation_to_csv('simulation_results.csv', t, y, csv_saving=True)



