from typing import Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
from pele.optimize._pele_opt import GradientOptimizer
from pele.optimize import CVODEBDFOptimizer
from pele.potentials._pele import BasePotential
from pele.potentials import Harmonic
from pele.potentials import PoweredCosineSum
from mcpele.monte_carlo._pele_mc import _Cdef_Action
from mcpele.monte_carlo import NullPotential, MetropolisTest, RecordCoordsTimeseries
from mcpele.galilean_monte_carlo import CheckHypercubicContainerConfigGMC
from mcpele.galilean_monte_carlo._gmc_cpp import _Cdef_GMCConfTest
from mcpele.guided_monte_carlo import _BaseGuidedMCOptimizerRunner
from basinvolume.monte_carlo import CheckHyperCubicContainerGMC as BVCheckHyperCubicContainerGMC
from basinvolume.utils import INVERSE_POWER_CVODE_95_ACC


class GuidedMonteCarloOptimizerRunner(_BaseGuidedMCOptimizerRunner):
    def __init__(self, potential: BasePotential, coords: np.ndarray, temperature: float, pniter: int, 
                 optimizer: GradientOptimizer, optimizer_niter: int, standard_deviation: float, 
                 max_standard_deviation: float = 0.0, conftests: Sequence[_Cdef_GMCConfTest] = (), 
                 late_conftests: Sequence[_Cdef_GMCConfTest] = (), actions: Sequence[_Cdef_Action] = (), 
                 seeds: Optional[dict[str, int]] = None, adaptive_iterations: int = 1000, adaptive_interval: int = 100,
                 adaptive_factor: float = 0.9, adaptive_acceptance: float = 0.5) -> None:
        if seeds is not None:
            if "guided_mc" not in seeds:
                raise ValueError("Seeds must contain guided_mc keys.")
        else:
            i32max = np.iinfo(np.int32).max
            rng = np.random.default_rng()
            seeds = dict(
                guided_mc=rng.integers(low=0, high=i32max),
            )
        super().__init__(potential, coords, temperature, pniter, optimizer, optimizer_niter, standard_deviation, 
                         seeds["guided_mc"], max_standard_deviation, adaptive_interval, adaptive_factor, 
                         adaptive_acceptance, adaptive_acceptance)
        self.set_report_steps(adaptive_iterations)  # set number of iterations for which steps are adapted
        self._conftests = conftests
        self._late_conftests = late_conftests
        self._actions = actions

        for conftest in self._conftests:
            # noinspection PyTypeChecker
            self.add_conf_test(conftest)
        for late_conftest in self._late_conftests:
            # noinspection PyTypeChecker
            self.add_late_conf_test(late_conftest)
        for action in self._actions:
            # noinspection PyTypeChecker
            self.add_action(action)

    def set_control(self, c: float) -> None:
        self.set_temperature(c)


if __name__ == '__main__':
    initial_coords = np.array([0.0, 0.0])
    action = RecordCoordsTimeseries(2)
    side_length = 1.0
    iterations = 100  # For large width, 100 iterations with standard_deviation=0.1
    optimizer_potential = PoweredCosineSum(2, 1.0, 0.5, 1.0)
    optimizer = CVODEBDFOptimizer(optimizer_potential, initial_coords, tol=1e-10, nsteps=1e5, atol=1e-7, rtol=1e-7)
    gmc = GuidedMonteCarloOptimizerRunner(
        potential=NullPotential(), coords=initial_coords, temperature=1.0, pniter=iterations,
        optimizer=optimizer, optimizer_niter=50, standard_deviation=0.03, max_standard_deviation=0.0, conftests=(),
        late_conftests=(BVCheckHyperCubicContainerGMC(np.array([0.0, 0.0]), side_length, 2, True),), actions=(action,),
        seeds={"guided_mc": 1}, adaptive_iterations=0, adaptive_interval=100, adaptive_factor=0.9,
        adaptive_acceptance=0.5)
    gmc.run()
    timeseries = np.concatenate((initial_coords.reshape((1, 2)), action.get_time_series()))
    plt.figure()
    plt.plot(timeseries[:, 0], timeseries[:, 1], marker=".")
    plt.xlim(-side_length / 2.0, side_length / 2.0)
    plt.ylim(-side_length / 2.0, side_length / 2.0)
    plt.plot([-0.5, 0.5], [-0.5, 0.5], color="k", zorder=-1, alpha=0.5)
    plt.plot([0.5, -0.5], [-0.5, 0.5], color="k", zorder=-1, alpha=0.5)
    plt.gca().set_aspect("equal")
    plt.savefig("ExampleGuidedMonteCarloOptimizerRunner.pdf")
    plt.close()
