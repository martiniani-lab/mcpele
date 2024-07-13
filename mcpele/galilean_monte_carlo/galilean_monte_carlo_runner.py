from typing import Optional, Sequence
import matplotlib.pyplot as plt
import numpy as np
from pele.potentials._pele import BasePotential
from pele.potentials import Harmonic
from mcpele.monte_carlo._pele_mc import _Cdef_Action
from mcpele.monte_carlo import NullPotential, MetropolisTest, RecordCoordsTimeseries
from mcpele.galilean_monte_carlo import _BaseGMCRunner, CheckHypercubicContainerConfigGMC
from mcpele.galilean_monte_carlo._gmc_cpp import _Cdef_GMCConfTest


class GalileanMonteCarloRunner(_BaseGMCRunner):
    def __init__(self, potential: BasePotential, coords: np.ndarray, temperature: float, pniter: int, timestep: float,
                 nparticles: int, ndim: int, resample_velocity_steps: int = 0, max_timestep: float = 0.0,
                 conftests: Sequence[_Cdef_GMCConfTest] = (), late_conftests: Sequence[_Cdef_GMCConfTest] = (),
                 actions: Sequence[_Cdef_Action] = (), seeds: Optional[dict[str, int]] = None,
                 adaptive_iterations: int = 1000, adaptive_interval: int = 100, adaptive_factor: float = 0.9,
                 adaptive_acceptance=0.5) -> None:
        if seeds is not None:
            if "takestep" not in seeds or "metropolis" not in seeds:
                raise ValueError("Seeds must contain takestep and metropolis keys.")
        else:
            i32max = np.iinfo(np.int32).max
            rng = np.random.default_rng()
            seeds = dict(
                takestep=rng.integers(low=0, high=i32max),
                metropolis=rng.integers(low=0, high=i32max),
            )
        super().__init__(potential, coords, temperature, pniter, timestep, nparticles, ndim, seeds["takestep"],
                         resample_velocity_steps, max_timestep, adaptive_interval, adaptive_factor,
                         adaptive_acceptance, adaptive_acceptance)
        self.set_report_steps(adaptive_iterations)  # set number of iterations for which steps are adapted
        self._conftests = conftests
        self._late_conftests = late_conftests
        self._actions = actions

        self._metropolis = MetropolisTest(seeds["metropolis"])

        # noinspection PyTypeChecker
        self.add_accept_test(self._metropolis)
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
    iterations = 100
    resample_velocity_steps = 50
    gmc = GalileanMonteCarloRunner(
        potential=NullPotential(), coords=initial_coords, temperature=1.0, pniter=iterations, timestep=0.1,
        nparticles=1, ndim=len(initial_coords), resample_velocity_steps=resample_velocity_steps, max_timestep=0.0,
        conftests=(), late_conftests=(CheckHypercubicContainerConfigGMC(side_length),), actions=(action,),
        seeds={"metropolis": 1, "takestep": 10}, adaptive_iterations=0, adaptive_interval=100, adaptive_factor=0.9,
        adaptive_acceptance=0.5)
    gmc.run()
    timeseries = np.concatenate((initial_coords.reshape((1, 2)), action.get_time_series()))
    plt.figure()
    plt.plot(timeseries[:, 0], timeseries[:, 1], marker=".")
    for r in range(0, iterations, resample_velocity_steps):
        plt.plot(timeseries[r, 0], timeseries[r, 1], marker="o", color="red")
    plt.xlim(-side_length / 2.0, side_length / 2.0)
    plt.ylim(-side_length / 2.0, side_length / 2.0)
    plt.gca().set_aspect("equal")
    plt.savefig("ExampleGalileanMonteCarloRunner.pdf")
    plt.close()

    initial_coords = np.array([0.0, 0.0])
    action = RecordCoordsTimeseries(2)
    side_length = 1.0
    iterations = 100
    resample_velocity_steps = 50
    gmc = GalileanMonteCarloRunner(
        potential=Harmonic(np.array([0.0, 0.0]), 15.0, bdim=2, com=False), coords=initial_coords, temperature=1.0,
        pniter=iterations, timestep=0.1, nparticles=1, ndim=len(initial_coords),
        resample_velocity_steps=resample_velocity_steps, max_timestep=0.0,
        conftests=(), late_conftests=(CheckHypercubicContainerConfigGMC(side_length),), actions=(action,),
        seeds={"metropolis": 1, "takestep": 10}, adaptive_iterations=0, adaptive_interval=100, adaptive_factor=0.9,
        adaptive_acceptance=0.5)
    gmc.run()
    timeseries = np.concatenate((initial_coords.reshape((1, 2)), action.get_time_series()))
    plt.figure()
    plt.plot(timeseries[:, 0], timeseries[:, 1], marker=".")
    for r in range(0, iterations, resample_velocity_steps):
        plt.plot(timeseries[r, 0], timeseries[r, 1], marker="o", color="red")
    plt.xlim(-side_length / 2.0, side_length / 2.0)
    plt.ylim(-side_length / 2.0, side_length / 2.0)
    plt.gca().set_aspect("equal")
    plt.savefig("ExampleGalileanMonteCarloRunnerWithBias.pdf")
    plt.close()
