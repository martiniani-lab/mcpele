# distutils: language = c++
# distutils: sources = gmc.cpp

from abc import ABC
import numpy as np
cimport numpy as np
from mcpele.monte_carlo._pele_mc cimport _Cdef_AcceptTest
from mcpele.monte_carlo._monte_carlo_cpp cimport _Cdef_MCBase
from mcpele.monte_carlo._monte_carlo_cpp import _BaseMCBaseRunner, Result
from ctypes import c_size_t as size_t
from pele.potentials._pele cimport array_wrap_np_size_t, pele_array_to_np_size_t


cdef class _Cdef_GMC(_Cdef_MCBase):
    cdef cppGMC * thisptr
    cdef public _pele.BasePotential potential
    cdef public start_coords
    cdef public double timestep
    cdef public size_t nparticles
    cdef public size_t ndim
    cdef public size_t rseed
    cdef public size_t resample_velocity_steps
    cdef public double max_timestep
    cdef public cbool use_random_timestep
    cdef public size_t adaptive_interval
    cdef public double adaptive_factor
    cdef public double adaptive_min_acceptance_ratio
    cdef public double adaptive_max_acceptance_ratio
    cdef public cbool reflect_boundary
    cdef public cbool reflect_potential

    def __init__(self, _pele.BasePotential potential, coords, double temperature, size_t pniter, double timestep,
                 size_t nparticles, size_t ndim, size_t rseed, size_t resample_velocity_steps = 0,
                 double max_timestep = 0.0, cbool use_random_timestep = False, size_t adaptive_interval = 100,
                 double adaptive_factor = 0.9, double adaptive_min_acceptance_ratio = 0.2,
                 double adaptive_max_acceptance_ratio = 0.5, cbool reflect_boundary = True,
                 cbool reflect_potential = False):
        cdef np.ndarray[double, ndim=1] cstart_coords = np.array(coords, dtype=float)
        self.potential = potential
        self.start_coords = cstart_coords
        self.temperature = temperature
        self.niter = pniter
        self.timestep = timestep
        self.nparticles = nparticles
        self.ndim = ndim
        self.rseed = rseed
        self.resample_velocity_steps = resample_velocity_steps
        self.max_timestep = max_timestep
        self.use_random_timestep = use_random_timestep
        self.adaptive_interval = adaptive_interval
        self.adaptive_factor = adaptive_factor
        self.adaptive_min_acceptance_ratio = adaptive_min_acceptance_ratio
        self.adaptive_max_acceptance_ratio = adaptive_max_acceptance_ratio
        self.reflect_boundary = reflect_boundary
        self.reflect_potential = reflect_potential
        self.baseptr = shared_ptr[cppMCBase](
            <cppMCBase *> new cppGMC(self.potential.thisptr,
                                     _pele.Array[double](<double *> cstart_coords.data, cstart_coords.size),
                                     self.temperature, self.timestep, self.nparticles, self.ndim, self.rseed,
                                     self.resample_velocity_steps, self.max_timestep, self.use_random_timestep,
                                     self.adaptive_interval, self.adaptive_factor, self.adaptive_min_acceptance_ratio,
                                     self.adaptive_max_acceptance_ratio, self.reflect_boundary, self.reflect_potential))
        self.thisptr = <cppGMC *> self.baseptr.get()

    def add_accept_test(self, _Cdef_AcceptTest test):
        self.thisptr.add_accept_test(test.thisptr)

    def add_conf_test(self, _Cdef_GMCConfTest test):
        self.thisptr.add_conf_test(test.thisptr)

    def add_late_conf_test(self, _Cdef_GMCConfTest test):
        self.thisptr.add_late_conf_test(test.thisptr)

    def get_timestep(self):
        return self.thisptr.get_timestep()

    def set_timestep(self, double timestep):
        self.thisptr.set_timestep(timestep)

    def get_count(self):
        return self.thisptr.get_count()

    def set_count(self, size_t count):
        self.thisptr.set_count(count)

    def get_adaptation_counters(self):
        return pele_array_to_np_size_t(self.thisptr.get_adaptation_counters())

    def set_adaptation_counters(self, np.ndarray[size_t, ndim=1] input not None):
        self.thisptr.set_adaptation_counters(array_wrap_np_size_t(input))

    def get_counters(self):
        return pele_array_to_np_size_t(self.thisptr.get_counters())

    def set_counters(self, np.ndarray[size_t, ndim=1] input not None):
        self.thisptr.set_counters(array_wrap_np_size_t(input))


class _BaseGMCRunner(_Cdef_GMC, _BaseMCBaseRunner, ABC):
    def __init__(self, potential, coords, temperature, niter, timestep, nparticles, ndim, rseed,
                 resample_velocity_steps = 0, max_timestep = 0.0, use_random_timestep = False, adaptive_interval = 100,
                 adaptive_factor = 0.9, adaptive_min_acceptance_ratio = 0.2, adaptive_max_acceptance_ratio = 0.5,
                 reflect_boundary = True, reflect_potential = False):
        super().__init__(potential, coords, temperature, niter, timestep, nparticles, ndim, rseed,
                         resample_velocity_steps, max_timestep, use_random_timestep, adaptive_interval, adaptive_factor,
                         adaptive_min_acceptance_ratio, adaptive_max_acceptance_ratio, reflect_boundary,
                         reflect_potential)
        # TODO: THIS FEELS WEIRD
        self.ndim = len(coords)
        self.result = Result()
        self.result.message = []
