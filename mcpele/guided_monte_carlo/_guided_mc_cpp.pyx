# distutils: language = c++
# distutils: sources = gmc.cpp

from abc import ABC
from ctypes import c_size_t as size_t
import numpy as np
cimport numpy as np
from pele.potentials._pele cimport array_wrap_np_size_t, pele_array_to_np_size_t
from mcpele.monte_carlo._monte_carlo_cpp cimport _Cdef_MCBase
from mcpele.monte_carlo._monte_carlo_cpp import _BaseMCBaseRunner, Result
from mcpele.galilean_monte_carlo._gmc_cpp cimport _Cdef_GMCConfTest


cdef class _Cdef_GuidedMC(_Cdef_MCBase):
    cdef cppGuidedMC * thisptr
    cdef public _pele.BasePotential potential
    cdef public start_coords
    cdef public double timestep
    cdef public double standard_deviation
    cdef public size_t rseed
    cdef public cbool normalize_conf_gradient
    cdef public cbool use_hessian
    cdef public double max_timestep
    cdef public size_t adaptive_interval
    cdef public double adaptive_factor
    cdef public double adaptive_min_acceptance_ratio
    cdef public double adaptive_max_acceptance_ratio

    def __init__(self, _pele.BasePotential potential, coords, double temperature, size_t pniter, double timestep,
                 double standard_deviation, size_t rseed, cbool normalize_conf_gradient = True,
                 cbool use_hessian = False, double max_timestep = 0.0, size_t adaptive_interval = 100,
                 double adaptive_factor = 0.9, double adaptive_min_acceptance_ratio = 0.2,
                 double adaptive_max_acceptance_ratio = 0.5):
        cdef np.ndarray[double, ndim=1] cstart_coords = np.array(coords, dtype=float)
        self.potential = potential
        self.start_coords = cstart_coords
        self.temperature = temperature
        self.niter = pniter
        self.timestep = timestep
        self.standard_deviation = standard_deviation
        self.rseed = rseed
        self.normalize_conf_gradient = normalize_conf_gradient
        self.use_hessian = use_hessian
        self.max_timestep = max_timestep
        self.adaptive_interval = adaptive_interval
        self.adaptive_factor = adaptive_factor
        self.adaptive_min_acceptance_ratio = adaptive_min_acceptance_ratio
        self.adaptive_max_acceptance_ratio = adaptive_max_acceptance_ratio
        self.baseptr = shared_ptr[cppMCBase](
            <cppMCBase *> new cppGuidedMC(self.potential.thisptr,
                                     _pele.Array[double](<double *> cstart_coords.data, cstart_coords.size),
                                     self.temperature, self.timestep, self.standard_deviation, self.rseed,
                                     self.normalize_conf_gradient, self.use_hessian, self.max_timestep,
                                     self.adaptive_interval, self.adaptive_factor, self.adaptive_min_acceptance_ratio,
                                     self.adaptive_max_acceptance_ratio))
        self.thisptr = <cppGuidedMC *> self.baseptr.get()

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


class _BaseGuidedMCRunner(_Cdef_GuidedMC, _BaseMCBaseRunner, ABC):
    def __init__(self, potential, coords, temperature, niter, timestep, standard_deviation, rseed,
                 normalize_conf_gradient = True, use_hessian = False, max_timestep = 0.0, adaptive_interval = 100,
                 adaptive_factor = 0.9, adaptive_min_acceptance_ratio = 0.2, adaptive_max_acceptance_ratio = 0.5):
        super().__init__(potential, coords, temperature, niter, timestep, standard_deviation, rseed,
                         normalize_conf_gradient, use_hessian, max_timestep, adaptive_interval, adaptive_factor,
                         adaptive_min_acceptance_ratio, adaptive_max_acceptance_ratio)
        # TODO: THIS FEELS WEIRD
        self.ndim = len(coords)
        self.result = Result()
        self.result.message = []
