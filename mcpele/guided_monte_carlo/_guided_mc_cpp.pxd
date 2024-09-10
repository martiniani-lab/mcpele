#cython: boundscheck=False
#cython: wraparound=False

from libcpp cimport bool as cbool
cimport pele.potentials._pele as _pele
cimport pele.optimize._pele_opt as _pele_opt
from pele.potentials._pele cimport shared_ptr
from mcpele.monte_carlo._pele_mc cimport cppMCBase
from mcpele.galilean_monte_carlo._gmc_cpp cimport cppGMCConfTest


cdef extern from "mcpele/guided_mc.h" namespace "mcpele":
    cdef cppclass cppGuidedMC "mcpele::GuidedMC"(cppMCBase):
        cppGuidedMC(shared_ptr[_pele.cBasePotential], _pele.Array[double] &, double, double, double, size_t, cbool,
                    cbool, double, size_t, double, double, double) except +
        void add_conf_test(shared_ptr[cppGMCConfTest]) except +
        void add_late_conf_test(shared_ptr[cppGMCConfTest]) except +
        double get_timestep() except +
        void set_timestep(double) except +
        size_t get_count() except +
        void set_count(size_t) except +
        _pele.Array[size_t] get_adaptation_counters() except +
        void set_adaptation_counters(_pele.Array[size_t]) except +
        _pele.Array[size_t] get_counters() except +
        void set_counters(_pele.Array[size_t]) except +


cdef extern from "mcpele/guided_mc_optimizer.h" namespace "mcpele":
    cdef cppclass cppGuidedMCOptimizer "mcpele::GuidedMCOptimizer"(cppMCBase):
        cppGuidedMCOptimizer(shared_ptr[_pele.cBasePotential], _pele.Array[double] &, double,
                             shared_ptr[_pele_opt.cGradientOptimizer], int, double, size_t,
                             double, size_t, double, double, double) except +
        void add_conf_test(shared_ptr[cppGMCConfTest]) except +
        void add_late_conf_test(shared_ptr[cppGMCConfTest]) except +
        double get_standard_deviation() except +
        void set_standard_deviation(double) except +
        size_t get_count() except +
        void set_count(size_t) except +
        _pele.Array[size_t] get_adaptation_counters() except +
        void set_adaptation_counters(_pele.Array[size_t]) except +
        _pele.Array[size_t] get_counters() except +
        void set_counters(_pele.Array[size_t]) except +
