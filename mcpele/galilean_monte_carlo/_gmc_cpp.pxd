#cython: boundscheck=False
#cython: wraparound=False

from libcpp cimport bool as cbool
cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport shared_ptr
from mcpele.monte_carlo._pele_mc cimport cppAcceptTest, cppConfTest, cppMCBase


cdef extern from "mcpele/gmc.h" namespace "mcpele":
    cdef cppclass cppGMCConfTest "mcpele::GMCConfTest"(cppConfTest)


cdef class _Cdef_GMCConfTest:
    """This class is the python interface for the c++ mcpele::GMCConfTest base class implementation."""
    cdef shared_ptr[cppGMCConfTest] thisptr


cdef extern from "mcpele/gmc.h" namespace "mcpele":
    cdef cppclass cppGMC "mcpele::GMC"(cppMCBase):
        cppGMC(shared_ptr[_pele.cBasePotential], _pele.Array[double] &, double, double, size_t, size_t, size_t, size_t,
               double, cbool, size_t, double, double, double, cbool, cbool) except +
        void add_accept_test(shared_ptr[cppAcceptTest]) except +
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
