# distutils: language = C++

from libcpp cimport bool as cbool
from pele.potentials._pele cimport shared_ptr
from mcpele.monte_carlo._conf_test_cpp cimport cppCheckSphericalContainerConfig, cppCheckHypercubicContainerConfig
from mcpele.monte_carlo._pele_mc cimport _Cdef_ConfTest, cppConfTest
from ._gmc_cpp cimport cppGMCConfTest, _Cdef_GMCConfTest


cdef class _Cdef_CheckSphericalContainerConfigGMC(_Cdef_GMCConfTest):
    cdef cppCheckSphericalContainerConfig* newptr
    def __cinit__(self, radius):
        self.thisptr = shared_ptr[cppGMCConfTest](<cppGMCConfTest*> new cppCheckSphericalContainerConfig(radius))
        self.newptr = <cppCheckSphericalContainerConfig*> self.thisptr.get()


class CheckSphericalContainerConfigGMC(_Cdef_CheckSphericalContainerConfigGMC):
    """Check that the configuration point of the system is within a spherical container

    Parameters
    ----------
    radius : double
        radius of the spherical container, centered at **0**
    """
    pass


cdef class _Cdef_CheckHypercubicContainerConfigGMC(_Cdef_GMCConfTest):
    cdef cppCheckHypercubicContainerConfig* newptr
    def __cinit__(self, double side_length, cbool use_powered_cosine_sum_gradient = False):
        self.thisptr = shared_ptr[cppGMCConfTest](<cppGMCConfTest*> new cppCheckHypercubicContainerConfig(
            side_length, use_powered_cosine_sum_gradient))
        self.newptr = <cppCheckHypercubicContainerConfig*> self.thisptr.get()


class CheckHypercubicContainerConfigGMC(_Cdef_CheckHypercubicContainerConfigGMC):
    pass
