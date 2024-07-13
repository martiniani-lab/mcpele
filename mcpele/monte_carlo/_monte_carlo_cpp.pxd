cimport pele.potentials._pele as _pele
from libcpp cimport bool as cbool
from _pele_mc cimport *

cdef class _Cdef_MCBase(_Cdef_BaseMCBase):
    cdef public size_t niter
    cdef public double temperature
    cdef cbool m_use_energy_change
