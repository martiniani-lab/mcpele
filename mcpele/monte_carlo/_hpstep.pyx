# distutils: language = c++
# distutils: sources = hpstep.cpp

cimport pele.potentials._pele as _pele
from _pele_mc cimport cppTakeStep,_Cdef_TakeStep, shared_ptr


import sys


cdef extern from "mcpele/hpstep.h" namespace "mcpele":
    cdef cppclass cppHPStep "mcpele::HPStep":
        cppHPStep() except +



cdef class _Cdef_HPStep(_Cdef_TakeStep):
    cdef cppHPStep* newptr
    def __cinit__(self):
        self.thisptr = shared_ptr[cppTakeStep](<cppTakeStep*> new cppHPStep())


class HPStep(_Cdef_HPStep):
    """HP model steps"""