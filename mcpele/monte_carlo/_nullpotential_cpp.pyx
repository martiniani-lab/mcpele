# distutils: language = C++
# cython: language_level=3str

cimport cython
cimport pele.potentials._pele as _pele
from pele.potentials._pele cimport shared_ptr
from pele.potentials._pele cimport BasePotential

# use external c++ class
cdef extern from "mcpele/nullpotential.h" namespace "mcpele":
    cdef cppclass cNullPotential "mcpele::NullPotential":
        cNullPotential() except +

cdef class _Cdef_NullPotential(BasePotential):
    """
    this is the null potential
    """
    cdef cNullPotential* newptr

    def __cinit__(self):
        self.thisptr = shared_ptr[_pele.cBasePotential]( <_pele.cBasePotential*>new cNullPotential() )
        self.newptr = <cNullPotential*> self.thisptr.get()

class NullPotential(_Cdef_NullPotential):
    """
    """
