from libcpp cimport bool as cbool
from libcpp.vector cimport vector
from _pele_mc cimport cppAcceptTest, cppConfTest, _Cdef_AcceptTest, _Cdef_ConfTest, shared_ptr
cimport pele.potentials._pele as _pele

#===============================================================================
# Metropolis acceptance criterion
#===============================================================================

cdef extern from "mcpele/metropolis_test.h" namespace "mcpele":
    cdef cppclass cppMetropolisTest "mcpele::MetropolisTest":
        cppMetropolisTest(size_t) except +
        size_t get_seed() except +
        void set_generator_seed(size_t) except +

#==========
# CloudTest
#==========

cdef extern from "mcpele/cloud_test.h" namespace "mcpele":
    ctypedef struct cppDrop "mcpele::Drop":
        double bias
        cbool oracle
        _pele.Array[double] x
    cdef cppclass Cloud(vector[shared_ptr[cppDrop]]):
        Cloud() except+
        Cloud(size_t) except+
    cdef cppclass cppCloudTest "mcpele::CloudTest":
        cppCloudTest(size_t, size_t, size_t, double, shared_ptr[_pele.cBasePotential]) except+
        cppCloudTest(size_t, size_t, size_t, double) except+
        void set_generator_seeds(size_t, size_t) except+
        void add_conf_test(shared_ptr[cppConfTest]) except +
        Cloud get_old_cloud() except+
        void set_cloud(Cloud) except+
        void set_cloud_bias(_pele.Array[double]) except+
        
