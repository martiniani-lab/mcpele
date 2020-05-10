cimport pele.potentials._pele as _pele
from _pele_mc cimport cppAction,_Cdef_Action, shared_ptr
from _pele_mc cimport cppAcceptTest,_Cdef_AcceptTest, shared_ptr


# note, both of these need to be added for wang landau sampling


#===============================================================================
# Wang Landau Action (Updater)
#===============================================================================


cdef extern from "mcpele/mc_wang_landau.h" namespace "mcpele":
    cdef cppclass cppWL_Updater "mcpele::WL_Updater":
        cppWL_Updater(double , double , double , double ,
                                   double , double ,
                                   double , size_t ) except +
        _pele.Array[double] get_log_dos() except +


#===============================================================================
# Wang Landau Accept Test
#===============================================================================

cdef extern from "mcpele/mc_wang_landau.h" namespace "mcpele":
    cdef cppclass cppWL_AcceptTest "mcpele::WL_AcceptTest":
        cppWL_AcceptTest(size_t, shared_ptr[cppWL_Updater]) except +
        size_t get_seed() except +
        void set_generator_seed(size_t) except +


