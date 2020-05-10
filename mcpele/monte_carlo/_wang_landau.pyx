# distutils: language = c++
# distutils: sources = mc_wang_landau.cpp




cimport cython
cimport numpy as np

import numpy as np
import sys

from libcpp.memory cimport dynamic_pointer_cast

from pele.potentials._pele cimport array_wrap_np

#===============================================================================
# Wang Landau Action (updater)
#===============================================================================


cdef class _Cdef_WL_Updater(_Cdef_Action):
    cdef cppWL_Updater* newptr
    def __cinit__(self, emin, emax, bin, log_f=1.0,
                  modifier=2.0, log_f_threshold=1e-8,
                  flatness_criterion=0.8, h_iter=1e6):
        self.thisptr = shared_ptr[cppAction](<cppAction*> new cppWL_Updater(emin, emax, bin, log_f,
                                                                            modifier, log_f_threshold,
                                                                            flatness_criterion, h_iter))
        self.newptr = <cppWL_Updater*> self.thisptr.get()
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_log_dos(self):
        """returns the dos array
        
        Returns
        -------
        numpy.array
            log of the density of states
        """
        cdef _pele.Array[double] log_dosp = self.newptr.get_log_dos()
        cdef double *log_dos_data = log_dosp.data()
        cdef np.ndarray[double, ndim=1, mode="c"] log_dos = np.zeros(log_dosp.size())
        cdef size_t i
        for i in xrange(log_dosp.size()):
            log_dos[i] = log_dos_data[i]
        return log_dos


class WL_Updater(_Cdef_WL_Updater):
   """ Wang Landau Action that runs the update steps along with other wang landau actions such as histogram
       resetting other than the acceptance test ). 
    
    This class is the Python interface for the c++ RecordEnergyHistogram implementation.

    
    Parameters
    ----------
    emin : double
        guess for the minimum energy expected
    emax : double
        guess for the maximum energy expected
    bin : double
        choice for the bin size
    log_f: double
        log of the factor to multiply density of states by when updating histogram
    modifier: double
        value to divide log_f by when the histogram is flat
    log_f_threshold: double
        threshold value for log_f when the simulation ends
    flatness_criterion: double
        defines the ratio of the minimum of the histogram to the mean of the histogram above which we say
        the histogram is flat
    h_iter: double 
        defines the number of steps at which the histogram flatness is checked
    """


#===============================================================================
# Wang Landau Accept Test
#===============================================================================
cdef class _Cdef_WL_AcceptTest(_Cdef_AcceptTest):
    cdef cppWL_AcceptTest* newptr
    def __cinit__(self, rseed, _Cdef_WL_Updater update):
        # TODO check that the nullptr raised when dynamic casting the wrong operator gives an exception
        self.thisptr = shared_ptr[cppAcceptTest](<cppAcceptTest*> new cppWL_AcceptTest(rseed, dynamic_pointer_cast[cppWL_Updater, cppAction](update.thisptr)))
        self.newptr = <cppWL_AcceptTest*> self.thisptr.get()