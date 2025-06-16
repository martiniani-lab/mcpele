# distutils: language = c++
# distutils: sources = actions.cpp
# cython: language_level=3str

cimport cython
cimport numpy as np

import numpy as np
import sys


# cython has no support for integer template argument.  This is a hack to get around it
# https://groups.google.com/forum/#!topic/cython-users/xAZxdCFw6Xs
# Basically you fool cython into thinking INT2 is the type integer,
# but in the generated c++ code you use 2 instead.
# The cython code MyClass[INT2] will create c++ code MyClass<2>.
cdef extern from *:
    ctypedef int INT2 "2"    # a fake type
    ctypedef int INT3 "3"    # a fake type

#===============================================================================
# Record Energy Histogram
#===============================================================================        

cdef class _Cdef_RecordEnergyHistogram(_Cdef_Action):
    cdef cppRecordEnergyHistogram* newptr
    def __cinit__(self, min, max, bin, eqsteps):
        self.thisptr = shared_ptr[cppAction](<cppAction*> new cppRecordEnergyHistogram(min, max, bin, eqsteps))
        self.newptr = <cppRecordEnergyHistogram*> self.thisptr.get()
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_histogram(self):
        """returns the histogram array
        
        Returns
        -------
        numpy.array
            energy histogram
        """
        cdef _pele.Array[double] histi = self.newptr.get_histogram()
        cdef double *histdata = histi.data()
        cdef np.ndarray[double, ndim=1, mode="c"] hist = np.zeros(histi.size())
        cdef size_t i
        for i in range(histi.size()):
            hist[i] = histdata[i]
              
        return hist
        
    def print_terminal(self):
        """draws histogram on the terminal
        """
        self.newptr.print_terminal()
    
    def get_bounds_val(self):
        """get energy boundaries of the histogram
        
        Returns
        -------
        Emax : double
            maximum energy of the histogram
        Emin : double
            minimum energy of the histogram
        """
        Emin = self.newptr.get_min()
        Emax = self.newptr.get_max()
        return Emin, Emax
    
    def get_mean_variance(self):
        """get mean and variance of the histogram
        
        Returns
        -------
        mean : double
            first moment of the distribution
        variance : double
            second central moment of the distribution
        """
        mean = self.newptr.get_mean()
        variance = self.newptr.get_variance()
        return mean, variance
    
    def get_count(self):
        """get number of entries in histogram
        
        Returns
        -------
        count : integer
            number of entries in histogram
        """
        count = self.newptr.get_count()
        return count
        
class RecordEnergyHistogram(_Cdef_RecordEnergyHistogram):
    """Bins energies into a resizable histogram
    
    This class is the Python interface for the c++ RecordEnergyHistogram implementation.
    
    .. warning :: :class:`RecordEnergyHistogram` should only start recording
                  entries when the system is equilibrated, set the number of steps
                  to skip with the ``eqsteps`` parameter.
    
    Parameters
    ----------
    min : double
        guess for the minimum energy expected
    max : double
        guess for the maximum energy expected
    bin : double
        choice for the bin size
    eqsteps: int
        number of iterations to skip before starting to record entries
        
    """

#===============================================================================
# Record Pair Distances Histogram
#===============================================================================
cdef class  _Cdef_RecordPairDistHistogram(_Cdef_Action):
    cdef cppRecordPairDistHistogram[INT2]* newptr2
    cdef cppRecordPairDistHistogram[INT3]* newptr3
    cdef _pele_opt.GradientOptimizer optimizer
    def __cinit__(self, boxvec, nr_bins, eqsteps, record_every, optimizer=None):
        cdef np.ndarray[double, ndim=1] bv = np.array(boxvec, dtype=float)
        bv_ = array_wrap_np(bv)
        ndim = len(boxvec)
        assert(ndim == 2 or ndim == 3)
        assert(len(boxvec)==ndim)
        if optimizer is None:
            self.quench = False
            if ndim == 2:
                self.thisptr = shared_ptr[cppAction](<cppAction*> new cppRecordPairDistHistogram[INT2](bv_, nr_bins, eqsteps, record_every))
            else:
                self.thisptr = shared_ptr[cppAction](<cppAction*> new cppRecordPairDistHistogram[INT3](bv_, nr_bins, eqsteps, record_every))
        else:
            self.quench = True
            self.optimizer = optimizer
            if ndim == 2:
                self.thisptr = shared_ptr[cppAction](<cppAction*> new cppRecordPairDistHistogram[INT2](bv_, nr_bins, eqsteps, record_every, self.optimizer.thisptr))
            else:
                self.thisptr = shared_ptr[cppAction](<cppAction*> new cppRecordPairDistHistogram[INT3](bv_, nr_bins, eqsteps, record_every, self.optimizer.thisptr))
        if ndim == 2:
            self.newptr2 = <cppRecordPairDistHistogram[INT2]*> self.thisptr.get()
        else:
            self.newptr3 = <cppRecordPairDistHistogram[INT3]*> self.thisptr.get()
        self.ndim = ndim
        
    def get_hist_r(self):
        """get array of :math:`r` values for :math:`g(r)` measurement
        
        Returns
        -------
        numpy.array
            array of :math:`r` values for :math:`g(r)` histogram
        """
        cdef _pele.Array[double] histi
        if self.ndim == 2:
            histi = self.newptr2.get_hist_r()
        else:
            histi = self.newptr3.get_hist_r()
        cdef double *histdata = histi.data()
        cdef np.ndarray[double, ndim=1, mode="c"] hist = np.zeros(histi.size())
        cdef size_t i
        for i in range(histi.size()):
            hist[i] = histdata[i]      
        return hist
    
    def get_hist_gr(self, number_density, nr_particles):
        """get array of :math:`g(r)` values for :math:`g(r)` measurement
        
        Returns
        -------
        numpy.array
            array of array of :math:`g(r)` values for :math:`g(r)`
        """
        cdef _pele.Array[double] histi
        if self.ndim == 2:
            histi = self.newptr2.get_hist_gr(number_density, nr_particles)
        else:
            histi = self.newptr3.get_hist_gr(number_density, nr_particles)        
        cdef double *histdata = histi.data()
        cdef np.ndarray[double, ndim=1, mode="c"] hist = np.zeros(histi.size())
        cdef size_t i
        for i in range(histi.size()):
            hist[i] = histdata[i]      
        return hist
    
    def get_eqsteps(self):
        """get number of equilibration steps
        
        Returns
        -------
        int
            number of equilibration steps
        """
        if self.ndim == 2:
            return self.newptr2.get_eqsteps()
        else:
            return self.newptr3.get_eqsteps()

class RecordPairDistHistogram(_Cdef_RecordPairDistHistogram):
    """Record a pair distribution function histogram
    
    This class is the Python interface for the c++ mcpele::RecordPairDistHistogram implementation.
    The pair correlation function (or radial distribution function) describes how the density of a
    system of particles varies as a function of distance from a reference particle. In simplest terms 
    it is a measure of the probability of finding a particle at a distance of :math:`r` away from a given 
    reference particle.
    
    Every time the action is called, it accumulates the present configuration into the same :math:`g(r)` histogram.
    The action function calls ``add_configuration`` which accumulates the current configuration into the :math:`g(r)` 
    histogram. The :math:`g(r)` histogram can be read out at any point after that.
     
    Parameters
    ----------
    boxvec : numpy.array
        array of box side lengths
    nr_bins : int
        number of bins for the :math:`g(r)` histogram
    eqsteps : int
        number of equilibration steps to be excluded from :math:`g(r)` computation
    record_every : int
        after ``eqsteps`` steps have been done, record every ``record_everyth`` steps
    optimizer : pele graident optimizer (optional)
        If an optimizer is passed, this will quench the snapshot of coords before
        accumulating the distances to the g(r) histogram. This is
        intended to give the quenched g(r) mentioned here:
        http://dx.doi.org/10.1063/1.449840
    """
    
#===============================================================================
# RecordEnergyTimeseries
#===============================================================================
        
cdef class _Cdef_RecordEnergyTimeseries(_Cdef_Action):
    cdef cppRecordScalarTimeseries* newptr
    def __cinit__(self, niter, record_every):
        self.thisptr = shared_ptr[cppAction](<cppAction*> new cppRecordEnergyTimeseries(niter, record_every))
        self.newptr = <cppRecordScalarTimeseries*> self.thisptr.get()
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_time_series(self):
        """get a energy time series array
        
        Returns
        -------
        numpy.array
            array containing the energy time series
        """
        cdef _pele.Array[double] seriesi = self.newptr.get_time_series()
        cdef double *seriesdata = seriesi.data()
        cdef np.ndarray[double, ndim=1, mode="c"] series = np.zeros(seriesi.size())
        cdef size_t i
        for i in range(seriesi.size()):
            series[i] = seriesdata[i]
              
        return series
    
    def clear(self):
        """clear time series container
        
        deletes the entries in the c++ container
        """
        self.newptr.clear()
    
class RecordEnergyTimeseries(_Cdef_RecordEnergyTimeseries):
    """Record a time series of the energy
    
    This class is the Python interface for the c++ bv::RecordEnergyTimeseries 
    :class:`Action` class implementation.
    
    Parameters
    ----------
    niter: int, Deprecated
        expected number of steps (to preallocate)
    record_every : int
        interval every which the energy is recorded
    """
    
#===============================================================================
# RecordLowestEValueTimeseries
#===============================================================================
        
cdef class _Cdef_RecordLowestEValueTimeseries(_Cdef_Action):
    cdef cppRecordScalarTimeseries* newptr
    cdef ranvec
    def __cinit__(self, niter, record_every, _pele.BasePotential landscape_potential, boxdimension,
                  ranvec, lbfgsniter):
        cdef np.ndarray[double, ndim=1] ranvecc = ranvec
        self.thisptr = shared_ptr[cppAction](<cppAction*> new 
                 cppRecordLowestEValueTimeseries(niter, record_every,
                                                     landscape_potential.thisptr, boxdimension,
                                                     _pele.Array[double](<double*> ranvecc.data, ranvecc.size), lbfgsniter))
        self.newptr = <cppRecordScalarTimeseries*> self.thisptr.get()
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_time_series(self):
        """get the lowest eigenvalue time series array
        
        Returns
        -------
        numpy.array:
            array time series of the lowest eigenvalue
        """
        cdef _pele.Array[double] seriesi = self.newptr.get_time_series()
        cdef double *seriesdata = seriesi.data()
        cdef np.ndarray[double, ndim=1, mode="c"] series = np.zeros(seriesi.size())
        cdef size_t i
        for i in range(seriesi.size()):
            series[i] = seriesdata[i]
              
        return series
    
    def clear(self):
        """clear time series container
        
        deletes the entries in the c++ container
        """
        self.newptr.clear()
    
class RecordLowestEValueTimeseries(_Cdef_RecordLowestEValueTimeseries):
    """Record lowest eigenvalue of the inherent structure
    
    This class is the Python interface for the c++ RecordLowestEValueTimeseries :class:`Action` 
    class implementation.
    The structure is quenched to a minimum energy configuration (its inherent structure) and
    the lowest eigenvalue is computed by the Rayleight-Ritz method for lowest eigenvalue (which
    computationally cheaper than the diagonalisation of the Hessian). The zero modes are
    orthogonalised through the Gram-Schmidt orthogonalisation procedure.
    
    Parameters
    ----------
    niter: int, Deprecated
        expected number of steps (to preallocate)
    record_every : int
        interval every which the energy is recorded
    landscape_potential : :class:`BasePotential <pele:pele.potentials.BasePotential>`
        potential associated with particles (so the underlying potential energy surface)
    boxdimension: int
        dimensionality of the space (dimensionality of box)
    ranvec : numpy.array
        random vector of length equal to the number of degrees of freedom [len(coords)],
        required by the Gram-Schmidt orthogonalisation procedure
    lbfgsniter : int
        maximum number of steps for the LBFG-S minimisation of the Rayleigh quotient
    """
    
#===============================================================================
# RecordDisplacementPerParticleTimeseries
#===============================================================================
        
cdef class _Cdef_RecordDisplacementPerParticleTimeseries(_Cdef_Action):
    cdef cppRecordScalarTimeseries* newptr
    cdef initial_coords
    def __cinit__(self, niter, record_every, initial_coords, boxdimension):
        cdef np.ndarray[double, ndim=1] initialc = initial_coords
        self.thisptr = shared_ptr[cppAction](<cppAction*> new 
                 cppRecordDisplacementPerParticleTimeseries(niter, record_every,
                                                            _pele.Array[double](<double*> initialc.data, initialc.size), 
                                                            boxdimension))
        self.newptr = <cppRecordScalarTimeseries*> self.thisptr.get()
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_time_series(self):
        """return a the root mean square displacement time series array
        
        Returns
        -------
        np.array
            root mean square displacement array
        """
        cdef _pele.Array[double] seriesi = self.newptr.get_time_series()
        cdef double *seriesdata = seriesi.data()
        cdef np.ndarray[double, ndim=1, mode="c"] series = np.zeros(seriesi.size())
        cdef size_t i
        for i in range(seriesi.size()):
            series[i] = seriesdata[i]
              
        return series
    
    def clear(self):
        """clear time series container
        
        deletes the entries in the c++ container
        """
        self.newptr.clear()

class RecordDisplacementPerParticleTimeseries(_Cdef_RecordDisplacementPerParticleTimeseries):
    """Record time series of the average root mean square displacement per particle at each step
    
    This class is the Python interface for the c++ RecordDisplacementPerParticleTimeseries 
    :class:`Action` class implementation.
    
    Parameters
    ----------
    niter: int, Deprecated
        expected number of steps (to preallocate)
    record_every : int
        interval every which the energy is recorded
    initial_coords : numpy.array
        initial system coordinates, used to compute rms distance
    boxdimension: int
        dimensionality of the space (dimensionality of box)
    """

cdef class _Cdef_RecordCoordsTimeseries(_Cdef_Action):
    cdef cppRecordVectorTimeseries* newptr
    cdef cppRecordCoordsTimeseries* newptr2
    def __cinit__(self, ndof, record_every=1, eqsteps=0):
        self.thisptr = shared_ptr[cppAction](<cppAction*> new cppRecordCoordsTimeseries(ndof, record_every, eqsteps))
        self.newptr = <cppRecordVectorTimeseries*> self.thisptr.get()
        self.newptr2 = <cppRecordCoordsTimeseries*> self.thisptr.get()
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_time_series(self):
        """get a trajectory
        
        Returns
        -------
        numpy.array
            deque of arrays
        """
        cdef deque[_pele.Array[double]] dq = self.newptr.get_time_series()
        cdef double *seriesdata
        cdef np.ndarray[double, ndim=2, mode="c"] series = np.zeros((dq.size(), dq[0].size()))
        cdef size_t i, j
        for i in range(dq.size()):
            seriesdata = dq[i].data()
            for j in range(dq[i].size()):
                series[i][j] = seriesdata[j]
        return series
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_mean_variance_time_series(self):
        """returns the mean coordinate vector
        
        Returns
        -------
        numpy.array
            mean coordinate vector
        numpy.array
            element-wise variance coordinate vector
        """
        cdef _pele.Array[double] coordi = self.newptr2.get_mean_coordinate_vector()
        cdef _pele.Array[double] vari = self.newptr2.get_variance_coordinate_vector()
        cdef double *coorddata = coordi.data()
        cdef double *vardata = vari.data()
        cdef np.ndarray[double, ndim=1, mode="c"] coord = np.zeros(coordi.size())
        cdef np.ndarray[double, ndim=1, mode="c"] var = np.zeros(vari.size())
        cdef size_t i
        for i in range(coordi.size()):
            coord[i] = coorddata[i]
            var[i] = vardata[i]
        return coord, var

    def get_avg_count(self):
        """get number of steps over which the average and variance were computed
        
        Returns
        -------
        count : integer
            sample size
        """
        count = self.newptr2.get_count()
        return count
    
    def clear(self):
        """clear time series container
        
        deletes the entries in the c++ container
        """
        self.newptr.clear()
    
class RecordCoordsTimeseries(_Cdef_RecordCoordsTimeseries):
    """Record a trajectory of the system coordinates
       
    This class is the Python interface for the c++ bv::RecordCoordsTimeseries 
    :class:`Action` class implementation.
    
    Parameters
    ----------
    ndof : int
        dimensionality of coordinate array
    record_every : int
        interval every which the coordinates are recorded
    eqsteps : int
        number of equilibration steps to skip when computing averages
    """
