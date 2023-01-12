# distutils: language = c++
# distutils: sources = takestep.cpp

import sys
import numpy as np
cimport numpy as np
from ctypes import c_size_t as size_t
from pele.potentials import _pele
from pele.potentials._pele cimport array_wrap_np, array_wrap_np_size_t, pele_array_to_np_size_t

#===============================================================================
# RandomCoordsDisplacement
#===============================================================================

cdef class _Cdef_RandomCoordsDisplacement(_Cdef_TakeStep):
    cdef cppRandomCoordsDisplacement* newptr
    def __cinit__(self, rseed, stepsize, report_interval=100, factor=0.9,
                  min_acc_ratio=0.2, max_acc_ratio=0.5, single=False,
                  nparticles=0, bdim=0, max_stepsize=0.0):
        if not single:
            self.newptr = <cppRandomCoordsDisplacement*> new cppRandomCoordsDisplacementAll(rseed, nparticles, bdim, stepsize, max_stepsize)
            self.thisptr = shared_ptr[cppTakeStep](<cppTakeStep*>
                   new cppAdaptiveTakeStep(shared_ptr[cppTakeStep](<cppTakeStep*> self.newptr),
                                           report_interval, factor, min_acc_ratio, max_acc_ratio))
        else:
            assert(bdim > 0 and nparticles > 0)
            self.newptr = <cppRandomCoordsDisplacement*> new cppRandomCoordsDisplacementSingle(rseed, nparticles, bdim, stepsize, max_stepsize)
            self.thisptr = shared_ptr[cppTakeStep](<cppTakeStep*>
                   new cppAdaptiveTakeStep(shared_ptr[cppTakeStep](<cppTakeStep*> self.newptr),
                                           report_interval, factor, min_acc_ratio, max_acc_ratio))

    def get_seed(self):
        """return random number generator seed

        Returns
        -------
        int
            random number generator seed
        """
        cdef res = self.newptr.get_seed()
        return res

    def set_generator_seed(self, input):
        """sets the random number generator seed

        Parameters
        ----------
        input : pos int
            random number generator seed
        """
        cdef inp = input
        self.newptr.set_generator_seed(inp)

    def get_count(self):
        """get the total count of the number of steps taken

        Returns
        -------
        int
            total count of steps taken
        """
        return self.newptr.get_count()

    def set_count(self, size_t input):
        """set the total count of the number of steps taken

        Parameters
        -------
        input : size_t
            total count of steps
        """
        self.newptr.set_count(input)

    def get_adaptation_counters(self):
        """get the counters defining the state of step size adaptation

        Returns
        -------
        NumPy array
            Counters defining the state of step size adaptation:
            0: m_total_steps, 1: m_accepted_steps
        """
        return pele_array_to_np_size_t((<cppAdaptiveTakeStep*>self.thisptr.get()).get_counters())

    def set_adaptation_counters(self, np.ndarray[size_t, ndim=1] input not None):
        """set the counters defining the state of step size adaptation

        Parameters
        -------
        input : NumPy array
            Counters defining the state of step size adaptation:
            0: m_total_steps, 1: m_accepted_steps
        """
        (<cppAdaptiveTakeStep*>self.thisptr.get()).set_counters(array_wrap_np_size_t(input))

    def get_stepsize(self):
        """get the step size

        Returns
        -------
        double
            stepsize
        """
        return self.newptr.get_stepsize()

    def set_stepsize(self, double input):
        """set the step size

        Parameters
        -------
        input : double
            stepsize
        """
        return self.newptr.set_stepsize(input)

class RandomCoordsDisplacement(_Cdef_RandomCoordsDisplacement):
    """Take a uniform random step in a ``bdim`` dimensional hypercube

    this class is the Python interface for the c++ RandomCoordsDisplacement implementation.
    Takes a step by sampling uniformly a ``bdim`` dimensional box

    Parameters
    ----------
    rseed : pos int
        seed for the random number generator (std:library 64 bits Merseene Twister)
    stepsize : double
        size of step in each dimension
    report_interval : int
        number of report steps for which the step size should be adapted
    factor : double
        factor by which the step size is adapted at each iteration durint the report interval
    min_acc_ratio : double
        minimum of target acceptance range
    max_acc_ratio: double
        maximum of target acceptance range
    single : bool
        True for single particle moves, False for global moves
    nparticles : int
        number of particles, typically len(coords)/bdim
    bdim : int
        dimensionality of the space (box dimensionality)
    """

#===============================================================================
# UniformSphericalSampling
#===============================================================================

cdef class _Cdef_UniformSphericalSampling(_Cdef_TakeStep):
    cdef cppUniformSphericalSampling* newptr
    cdef _pele.Array[double] corigin
    def __cinit__(self, rseed=42, radius=1, origin=None):
        self.radius = radius
        self.thisptr = shared_ptr[cppTakeStep](<cppTakeStep*> new cppUniformSphericalSampling(rseed, radius))
        self.newptr = <cppUniformSphericalSampling*> self.thisptr.get()
        if origin is not None:
            corigin = array_wrap_np(origin)
            self.newptr.set_origin(corigin)

    def set_generator_seed(self, input):
        """sets the random number generator seed

        Parameters
        ----------
        input : pos int
            random number generator seed
        """
        cdef inp = input
        self.newptr.set_generator_seed(inp)

    def get_stepsize(self):
        return self.radius

class UniformSphericalSampling(_Cdef_UniformSphericalSampling):
    """Sample uniformly at random inside N-ball.

    Implements the method described here:
    http://math.stackexchange.com/questions/87230/picking-random-points-in-the-volume-of-sphere-with-uniform-probability
    Variates $X_1$ to $X_N$ are sampled independently from a standard
    normal distribution and then rescaled as described in the reference.
    See also, e.g. Numerical Recipes, 3rd ed, page 1129.

    Parameters
    ----------
    rseed : pos int
        seed for the random number generator (std:library 64 bits Merseene Twister)
    radius : double
        radius of ball
    """

#===============================================================================
# UniformRectangularSampling
#===============================================================================

cdef class _Cdef_UniformRectangularSampling(_Cdef_TakeStep):
    cdef cppUniformRectangularSampling* newptr
    cdef _pele.Array[double] bv
    def __cinit__(self, rseed=42, delta=1, np.ndarray[double, ndim=1] boxvec=None):
        if boxvec is None:
            boxvec = np.array([2. * delta])
        bv = array_wrap_np(boxvec)
        self.thisptr = shared_ptr[cppTakeStep](<cppTakeStep*> new cppUniformRectangularSampling(rseed, bv))
        self.newptr = <cppUniformRectangularSampling*> self.thisptr.get()
    def set_generator_seed(self, input):
        """sets the random number generator seed

        Parameters
        ----------
        input : pos int
            random number generator seed
        """
        cdef inp = input
        self.newptr.set_generator_seed(inp)

class UniformRectangularSampling(_Cdef_UniformRectangularSampling):
    """Sample uniformly at random inside rectangle (prism etc.) centred
    at zero.

    If parameter delta is given (see below), coordinates are sampled
    uniformly at random in a n-dim cube of side length 2*delta. If
    instead boxvec is given (see below), coordinates of particles in
    len(boxvec)-dim space are sampled uniformly in the len(boxvec)-dim
    box specified by boxvec.

    Warning: There is no correlation between the coordinates at
    different MC steps. For a take-step module which at each step adds
    random displacements of a certain (average) stepsize to the
    coodinates, see, e.g. RandomCoordsDisplacement.

    Parameters
    ----------
    rseed : pos int
        seed for the random number generator (std-library 64 bits
        Merseene Twister)
    delta : double
        half side length of cube
    boxvec : array (optional)
        if set, sampling will be uniform in box volume
    """

#===============================================================================
# GaussianCoordsDisplacement
#===============================================================================

cdef class _Cdef_GaussianCoordsDisplacement(_Cdef_TakeStep):
    cdef cppGaussianTakeStep* newptr
    def __cinit__(self, rseed, stepsize, ndim):
        self.thisptr = shared_ptr[cppTakeStep](<cppTakeStep*> new cppGaussianCoordsDisplacement(rseed, stepsize, ndim))
        self.newptr = <cppGaussianTakeStep*> self.thisptr.get()

    def get_seed(self):
        """return random number generator seed

        Returns
        -------
        int
            random number generator seed
        """
        cdef res = self.newptr.get_seed()
        return res

    def set_generator_seed(self, input):
        """sets the random number generator seed

        Parameters
        ----------
        input : pos int
            random number generator seed
        """
        cdef inp = input
        self.newptr.set_generator_seed(inp)

    def get_count(self):
        """get the total count of the number of steps taken

        Returns
        -------
        int
            total count of steps taken
        """
        return self.newptr.get_count()

    def get_stepsize(self):
        """get the step size

        Returns
        -------
        double
            stepsize
        """
        return self.newptr.get_stepsize()

class GaussianCoordsDisplacement(_Cdef_GaussianCoordsDisplacement):
    """Take a uniform random step in a ``bdim`` dimensional hypersphere of radius ``stepsize``

    this class is the Python interface for the c++ GaussianCoordsDisplacement implementation.
    Takes a step by sampling uniformly a ``bdim`` dimensional hypersphere or radius ``stepsize``

    Parameters
    ----------
    rseed : pos int
        seed for the random number generator (std:library 64 bits Merseene Twister)
    stepsize : double
        size of step in each dimension
    ndim : int
        dimensionality of coordinates array
    """

cdef class _Cdef_SampleGaussian(_Cdef_TakeStep):
    cdef cppGaussianTakeStep* newptr
    def __cinit__(self, rseed, stepsize, origin):
        cdef _pele.Array[double] origin_ = array_wrap_np(origin)
        self.thisptr = shared_ptr[cppTakeStep](<cppTakeStep*> new cppSampleGaussian(rseed, stepsize, origin_))
        self.newptr = <cppGaussianTakeStep*> self.thisptr.get()

    def get_seed(self):
        """return random number generator seed

        Returns
        -------
        int
            random number generator seed
        """
        cdef res = self.newptr.get_seed()
        return res

    def set_generator_seed(self, input):
        """sets the random number generator seed

        Parameters
        ----------
        input : pos int
            random number generator seed
        """
        cdef inp = input
        self.newptr.set_generator_seed(inp)

    def get_count(self):
        """get the total count of the number of steps taken

        Returns
        -------
        int
            total count of steps taken
        """
        return self.newptr.get_count()

    def set_count(self, size_t input):
        """set the total count of the number of steps taken

        Parameters
        -------
        input : size_t
            total count of steps
        """
        self.newptr.set_count(input)

    def get_stepsize(self):
        """get the step size

        Returns
        -------
        double
            stepsize
        """
        return self.newptr.get_stepsize()

    def set_stepsize(self, double input):
        """set the step size

        Parameters
        -------
        input : double
            stepsize
        """
        return self.newptr.set_stepsize(input)

class SampleGaussian(_Cdef_SampleGaussian):
    """Sample directly a Gaussian centered at ``origin`` with standard deviation ``stepsize``

    this class is the Python interface for the c++ SampleGaussian implementation.
    Sample directly a Gaussian centered at ``origin`` with standard deviation ``stepsize``

    Parameters
    ----------
    rseed : pos int
        seed for the random number generator (std:library 64 bits Merseene Twister)
    stepsize : double
        standard deviation for direct sampling
    origin : numpy.array
        coordinates where the gaussian should be centered
    """

# Adaptive Swap
cdef class _Cdef_AdaptiveSwap(_Cdef_TakeStep):
    cdef cppTakeStep* newptr
    def __cinit__(self, seed, n_particles, dim, report_interval=100, factor=0.9,
                  min_acc_ratio=0.2, max_acc_ratio=0.5):
        self.newptr = <cppTakeStep*> new cppParticlePairSwap(seed, n_particles, dim)
        self.thisptr = shared_ptr[cppTakeStep](<cppTakeStep*>
        new cppAdaptiveTakeStep(shared_ptr[cppTakeStep](<cppTakeStep*> self.newptr),
                                report_interval, factor, min_acc_ratio, max_acc_ratio))
        

class AdaptiveSwap(_Cdef_AdaptiveSwap):
    """Adaptive Swap

    this class is the Python interface for the c++ AdaptiveSwap implementation.
    Takes a step by swapping two particles. The swaps are only chosen.
    if the difference between the radii of the two particles being swapped
    is less than a certain threshold i.e radius1 - radius2 < threshold.
    
    
    This threshold is adaptively adjusted
    until the acceptance ratio is between min_acc_ratio and max_acc_ratio.
    
    If the acceptance ratio is above max_acc_ratio, and the maximum radius
    difference between any two particles **is less or equal** to the threshold
    then the threshold is not increased, because doing so will not increase the
    acceptance ratio.
    if the acceptance ratio is below min_acc_ratio, and the threshold is so 
    small that no other particle has a radius that meets the criterion
    radius1 - radius2 < threshold, then the particle 1 is deterministically 
    swapped with the particle with the closest radius
    

    Parameters
    ----------
    seed : pos int
        seed for the random number generator (std:library 64 bits Merseene Twister)
    n_particles : int
        number of particles
    dim : int
        dimensionality of coordinates array
    report_interval : int
        interval for reporting the acceptance ratio
    factor : double
        factor for adjusting the stepsize
    min_acc_ratio : double
        minimum acceptance ratio
    max_acc_ratio : double
        maximum acceptance ratio
"""


cdef class _Cdef_ParticlePairSwap(_Cdef_TakeStep):
    cdef cppParticlePairSwap* newptr
    def __cinit__(self, seed, nr_particles, bdim):
        self.thisptr = shared_ptr[cppTakeStep](<cppTakeStep*> new cppParticlePairSwap(seed, nr_particles, bdim))
        self.newptr = <cppParticlePairSwap*> self.thisptr.get()

    def get_seed(self):
        """return random number generator seed

        Returns
        -------
        int
            random number generator seed
        """
        cdef res = self.newptr.get_seed()
        return res

    def set_generator_seed(self, input):
        """sets the random number generator seed

        Parameters
        ----------
        input : pos int
            random number generator seed
        """
        cdef inp = input
        self.newptr.set_generator_seed(inp)

class ParticlePairSwap(_Cdef_ParticlePairSwap):
    """Swap a pair of particles

    Python interface for c++ ParticlePairSwap

    Parameters
    ----------
    seed : pos integer
        Seed for random number generator.
    nr_particles : pos integer
        Number of particles.
    swap_every : pos integer
        Spacing for swapping attempts: particle pair swap is attempted every
        ``swap_every`` move.
    """

#
# TakeStepPattern
#

cdef class _Cdef_TakeStepPattern(_Cdef_TakeStep):
    cdef cppTakeStepPattern* newptr
    def __cinit__(self):
        self.thisptr = shared_ptr[cppTakeStep](<cppTakeStep*> new cppTakeStepPattern())
        self.newptr = <cppTakeStepPattern*> self.thisptr.get()

    def add_step(self, _Cdef_TakeStep step, nr_repetitions):
        """add a step to a pattern

        Parameters
        ----------
        step : :class:`TakeStep`
            object of class :class:`TakeStep` constructed beforehand
        nr_repetitions: int
            number of Monte Carlo iterations in a row for which this
            move is performed
        """
        self.newptr.add_step(step.thisptr, nr_repetitions)

class TakeStepPattern(_Cdef_TakeStepPattern):
    """Takes multiple steps in a repeated pattern

    Python interface for c++ TakeStepPattern. This move
    takes multiple steps in a repeated deterministic pattern.

    .. warning:: breaks detailed balance locally
    """

#
# TakeStepProbabilities
#

cdef class _Cdef_TakeStepProbabilities(_Cdef_TakeStep):
    cdef cppTakeStepProbabilities* newptr
    def __cinit__(self, seed):
        self.thisptr = shared_ptr[cppTakeStep](<cppTakeStep*> new cppTakeStepProbabilities(seed))
        self.newptr = <cppTakeStepProbabilities*> self.thisptr.get()
    def add_step(self, _Cdef_TakeStep step, weight):
        """add a step to a pattern

        all the weights are combined in a normalised discrete distribution

        Parameters
        ----------
        step : :class:`TakeStep`
            object of class :class:`TakeStep` constructed beforehand
        weight: double
            weight to assign to each move
        """
        self.newptr.add_step(step.thisptr, weight)

class TakeStepProbabilities(_Cdef_TakeStepProbabilities):
    """Takes multiple steps in a repeated pattern

    Python interface for c++ TakeStepProbabilities. This move
    takes multiple steps, each with some probability thus
    not affecting the detailed balance condition.

    .. note:: it does NOT break detailed balance
              hence it is the recommended choice
    """
