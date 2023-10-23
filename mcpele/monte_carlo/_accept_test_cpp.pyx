# distutils: language = c++
# distutils: sources = accept_test.cpp

cimport cython
cimport numpy as np
from collections import Iterator
import numpy as np
import sys
from pele.potentials._pele cimport array_wrap_np

#===============================================================================
# Metropolis acceptance criterion
#===============================================================================

cdef class _Cdef_Metropolis(_Cdef_AcceptTest):
    cdef cppMetropolisTest* newptr
    def __cinit__(self, rseed):
        self.thisptr = shared_ptr[cppAcceptTest](<cppAcceptTest*> new cppMetropolisTest(rseed))
        self.newptr = <cppMetropolisTest*> self.thisptr.get()
    
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
        
class MetropolisTest(_Cdef_Metropolis):
    """Metropolis acceptance criterion
    
    This class is the Python interface for the c++ mcpele::MetropolisTest 
    acceptance test class implementation. The Metropolis acceptance criterion
    accepts each move with probability
    
    .. math:: P( x_{old} \Rightarrow x_{new}) = min \{ 1, \exp [- \\beta (E_{new} - E_{old})] \}
    
    where :math:`\\beta` is the reciprocal of the temperature.
    """
    
#==========
# CloudTest
#==========

cdef class _Cdef_CloudTest(_Cdef_AcceptTest):
    cdef cppCloudTest* newptr
    cdef _pele.BasePotential bias
    def __cinit__(self, rseed, cloud_seed, nr_cloud_points, cloud_radius, _pele.BasePotential bias_=None):
        if bias_ is None:
            self.thisptr = shared_ptr[cppAcceptTest](<cppAcceptTest*> new cppCloudTest(rseed, cloud_seed, nr_cloud_points, cloud_radius))
        else:
            self.bias = bias_
            self.thisptr = shared_ptr[cppAcceptTest](<cppAcceptTest*> new cppCloudTest(rseed, cloud_seed, nr_cloud_points, cloud_radius, self.bias.thisptr))
        self.newptr = <cppCloudTest*> self.thisptr.get()
        
    def set_generator_seeds(self, rseed, cloud_seed):
        """sets the random number generator seed
        
        Parameters
        ----------
        input : pos int
            random number generator seed
        """
        cdef inp = input
        self.newptr.set_generator_seeds(rseed, cloud_seed)

    def add_conf_test(self, _Cdef_ConfTest test):
        """add conf test to be run on all cloud drops

        Parameters
        ----------
        test : :class:`ConfTest`
            class of type :class:`ConfTest`, constructed beforehand
        """
        self.newptr.add_conf_test(test.thisptr)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def get_old_cloud(self):
        cdef Cloud cloud = self.newptr.get_old_cloud()
        cdef size_t ndrops = cloud.size()
        cdef size_t ndof = cloud.at(0).get().x.size()
        cdef np.ndarray[double, ndim=2, mode="c"] cloud_x = np.zeros((ndrops, ndof))
        cdef np.ndarray[double, ndim=1, mode = "c"] cloud_oracle = np.zeros(ndrops)
        cdef np.ndarray[double, ndim=1, mode = "c"] cloud_bias = np.zeros(ndrops)
        cdef double *xdata
        for i in xrange(ndrops):
            cloud_oracle[i] = cloud.at(i).get().oracle
            cloud_bias[i] = cloud.at(i).get().bias
            xdata = cloud.at(i).get().x.data()
            for j in xrange(ndof):
                cloud_x[i][j] = xdata[j]
        return PyCloud.__new__(PyCloud, cloud_x, cloud_oracle, cloud_bias)


    @cython.boundscheck(False)
    @cython.wraparound(False)
    def set_cloud(self, pycloud):
        cdef size_t nr_drops = pycloud.size
        cdef Cloud cloud = Cloud(nr_drops)
        for i,c in enumerate(pycloud):
            cloud.at(i).get().x = array_wrap_np(c[0]).copy()
            cloud.at(i).get().oracle = <cbool> c[1]
            cloud.at(i).get().bias = <double> c[2]
        self.newptr.set_cloud(cloud)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def set_cloud_bias(self, pycloud):
        cdef _pele.Array[double] cbias =  array_wrap_np(pycloud.bias)
        self.newptr.set_cloud_bias(cbias)

class CloudTest(_Cdef_CloudTest):
    """CloudTest acceptance criterion
    """


cdef class _Cdef_Cloud(object):
    cdef double [:,:] x
    cdef long [:] oracle
    cdef double [:] bias
    cdef size_t current
    cdef public size_t size, ndof
    def __cinit__(self, x, oracle, bias):
        self.x = np.array(x, dtype='d')
        self.oracle = np.array(oracle, dtype='int')
        self.bias = np.array(bias, dtype='d')
        self.size = oracle.size
        self.ndof = self.x[0].size

    def __richcmp__(self, other, op):
        if op == 2:
            if isinstance(other, _Cdef_Cloud):
                if np.array_equal(self.x, other.x) and np.array_equal(self.oracle, other.oracle) \
                        and np.array_equal(self.bias, other.bias):
                    return True
                else:
                    return False
            else:
                return NotImplemented
        elif op == 3:
            if isinstance(other, _Cdef_Cloud):
                if np.array_equal(self.x, other.x) and np.array_equal(self.oracle, other.oracle) \
                        and np.array_equal(self.bias, other.bias):
                    return False
                else:
                    return True
            else:
                return NotImplemented
        else:
            raise NotImplementedError

    def __len__(self):
        return self.size

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __iter__(self):
        self.current = 0
        return self

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __setitem__(self, idx, drop):
        cdef double [:] x_view = drop[0]
        self.x[idx,:] = x_view
        self.oracle[idx] = <long> drop[1]
        self.bias[idx] = <double> drop[2]

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def next(self):
        if self.current > self.size-1:
            raise StopIteration
        else:
            x = np.asarray(self.x[self.current], dtype='d')
            o = self.oracle[self.current]
            b = self.bias[self.current]
            self.current += 1
            return (x, o, b)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __getitem__(self, item):
        x = np.asarray(self.x[item], dtype='d')
        o = self.oracle[item]
        b = self.bias[item]
        return (x, o, b)

    def __str__(self):
        mystr = []
        for i in xrange(self.size):
            mystr.append("drop {} \ncoords: {} \noracle: {} \nbias: {} \n".format(i, np.asarray(self.x[i]),
                                                                                  self.oracle[i], self.bias[i]))
        return '\n'.join(map(str, mystr))

    @property
    def x(self):
        return np.asarray(self.x)

    @property
    def oracle(self):
        return np.asarray(self.oracle)

    @property
    def bias(self):
        return np.asarray(self.bias)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @x.setter
    def x(self, val):
        cdef double [:,:] val_view = val
        self.x[...] = val_view

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @oracle.setter
    def oracle(self, val):
        cdef long [:] val_view = val
        self.oracle[...] = val_view

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @bias.setter
    def bias(self, val):
        cdef double [:] val_view = val
        self.bias[...] = val_view

    def rosenbluth_weight(self):
        """
        computes the rosenbluth weight from the imported bias
        """
        oracle = np.asarray(self.oracle, dtype='d')
        bias = np.asarray(self.bias, dtype='d')
        return np.sum(oracle * bias)

class PyCloud(_Cdef_Cloud):
    """
    Python wrapper to _Cdef_Cloud
    """

    def __reduce__(self):
        return (PyCloud, (self.x, self.oracle, self.bias,))