from __future__ import division
from builtins import range
import abc
import numpy as np
import random
import os
from mpi4py import MPI
from mcpele.parallel_tempering import _MPI_Parallel_Tempering
import time
import logging


def trymakedir(path):
    """this function deals with common race conditions"""
    while True:
        if not os.path.exists(path):
            try:
                os.makedirs(path)
                break
            except OSError as e:
                if e.errno != 17:
                    raise
                # time.sleep might help here
                pass
        else:
            break


class MPI_PT_RLhandshake(_MPI_Parallel_Tempering):
    """Perform Parallel Tempering by a right-left handshake

    This class performs parallel tempering alternating swaps with right and left neighbours
    with geometrically distributed temperatures.

    Parameters
    ----------
    mcrunner : :class:`_BaseMCrunner`
        object of :class:`_BaseMCrunner` that performs
        the MCMC walks
    Tmax : double
        maximum temperature to simulate (or equivalent control parameters)
    Tmin : double
        minimum temperature to simulate (or equivalent control parameters)
    max_ptiter : int
        maximum number of Parallel Tempering iterations
    pfreq : int
        frequency with which histogram and other information is dumped
        to a file
    skip : int
        number of parallel tempering iteration for which swaps should
        not be performed. Swaps should be avoided for instance while
        adjusting the step size
    print_status : bool
        choose whether to print MCrunner status at each iteration
    base_directory : string
        path to base directory where to save output
    suppress_histogram : bool
        suppress histogram output

    Attributes
    ----------
    exchange_dic : dictionary
        assign -1 to left and 1 to right
    exchange_choice : int
        current swap choice (alternating)
        initialised randomlly to be 1 or -1
    anyswap : bool
        set to True is any of the attempted swaps
        have been succesfull
    permutation_pattern : numpy.array
        record pattern of exchanges, used to print
        the exchange permutations
    exchange_cnts : numpy.array
        record number of exchanges per replica pair
    suppress_histgoram : bool
        suppress the output of the histogram
    """

    def __init__(
        self,
        mcrunner,
        Tmax,
        Tmin,
        max_ptiter=10,
        pfreq=1,
        skip=0,
        print_status=True,
        base_directory=None,
        suppress_histogram=True,
    ):
        super(MPI_PT_RLhandshake, self).__init__(
            mcrunner,
            Tmax,
            Tmin,
            max_ptiter,
            pfreq=pfreq,
            skip=skip,
            print_status=print_status,
            base_directory=base_directory,
        )
        i32max = np.iinfo(np.int32).max
        self.seed_exchanges = random.randint(0, i32max)
        np.random.seed(self.seed_exchanges)
        logging.info("seed_exchanges: %i" % self.seed_exchanges)
        self.exchange_dic = {1: "right", -1: "left"}
        self.exchange_choice = np.random.choice(list(self.exchange_dic.keys()))
        self.anyswap = False  # set to true if any swap will happen
        self.permutation_pattern = np.zeros(
            self.nprocs, dtype="int32"
        )  # this is useful to print exchange permutations
        self.exchange_cnts = np.zeros(self.nprocs - 1, dtype="int32")
        self.suppress_histogram = suppress_histogram

    def _print_data(self):
        self._all_dump_histogram()

    def _print_status(self):
        self._all_print_status()

    def _print_initialise(self):
        base_directory = self.base_directory
        trymakedir(base_directory)
        directory = "{0}/{1}".format(base_directory, self.rank)
        trymakedir(directory)
        self._master_print_temperatures()
        self._all_print_parameters()
        self.status_stream = open("{0}/{1}".format(directory, "status"), "w")
        self.histogram_mean_stream = open("{0}/{1}".format(directory, "hist_mean"), "w")
        self.histogram_mean_stream.write("{:<15}\t{:<15}\n".format("iteration", "<E>"))
        if self.rank == 0:
            self.permutations_stream = open(
                r"{0}/rem_permutations".format(base_directory), "w"
            )

    def _close_flush(self):
        self.histogram_mean_stream.flush()
        self.histogram_mean_stream.close()
        self.status_stream.flush()
        self.status_stream.close()
        if self.rank == 0:
            self.permutations_stream.flush()
            self.permutations_stream.close()

    def _master_print_temperatures(self):
        base_directory = self.base_directory
        if self.rank == 0:
            fname = "{0}/temperatures".format(base_directory)
            np.savetxt(fname, self.Tarray, delimiter="\t", fmt="%1.16f")

    def _all_print_parameters(self):
        base_directory = self.base_directory
        directory = "{0}/{1}".format(base_directory, self.rank)
        fname = "{0}/{1}".format(directory, "parameters")
        # init_stepsize = self.mcrunner.stepsize
        ncount = self.mcrunner.niter
        f = open(fname, "w")
        f.write("node:\t{0}\n".format(self.rank))
        f.write("temperature:\t{0}\n".format(self.T))
        # f.write('initial step size:\t{0}\n'.format(init_stepsize))
        f.write("PT iterations:\t{0}\n".format(self.max_ptiter))
        f.write("total MC iterations:\t{0}\n".format(ncount))
        f.close()

    def _master_print_permutations(self):
        if self.rank == 0 and self.anyswap == True:
            iteration = self.mcrunner.get_iterations_count()
            f = self.permutations_stream
            f.write("{0}\t".format(iteration))
            for p in self.permutation_pattern:
                f.write("{0}\t".format(p))
            f.write("\n")
            f.flush()

    def _all_dump_histogram(self):
        """for this to work the directory must have been initialised in _print_initialise"""
        base_directory = self.base_directory
        directory = "{0}/{1}".format(base_directory, self.rank)
        iteration = self.mcrunner.get_iterations_count()
        fname = "{0}/Visits.his.{1}".format(directory, float(iteration))
        if not self.suppress_histogram:
            mean, variance = self.mcrunner.dump_histogram(fname)
        else:
            mean, variance = self.mcrunner.histogram.get_mean_variance()
        self.histogram_mean_stream.write(
            "{:<15}\t{:>15.15e}\t{:>15.15e}\n".format(iteration, mean, variance)
        )

    def _all_print_status(self):
        status = self.mcrunner.get_status()
        # logging.debug(float(self.swap_accepted_count) / (self.swap_accepted_count+self.swap_rejected_count))
        nswaps = self.swap_accepted_count + self.swap_rejected_count
        if nswaps == 0:
            status.frac_acc_swaps = 1.0
        else:
            status.frac_acc_swaps = float(self.swap_accepted_count) / nswaps
        f = self.status_stream
        if self.ptiter == self.skip:
            f.write("#")
            for key, value in status.items():
                f.write("{:<12}\t".format(key))
            f.write("\n")
        for key, value in status.items():
            f.write("{:>12.3f}\t".format(value))
        f.write("\n")

    def _print_exchanges(self):
        if self.rank == 0:
            logging.info("Number of exchanges:")
            for i in range(self.nprocs - 1):
                logging.info(
                    "{0:>2} <-> {1:<2}:{2:>6}".format(i, i + 1, self.exchange_cnts[i])
                )

    def _get_temps(self):
        """
        set up the temperatures by distributing them exponentially. We give root the lowest temperature.
        This should increase performance when pair lists are used (they are updated less often at low temperature
        or when steps involve minimisation, as the low temperatures are closer to the minimum)
        """
        if self.rank == 0:
            CTE = np.exp(np.log(self.Tmax / self.Tmin) / (self.nprocs - 1))
            Tarray = [self.Tmin * CTE**i for i in range(self.nprocs)]
            # Tarray = np.linspace(self.Tmin,self.Tmax,num=self.nprocs)
            self.Tarray = np.array(Tarray[::-1], dtype="d")
        else:
            self.Tarray = None

    def _initialise(self):
        """
        perform all the tasks required prior to starting the computation
        """
        self._get_temps()
        self.T = self._scatter_single_value(self.Tarray)
        logging.debug("Temperature {}".format(self.T))
        self.mcrunner.set_control(self.T)
        self.config, self.energy = self.mcrunner.get_config()
        self._print_initialise()
        self.initialised = True

    def _find_exchange_buddy(self, Earray):
        """
        This function determines the exchange pattern alternating swaps with right and left neighbours.
        An exchange pattern array is constructed, filled with self.no_exchange_int which
        signifies that no exchange should be attempted. This value is replaced with the
        rank of the process with which to perform the swap if the swap attempt is successful.
        The exchange partner is then scattered to the other processes.
        """
        if self.rank == 0:
            assert len(Earray) == len(self.Tarray)
            exchange_pattern = np.empty(len(Earray), dtype="int32")
            exchange_pattern.fill(self.no_exchange_int)
            self.anyswap = False
            for i in range(0, self.nprocs, 2):
                logging.debug(
                    "Exchange choice: {}".format(
                        self.exchange_dic[self.exchange_choice]
                    )
                )  # this is a print statement that has to be removed after initial implementation
                E1 = Earray[i]
                T1 = self.Tarray[i]
                E2 = Earray[i + self.exchange_choice]
                T2 = self.Tarray[i + self.exchange_choice]
                deltaE = E1 - E2
                deltabeta = 1.0 / T1 - 1.0 / T2
                w = min(1.0, np.exp(deltaE * deltabeta))
                rand = np.random.rand()
                # logging.debug("E1 {0} T1 {1} E2 {2} T2 {3} w {4}".format(E1,T1,E2,T2,w))
                if w > rand:
                    self.exchange_cnts[i + min(0, self.exchange_choice)] += 1
                    # accept exchange
                    if logging.getLogger().isEnabledFor(logging.DEBUG):
                        self.ex_outstream.write(
                            "accepting exchange %d %d %g %g %g %g %d\n"
                            % (
                                self.nodelist[i],
                                self.nodelist[i + self.exchange_choice],
                                E1,
                                E2,
                                T1,
                                T2,
                                self.ptiter,
                            )
                        )
                    assert (
                        exchange_pattern[i] == self.no_exchange_int
                    )  # verify that is not using the same process twice for swaps
                    assert (
                        exchange_pattern[i + self.exchange_choice]
                        == self.no_exchange_int
                    )  # verify that is not using the same process twice for swaps
                    exchange_pattern[i] = self.nodelist[i + self.exchange_choice]
                    exchange_pattern[i + self.exchange_choice] = self.nodelist[i]
                    self.anyswap = True
            ############end of for loop###############
            # record self.permutation_pattern to print permutations in print function
            if self.anyswap:
                for i, buddy in enumerate(exchange_pattern):
                    if buddy != self.no_exchange_int:
                        self.permutation_pattern[i] = (
                            buddy + 1
                        )  # to conform to fortran notation
                    else:
                        self.permutation_pattern[i] = (
                            i + 1
                        )  # to conform to fortran notation
                if self._print_status:
                    self._master_print_permutations()
        else:
            exchange_pattern = None

        self.exchange_choice *= -1  # swap direction of exchange choice
        return exchange_pattern
