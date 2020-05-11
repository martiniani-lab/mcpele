#ifndef _MCPELE_MC_WANG_LANDAU_H
#define _MCPELE_MC_WANG_LANDAU_H



/**
 *  Defines action and update that collectively define the wang landau class of monte carlo
 *  
 *  
 */


#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <random>

#include "mc.h"
#include "pele/array.h"
#include "pele/base_potential.h"
#include "histogram.h"



// Wang Landau Necessities

namespace mcpele {




/**
 * This records the Wang Landau actions for 1. the histogram dos updates 2. the histogram flatness check. 3. log modifier threshold which determines the end of the WL simulation These are combined since the wang landau updater that goes with the accept test
 */
class WL_Updater : public Action {
protected:
    double m_log_f;
    double m_modifier;
    const double m_log_f_threshold;
    const double m_flatness_criterion;
    
    virtual void action(pele::Array<double> &coords, double energy, bool accepted, MC* mc);
    
    // Inline helper functions for action
    bool flat_histogram_check();            // Checks for the flat histogram and updates accordingly
    inline bool modification_factor_check();       // 
    void hist_dos_update(double energy);//  updates density of states and the histogram
    size_t hiter;  // Keeps track of histogram flatness iterations
    size_t m_h_iter;            // no of MC iterations after which a flat histogram is checked for
    double m_max;
    double m_min;
    double m_bin;
    double m_eps;
    int m_N;
    std::vector<bool> visit_tracker;
    std::vector<double> log_dos_vec;           // log density of states
    Histogram H_e;               // Energy Histogram
public:
    WL_Updater(double emin, double emax, double bin, double log_f, double modifier, double log_f_threshold, double flatness_criterion, size_t h_iter);
    virtual ~WL_Updater(){};
    pele::Array<double> get_log_dos() {return pele::Array<double>(log_dos_vec);}
    double get_log_dos_en(double energy);            // Gets the log dos for the given energy
};

 // * Wang Landau Accept Test. needs a pointer to the WL_UpdateAction since there's no point of wang landau without density of states
class WL_AcceptTest : public AcceptTest {
protected:
    size_t m_seed;
    double m_eps;
    std::mt19937_64 m_generator;
    std::uniform_real_distribution<double> m_distribution;
    std::shared_ptr<WL_Updater> m_wl_update;
public:
    WL_AcceptTest(const size_t rseed, std::shared_ptr<WL_Updater> wl_update);
    virtual ~WL_AcceptTest() {}
    virtual bool test(pele::Array<double> &trial_coords, double trial_energy,
            pele::Array<double> & old_coords, double old_energy, double temperature,
            MC * mc);
    size_t get_seed() const {return m_seed;}
    void set_generator_seed(const size_t inp) { m_generator.seed(inp); }
};




} // namespace mcpele











#endif