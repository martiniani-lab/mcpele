// Action/update that collectively define wang landau
#include "mcpele/mc_wang_landau.h"
#include <cmath>
#include <algorithm>

using pele::Array;



namespace mcpele {


    WL_Updater::WL_Updater(double emin, double emax, double bin,
                           double log_f, double modifier,
                           double log_f_threshold, double flatness_criterion, size_t h_iter)
        : H_e(emin, emax, bin),
          hiter(0),
          m_min(floor((emin / bin)) * bin),
          m_max(floor((emax / bin) + 1) * bin),
          m_bin(bin),
          m_h_iter(h_iter),
          m_N((m_max - m_min) / bin),
          log_dos_vec(m_N, 0),
          m_log_f(log_f),
          visit_tracker(m_N, false),
          m_modifier(modifier),
          m_eps(std::numeric_limits<double>::epsilon()),
          m_log_f_threshold(log_f_threshold),
          m_flatness_criterion(flatness_criterion)
    {}
    // This is written assuming it happens once every few xyz steps, and does not happen at first
    bool WL_Updater::flat_histogram_check() {
        // This is shitty replace pele array at some point.
        std::vector<double> hist = H_e.get_hist();
        double mean = 0;
        double min = std::numeric_limits<double>::max();

        for (int i = 0; i< hist.size(); i++) {
            if (visit_tracker[i]) {
                mean += hist[i];
                if (min<hist[i]) {min=hist[i];}
            }
        }
        if (min > m_flatness_criterion*mean) {
            return true;
        }
        else {
            return false;
        }
    }
    inline bool WL_Updater::modification_factor_check() {
        if (m_log_f_threshold>m_log_f) {
            return true;
        }
        else {
            return false;
        }                 
    }

    
    void WL_Updater::hist_dos_update(double energy) {
        energy = energy + m_eps;
        int i;
        i = floor((energy - m_min)/m_bin);
        if (i > m_N && i >=0) {
            std::cout << energy << "energy value generated \n";
            throw std::runtime_error("Not implemented when energy is outside the range yet");
        }
        else {
            H_e.add_entry(energy);
            log_dos_vec[i] += m_log_f;
            if (!(visit_tracker[i])) {
                visit_tracker[i] = true;
            }
        }
    }


    double WL_Updater::get_log_dos_en(double energy) {
        energy = energy + m_eps;
        int i;
        i = floor((energy - m_min)/m_bin);
        return log_dos_vec[i];
    }
    

    void WL_Updater::action(pele::Array<double> &coords, double energy, bool accepted, MC *mc) {
        // update histogram and density of states
        hist_dos_update(energy);
        // after every m_h_iter run a flat histogram check note:: flat histogram check needs to happen after n steps
        if (mc->m_niter>0 && mc->m_niter % m_h_iter == 0) {
            if (flat_histogram_check()) {
                H_e.reset();
                m_log_f -= m_modifier;
                hiter++;
                // if log f falls below threshold break out of simulation
                if (m_log_f_threshold>m_log_f) {
                    mc->m_niter = mc->m_max_iter;
                }
            }
        } 
    }
    
    
    WL_AcceptTest::WL_AcceptTest(const size_t rseed, std::shared_ptr<WL_Updater> wl_update)
            : m_seed(rseed),
            m_generator(rseed),
            m_distribution(0.0, 1.0),
            m_wl_update(wl_update)
            {
#ifdef DEBUG
                std::cout << "seed wang landau:" << m_seed << "\n";
                //std::chrono::system_clock::now().time_since_epoch().count()
#endif
            }
    // 
    bool WL_AcceptTest::test(Array<double> &trial_coords, double trial_energy,
                              Array<double>& old_coords, double old_energy, double temperature,
                              MC * mc)
    {
        double w, rand; 
        bool success = true;
        double log_dos_diff = m_wl_update->get_log_dos_en(old_energy) - m_wl_update->get_log_dos_en(trial_energy);
        if (log_dos_diff > 0.){
            double w = exp(log_dos_diff);
            rand = m_distribution(m_generator);
            if (rand > w) {
                success = false;
            }
        }
        return success;
    }
    

} // namespace mcpele