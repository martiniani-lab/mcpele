#ifndef _MCPELE_GMC_H
#define _MCPELE_GMC_H

#include <utility>

#include "adaptive_takestep.h"
#include "gmc_take_step.h"
#include "mc.h"

namespace mcpele {

class GMCConfTest : public ConfTest {
 public:
  ~GMCConfTest() override = default;
  virtual pele::Array<double> gmc_gradient(pele::Array<double> &coords,
                                           MCBase *mc) = 0;
};

class GMC final : public MCBase {
 public:
  typedef std::vector<std::shared_ptr<AcceptTest>> accept_t;
  typedef std::vector<std::shared_ptr<GMCConfTest>> conf_t;

 protected:
  accept_t m_accept_tests;
  conf_t m_conf_tests;
  conf_t m_late_conf_tests;
  std::shared_ptr<GMCTakeStep> m_take_step;
  std::shared_ptr<AdaptiveTakeStep> m_adaptive_step;
  pele::Array<double> accumulated_gradient;
  size_t m_E_reject_count;
  size_t m_conf_reject_count;
  size_t resample_velocity_steps;

  bool do_conf_tests(pele::Array<double> &x) {
    for (const auto &test : m_conf_tests) {
      if (const bool result = test->conf_test(x, this); not result) {
        return false;
      }
    }
    return true;
  }
  bool do_accept_tests(pele::Array<double> &trial_coords, double trial_energy,
                       pele::Array<double> &old_coords, double old_energy) {
    for (const auto &test : m_accept_tests) {
      if (const bool result = test->test(trial_coords, trial_energy, old_coords,
                                         old_energy, m_temperature, this);
          not result) {
        return false;
      }
    }
    return true;
  }
  bool do_late_conf_tests(pele::Array<double> &x) {
    for (const auto &test : m_late_conf_tests) {
      if (const bool result = test->conf_test(x, this); not result) {
        return false;
      }
    }
    return true;
  }
  bool check_configuration_short(pele::Array<double> &trial_coords);

 public:
  GMC(std::shared_ptr<pele::BasePotential> potential,
      const pele::Array<double> &coords, double temperature, double timestep,
      size_t nparticles, size_t ndim, size_t rseed,
      size_t resample_velocity_steps = 0, double max_timestep = 0.0,
      size_t adaptive_interval = 100, double adaptive_factor = 0.9,
      double adaptive_min_acceptance_ratio = 0.2,
      double adaptive_max_acceptance_ratio = 0.5);
  ~GMC() override = default;
  void one_iteration() override;
  void check_input() override;

  void add_accept_test(const std::shared_ptr<AcceptTest> &accept_test) {
    m_accept_tests.push_back(accept_test);
  }
  void add_conf_test(const std::shared_ptr<GMCConfTest> &conf_test) {
    m_conf_tests.push_back(conf_test);
  }
  void add_late_conf_test(const std::shared_ptr<GMCConfTest> &conf_test) {
    m_late_conf_tests.push_back(conf_test);
  }
};

}  // namespace mcpele

#endif  //_MCPELE_GMC_H
