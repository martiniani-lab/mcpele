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
  const bool m_reflect_boundary;
  const bool m_reflect_potential;

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
      bool use_random_timestep = false, size_t adaptive_interval = 100,
      double adaptive_factor = 0.9, double adaptive_min_acceptance_ratio = 0.2,
      double adaptive_max_acceptance_ratio = 0.5,
      bool reflect_boundary = true, bool reflect_potential = false);
  ~GMC() override = default;
  void one_iteration() override;
  void check_input() override;
  std::shared_ptr<TakeStep> get_takestep() const override { return m_take_step; }

  void add_accept_test(const std::shared_ptr<AcceptTest> &accept_test) {
    m_accept_tests.push_back(accept_test);
  }
  void add_conf_test(const std::shared_ptr<GMCConfTest> &conf_test) {
    m_conf_tests.push_back(conf_test);
  }
  void add_late_conf_test(const std::shared_ptr<GMCConfTest> &conf_test) {
    m_late_conf_tests.push_back(conf_test);
  }
  const std::vector<size_t> get_changed_atoms() const override {
    throw std::runtime_error("GMC::get_changed_atoms: not implemented");
  }
  const std::vector<double> get_changed_coords_old() const override {
    throw std::runtime_error("GMC::get_changed_coords_old: not implemented");
  }
};

}  // namespace mcpele

#endif  //_MCPELE_GMC_H
