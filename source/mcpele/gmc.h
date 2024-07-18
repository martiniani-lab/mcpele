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
  void run(size_t max_iter) override;
  void one_iteration() override;
  void check_input() override;
  // TODO: Is this method really necessary in basinvolume?
  std::shared_ptr<TakeStep> get_takestep() const override {
    return m_take_step;
  }

  // TODO: If this is required in basinvolume, shouldn't that really be in
  // MCBase?
  double get_timestep() const { return m_take_step->get_timestep(); }
  void set_timestep(const double input) const {
    m_take_step->set_timestep(input);
  }
  size_t get_count() const { return m_take_step->get_count(); }
  void set_count(const size_t input) const { m_take_step->set_count(input); }
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
  pele::Array<size_t> get_adaptation_counters() const {
    return m_adaptive_step->get_counters();
  }
  void set_adaptation_counters(pele::Array<size_t> const &counters) const {
    m_adaptive_step->set_counters(counters);
  }
  pele::Array<size_t> get_counters() const {
    pele::Array<size_t> counters(5);
    counters[0] = m_nitercount;
    counters[1] = m_accept_count;
    counters[2] = m_E_reject_count;
    counters[3] = m_conf_reject_count;
    counters[4] = m_neval;
    return counters;
  }
  void set_counters(pele::Array<size_t> const &counters) {
    m_nitercount = counters[0];
    m_accept_count = counters[1];
    m_E_reject_count = counters[2];
    m_conf_reject_count = counters[3];
    m_neval = counters[4];
  }
};

}  // namespace mcpele

#endif  //_MCPELE_GMC_H
