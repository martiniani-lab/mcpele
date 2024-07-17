#ifndef _MCPELE_GUIDED_MC_H
#define _MCPELE_GUIDED_MC_H

#include "gmc.h"
#include "mc.h"

namespace mcpele {

class GuidedMC final : public MCBase {
 public:
  typedef std::vector<std::shared_ptr<AcceptTest>> accept_t;
  typedef std::vector<std::shared_ptr<GMCConfTest>> conf_t;

 protected:
  accept_t m_accept_tests;
  conf_t m_conf_tests;
  conf_t m_late_conf_tests;
  size_t m_E_reject_count;
  size_t m_conf_reject_count;
  size_t m_displace_count;
  double m_timestep;
  double m_standard_deviation;
  bool m_forward;
  const bool m_normalize_conf_gradient;

  const size_t m_seed;
  std::mt19937_64 m_generator;
  std::normal_distribution<> m_normal_distribution;
  std::uniform_real_distribution<> m_uniform_distribution;
  std::uniform_int_distribution<> m_random_bool_distribution;

  const double m_max_timestep;
  const size_t m_adaptive_interval;
  const double m_adaptive_factor;
  const double m_adaptive_min_acceptance_ratio;
  const double m_adaptive_max_acceptance_ratio;
  size_t m_adaptive_total_steps;
  size_t m_adaptive_accepted_steps;

  pele::Array<double> get_conf_gradient(pele::Array<double> &coords);

 public:
  GuidedMC(std::shared_ptr<pele::BasePotential> potential,
           const pele::Array<double> &coords, double temperature,
           double timestep, double standard_deviation, size_t rseed,
           bool normalize_conf_gradient = true, double max_timestep = 0.0,
           size_t adaptive_interval = 100, double adaptive_factor = 0.9,
           double adaptive_min_acceptance_ratio = 0.2,
           double adaptive_max_acceptance_ratio = 0.5);
  ~GuidedMC() override = default;
  void run(size_t max_iter) override;
  void one_iteration() override;
  void check_input() override;
  std::shared_ptr<TakeStep> get_takestep() const override {
    throw std::runtime_error("GuidedMC::get_takestep: not implemented");
  }

  double get_timestep() const { return m_timestep; }
  void set_timestep(const double input) { m_timestep = input; }
  size_t get_count() const { return m_displace_count; }
  void set_count(const size_t input) { m_displace_count = input; }
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
    throw std::runtime_error("GuidedMC::get_changed_atoms: not implemented");
  }
  const std::vector<double> get_changed_coords_old() const override {
    throw std::runtime_error(
        "GuidedMC::get_changed_coords_old: not implemented");
  }
  pele::Array<size_t> get_adaption_counters() const {
    pele::Array<size_t> counters(2);
    counters[0] = m_adaptive_total_steps;
    counters[1] = m_adaptive_accepted_steps;
    return counters;
  }
  void set_adaption_counters(const pele::Array<size_t> &input) {
    m_adaptive_total_steps = input[0];
    m_adaptive_accepted_steps = input[1];
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

#endif  //_MCPELE_GUIDED_MC_H
