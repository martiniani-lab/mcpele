#ifndef _MCPELE_GMC_TAKE_STEP_H
#define _MCPELE_GMC_TAKE_STEP_H

#include <random>

#include "mc.h"

namespace mcpele {
class GMCTakeStep final : public TakeStep {
 private:
  pele::Array<double> m_velocity;
  double m_timestep;
  double m_max_timestep;
  std::vector<size_t> m_changed_atoms;
  std::vector<double> m_changed_coords_old;
  size_t m_seed;
  std::mt19937_64 m_generator;
  std::normal_distribution<> m_distribution;

 public:
  GMCTakeStep(double timestep, size_t nparticles, size_t ndim, size_t rseed,
              double max_timestep = 0.0);
  ~GMCTakeStep() override = default;
  void displace(pele::Array<double> &coords, MCBase *mc) override;
  void increase_acceptance(const double factor) override {
    assert(factor < 1.0 && factor > 0.0);
    m_timestep *= factor;
  }
  void decrease_acceptance(const double factor) override {
    assert(factor < 1.0 && factor > 0.0);
    if (m_max_timestep == 0.0 || m_timestep < m_max_timestep) {
      m_timestep /= factor;
      if (m_timestep > m_max_timestep) {
        m_timestep = m_max_timestep;
      }
    }
  }
  [[nodiscard]] const std::vector<size_t> get_changed_atoms() const override {
    return m_changed_atoms;
  }
  [[nodiscard]] const std::vector<double> get_changed_coords_old()
      const override {
    return m_changed_coords_old;
  }
  void resample_velocity();
  pele::Array<double> get_velocity() const { return m_velocity.copy(); }
  void set_velocity(const pele::Array<double> &vel) { m_velocity = vel.copy(); }
};

}  // namespace mcpele

#endif  //_MCPELE_GMC_TAKE_STEP_H
