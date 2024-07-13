#include "mcpele/gmc_take_step.h"

namespace mcpele {

GMCTakeStep::GMCTakeStep(const double timestep, const size_t nparticles,
                         const size_t ndim, const size_t rseed,
                         const double max_timestep)
    : m_velocity(nparticles * ndim),
      m_timestep(timestep),
      m_max_timestep(max_timestep),
      m_changed_atoms(nparticles),
      m_changed_coords_old(nparticles * ndim),
      m_seed(rseed),
      m_generator(rseed),
      m_distribution(0.0, 1.0) {
  if (timestep <= 0.0) {
    throw std::runtime_error(
        "GMCTakeStep::GMCTakeStep: initial timestep must be positive, got " +
        std::to_string(timestep));
  }
  if (max_timestep < 0.0) {
    throw std::runtime_error(
        "GMCTakeStep::GMCTakeStep: max timestep must be positive, got " +
        std::to_string(max_timestep));
  }
  if (max_timestep != 0.0 && max_timestep < timestep) {
    throw std::runtime_error(
        "GMCTakeStep::GMCTakeStep: max timestep must be greater than or "
        "equal to initial timestep, got " +
        std::to_string(max_timestep));
  }
  std::iota(m_changed_atoms.begin(), m_changed_atoms.end(), 0);
  resample_velocity();
}

void GMCTakeStep::displace(pele::Array<double>& coords, MCBase *mc) {
  assert(coords.size() == m_changed_coords_old.size());
  if (m_changed_coords_old.size() != 0) {
    std::copy(coords.begin(), coords.end(), m_changed_coords_old.begin());
  }
  for (size_t i = 0; i < coords.size(); ++i) {
    coords[i] += m_timestep * m_velocity[i];
  }
}

void GMCTakeStep::resample_velocity() {
  for (size_t i = 0; i < m_velocity.size(); ++i) {
    m_velocity[i] = m_distribution(m_generator);
  }
  m_velocity /= norm(m_velocity);
}
}  // namespace mcpele