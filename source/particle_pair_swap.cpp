#include <algorithm>

#include "mcpele/particle_pair_swap.h"

namespace mcpele {

ParticlePairSwap::ParticlePairSwap(const size_t seed, const size_t nr_particles, const size_t ndim)
    : m_seed(seed),
      m_generator(seed),
      m_distribution(0, nr_particles - 1),
      m_nr_particles(nr_particles),
      m_ndim(ndim),
      m_changed_coords_old(2 * ndim)
{
    if (nr_particles == 0) {
        throw std::runtime_error("ParticlePairSwap: illegal input");
    }
}

void ParticlePairSwap::displace(pele::Array<double>& coords, MC* mc)
{
    size_t particle_a = 42;
    size_t particle_b = 42;
    while (particle_a == particle_b) {
        particle_a = m_distribution(m_generator);
        particle_b = m_distribution(m_generator);
    }
    assert(particle_a < m_nr_particles && particle_b < m_nr_particles);
    assert(particle_a != particle_b);
    m_changed_atoms[0] = particle_a;
    m_changed_atoms[1] = particle_b;
    swap_coordinates(particle_a, particle_b, coords);
}

void ParticlePairSwap::swap_coordinates(const size_t particle_a, const size_t particle_b, pele::Array<double>& coords)
{
    const size_t index_a = particle_a * m_ndim;
    const size_t index_b = particle_b * m_ndim;
    double*const& x = coords.data();
    double*const& xa = x + index_a;
    double*const& xb = x + index_b;
    std::copy(xa, xa + m_ndim, m_changed_coords_old.begin());
    std::copy(xb, xb + m_ndim, m_changed_coords_old.begin() + m_ndim);
    if (particle_a != particle_b) {
        std::swap_ranges(xa, xa + m_ndim, xb);
    }
}

void ParticlePairSwap::set_generator_seed(const size_t inp)
{
    m_generator.seed(inp);
    m_seed = inp;
}

} // namespace mcpele
