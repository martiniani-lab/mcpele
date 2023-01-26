#ifndef _MCPELE_PAIR_DIST_HISTOGRAM_H
#define _MCPELE_PAIR_DIST_HISTOGRAM_H

#include <algorithm>
#include <cassert>
#include <cmath>
#include <stdexcept>
#include <utility>

#include "mcpele/histogram.h"
#include "pele/distance.hpp"

namespace mcpele {

template <size_t BOXDIM>
class PairDistHistogram {
 private:
  pele::periodic_distance<BOXDIM> m_distance;
  const size_t m_nr_bins;
  const double m_min_dist;
  const double m_max_dist;
  const double m_delta_bin;
  mcpele::Histogram m_hist;
  size_t m_nr_configs;

 public:
  PairDistHistogram(pele::Array<double> boxvector, const size_t nr_bins)
      : m_distance(boxvector),
        m_nr_bins(nr_bins),
        m_min_dist(0),
        m_max_dist(0.5 * *std::min_element(boxvector.data(),
                                           boxvector.data() + BOXDIM)),
        m_delta_bin((m_max_dist - m_min_dist) / static_cast<double>(m_nr_bins)),
        m_hist(m_min_dist, m_max_dist, m_delta_bin),
        m_nr_configs(0) {
    if (BOXDIM != boxvector.size()) {
      throw std::runtime_error("PairDistHistogram: illegal boxvector size");
    }
  }
  virtual ~PairDistHistogram() {}
  void add_configuration(pele::Array<double> coords) {
    ++m_nr_configs;
    const size_t nr_particles(coords.size() / BOXDIM);
    for (size_t i = 0; i < nr_particles; ++i) {
      for (size_t j = i + 1; j < nr_particles; ++j) {
        add_distance(i, j, coords.data());
      }
    }
  }
  void add_distance(const size_t i, const size_t j, const double *coor) {
    double rij[BOXDIM];
    m_distance.get_rij(rij, coor + i * BOXDIM, coor + j * BOXDIM);
    const double r =
        sqrt(std::inner_product(rij, rij + BOXDIM, rij, double(0)));
    if (r > m_max_dist) {
      // here, g(r) measurement is resticted to a disc domain of radius
      // m_max_dist in distance space; could be done differently
      return;
    }
    m_hist.add_entry(r);
  }
  double volume_nball(const double radius, const size_t ndim) const {
    return pow(M_PI, 0.5 * ndim) * pow(radius, ndim) / tgamma(0.5 * ndim + 1);
  }
  std::vector<double> get_vecdata_r() const {
    std::vector<double> result(m_nr_bins);
    for (size_t i = 0; i < m_nr_bins; ++i) {
      const double r = m_hist.get_position(i);
      result.at(i) = r;
    }
    return result;
  }
  std::vector<double> get_vecdata_gr(const double number_density,
                                     const size_t nr_particles) const {
    std::vector<double> result(m_nr_bins);
    for (size_t i = 0; i < m_nr_bins; ++i) {
      const double r = m_hist.get_position(i);
      const double delta_r = m_hist.bin();
      const double shell_volume_r = volume_nball(r + 0.5 * delta_r, BOXDIM) -
                                    volume_nball(r - 0.5 * delta_r, BOXDIM);
      const double nid = shell_volume_r * number_density;
      const double normalization =
          2.0 / (static_cast<double>(m_nr_configs) *
                 static_cast<double>(nr_particles) * nid);
      const double g_of_r =
          normalization * static_cast<double>(m_hist.get_entry(i));
      result.at(i) = g_of_r;
    }
    return result;
  }
};

}  // namespace mcpele

#endif  //#ifndef _MCPELE_PAIR_DIST_HISTOGRAM_H
