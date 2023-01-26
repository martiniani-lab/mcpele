
#include <cmath>
#include <gtest/gtest.h>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "mcpele/moving_average.h"
#include "mcpele/record_displacement_per_particle_timeseries.h"
#include "mcpele/record_energy_timeseries.h"
#include "mcpele/record_lowest_evalue_timeseries.h"
#include "pele/array.hpp"
#include "pele/harmonic.hpp"
#include "pele/hs_wca.hpp"
#include "pele/lj.hpp"
#define EXPECT_NEAR_RELATIVE(A, B, T)            \
  EXPECT_NEAR(fabs(A) / (fabs(A) + fabs(B) + 1), \
              fabs(B) / (fabs(A) + fabs(B) + 1), T)

struct TrivialTakestep : public mcpele::TakeStep {
  size_t call_count;
  TrivialTakestep() : call_count(0) {}
  virtual void displace(pele::Array<double> &coords, mcpele::MC *mc = NULL) {
    call_count++;
  }
};

TEST(EnergyTimeseries, Basic) {
  const size_t boxdim = 3;
  const size_t nparticles = 100;
  const size_t ndof = nparticles * boxdim;
  const size_t niter = 10000;
  const size_t record_every = 100;
  const double k = 400;
  pele::Array<double> coords(ndof, 2);
  pele::Array<double> origin(ndof, 0);
  std::shared_ptr<pele::Harmonic> potential =
      std::make_shared<pele::Harmonic>(origin, k, boxdim);
  const auto enumerical = potential->get_energy(coords);
  double etrue(0);
  for (size_t i = 0; i < ndof; ++i) {
    const auto delta = coords[i] - origin[i];
    etrue += 0.5 * k * delta * delta;
  }
  EXPECT_DOUBLE_EQ(enumerical, etrue);
  std::shared_ptr<mcpele::MC> mc =
      std::make_shared<mcpele::MC>(potential, coords, 1);
  mc->set_use_energy_change(false);
  mcpele::RecordEnergyTimeseries *ts =
      new mcpele::RecordEnergyTimeseries(niter, record_every);
  mc->add_action(std::shared_ptr<mcpele::RecordEnergyTimeseries>(ts));
  mc->set_takestep(std::make_shared<TrivialTakestep>());
  mc->run(niter);
  EXPECT_EQ(mc->get_iterations_count(), niter);
  pele::Array<double> series = ts->get_time_series();
  EXPECT_EQ(series.size(), niter / record_every);
  for (size_t i = 0; i < series.size(); ++i) {
    EXPECT_DOUBLE_EQ(series[i], enumerical);
  }
}

TEST(EVTimeseries, Works) {
  typedef mcpele::RecordLowestEValueTimeseries series_t;
  // typedef mcpele::RecordEnergyTimeseries series_t;
  const size_t boxdim = 3;
  const size_t nparticles = 4;
  const size_t ndof = nparticles * boxdim;
  const size_t niter = 100;
  const size_t record_every = 10;
  const double k = 400;
  pele::Array<double> coords(ndof, 2);
  for (size_t i = 0; i < coords.size(); ++i) {
    coords[i] += 0.1 * i;
  }
  pele::Array<double> origin(ndof, 1);
  for (size_t i = 0; i < origin.size(); ++i) {
    origin[i] -= 0.01 * i;
  }
  std::shared_ptr<pele::Harmonic> potential =
      std::make_shared<pele::Harmonic>(origin, k, boxdim);

  const double eps = 1;
  const double sca = 1;
  pele::Array<double> radii(nparticles, 1);
  std::shared_ptr<pele::HS_WCA<3>> landscape_potential =
      std::make_shared<pele::HS_WCA<3>>(eps, sca, radii);

  // pele::Harmonic* landscape_potential = new pele::Harmonic(origin, k,
  // boxdim); pele::LJ* landscape_potential = new pele::LJ(1, 1);

  const auto enumerical = potential->get_energy(coords);
  double etrue(0);
  for (size_t i = 0; i < ndof; ++i) {
    const auto delta = coords[i] - origin[i];
    etrue += 0.5 * k * delta * delta;
  }
  EXPECT_DOUBLE_EQ(enumerical, etrue);
  std::shared_ptr<mcpele::MC> mc =
      std::make_shared<mcpele::MC>(potential, coords, 1);
  mc->set_use_energy_change(false);

  const size_t lbfgsniter = 30;
  pele::Array<double> ranvec = origin.copy();
  // series_t* ts = new series_t(niter, record_every);
  std::shared_ptr<series_t> ts = std::make_shared<series_t>(
      niter, record_every, landscape_potential, boxdim, ranvec, lbfgsniter);
  // series_t* ts = new series_t(niter, record_every, potential, boxdim,
  // ranvec);

  mc->add_action(std::shared_ptr<series_t>(ts));
  mc->set_takestep(std::make_shared<TrivialTakestep>());
  // mc->set_print_progress();
  mc->run(niter);
  EXPECT_EQ(mc->get_iterations_count(), niter);
  pele::Array<double> series = ts->get_time_series();
  EXPECT_EQ(series.size(), niter / record_every);
  const double eigenvalue_reference = series[0];
  for (size_t i = 0; i < series.size(); ++i) {
    EXPECT_NEAR_RELATIVE(series[i], eigenvalue_reference, 1e-9);
  }
}

TEST(ParticleDisplacementTimeseries, Works) {
  const size_t boxdim = 3;
  const size_t nparticles = 100;
  const size_t ndof = nparticles * boxdim;
  const size_t niter = 10000;
  const size_t record_every = 100;
  pele::Array<double> coords(ndof, 2);
  pele::Array<double> origin(ndof, 0);

  const double k = 400;
  std::shared_ptr<pele::Harmonic> potential =
      std::make_shared<pele::Harmonic>(origin, k, boxdim);
  std::shared_ptr<mcpele::MC> mc =
      std::make_shared<mcpele::MC>(potential, coords, 1);
  mc->set_use_energy_change(false);

  mcpele::RecordDisplacementPerParticleTimeseries *ts =
      new mcpele::RecordDisplacementPerParticleTimeseries(niter, record_every,
                                                          origin, boxdim);
  mc->add_action(
      std::shared_ptr<mcpele::RecordDisplacementPerParticleTimeseries>(ts));
  mc->set_takestep(std::make_shared<TrivialTakestep>());

  mc->run(niter);
  EXPECT_EQ(mc->get_iterations_count(), niter);

  pele::Array<double> series = ts->get_time_series();
  EXPECT_EQ(series.size(), niter / record_every);
  for (size_t i = 0; i < series.size(); ++i) {
    EXPECT_NEAR_RELATIVE(series[i], sqrt(12), 1e-14);
  }
}

TEST(MovingAverage, Works) {
  std::vector<double> ts = {0, 0, 0, 1, 1, 1, 1, 4, 4, 5};
  const double true_mean =
      std::accumulate(ts.begin(), ts.end(), double(0)) / ts.size();
  std::cout << "true_mean: " << true_mean << std::endl;
  mcpele::MovingAverageAcc ma10(ts, ts.size(), ts.size());
  ma10.reset();
  EXPECT_DOUBLE_EQ(true_mean, ma10.get_mean());
  mcpele::MovingAverageAcc ma2(ts, ts.size(), 2);
  EXPECT_DOUBLE_EQ(0, ma2.get_mean());
  ma2.shift_right();
  EXPECT_DOUBLE_EQ(0, ma2.get_mean());
  ma2.shift_right();
  EXPECT_DOUBLE_EQ(0.5, ma2.get_mean());
}

TEST(RecordEnergyTimeseries, Throws) {
  bool threw = false;
  try {
    mcpele::RecordEnergyTimeseries(42, 0);
  } catch (...) {
    threw = true;
  }
  EXPECT_TRUE(threw);
}
