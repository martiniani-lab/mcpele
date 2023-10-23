#include <iostream>
#include <stdexcept>
#include <numeric>
#include <cmath>
#include <vector>
#include <gtest/gtest.h>

#include "mcpele/cloud_test.h"
#include "mcpele/energy_window_test.h"

TEST(EnergyWindow, Works){
    double emin = 1.;
    double emax = 2.;
    pele::Array<double> xnew, xold;
    mcpele::MC * mc = NULL;
    double T = 0., eold = 0.;
    double eps = 1e-10;
    mcpele::EnergyWindowTest test(emin, emax);
    EXPECT_TRUE(test.test(xnew, emin+eps, xold, eold, T, mc));
    EXPECT_TRUE(test.test(xnew, emax-eps, xold, eold, T, mc));
    EXPECT_FALSE(test.test(xnew, emin-eps, xold, eold, T, mc));
    EXPECT_FALSE(test.test(xnew, emax+eps, xold, eold, T, mc));
}

void test_cloud_basics(const mcpele::Cloud& c, const size_t nr_drops, const pele::Array<double>& center, const double cloud_radius)
{
    EXPECT_EQ(c.size(), nr_drops);
    std::vector<double> x_sum(center.size(), 0);
    for (const std::shared_ptr<mcpele::Drop>& d : c) {
        EXPECT_TRUE(d->bias);
        EXPECT_TRUE(d->oracle);
        for (size_t i = 0; i < x_sum.size(); ++i) {
            x_sum[i] += d->x[i];
        }
        std::vector<double> tmp(center.begin(), center.end());
        for (size_t i = 0; i < tmp.size(); ++i) {
            tmp[i] -= d->x[i];
        }
        EXPECT_TRUE((std::inner_product(tmp.begin(), tmp.end(), tmp.begin(), double(0))) <= (cloud_radius * cloud_radius));
    }
    for (size_t i = 0; i < x_sum.size(); ++i) {
        EXPECT_NEAR(center[i], x_sum[i] / nr_drops, 4 / std::sqrt(nr_drops));
    }
}

TEST(CloudAccept, Works){
    const size_t nr_drops = 100;
    const size_t cloud_radius = 2;
    const size_t ndof = 33;
    pele::Array<double> trial_coords(ndof, 1.42);
    const double trial_energy = 42;
    pele::Array<double> old_coords(ndof, 10.22);
    const double old_energy = 41;
    const double temperature = 1;
    std::shared_ptr<mcpele::CloudTest> ct = std::make_shared<mcpele::CloudTest>(42, 44, nr_drops, cloud_radius);
    ct->test(trial_coords, trial_energy, old_coords, old_energy, temperature, NULL);
    const mcpele::Cloud& old_cloud = ct->display_old_cloud();
    const mcpele::Cloud& trial_cloud = ct->display_trial_cloud();
    // Note: mcpele::CloudTest::test calls swap on old and trial clouds if test will return true
    test_cloud_basics(old_cloud, nr_drops, trial_coords, cloud_radius);
    test_cloud_basics(trial_cloud, nr_drops, old_coords, cloud_radius);
}
