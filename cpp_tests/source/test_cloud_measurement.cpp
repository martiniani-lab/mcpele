#include <gtest/gtest.h>

#include "mcpele/adaptive_takestep.h"
#include "mcpele/check_spherical_container_config.h"
#include "mcpele/random_coords_displacement.h"
#include "mcpele/record_cloud_r2.h"
#include "mcpele/record_cloud_drops_timeseries.h"


#define EXPECT_NEAR_RELATIVE(A, B, T)  EXPECT_NEAR(fabs(A)/(fabs(A)+fabs(B)+1), fabs(B)/(fabs(A)+fabs(B)+1), T)

struct TrivialPotential : public pele::BasePotential {
    size_t call_count;
    virtual double get_energy(pele::Array<double> coords)
    {
        call_count++;
        return 0.;
    }
};

TEST(CloudMeasurementR2, Works)
{
    std::shared_ptr<TrivialPotential> potential = std::make_shared<TrivialPotential>();
    const size_t nr_drops = 10;
    const size_t cloud_radius = 1;
    const size_t disk_radius = 10;
    const size_t ndof = 2;
    pele::Array<double> old_coords(ndof, 0);
    const double temperature = 1;
    const size_t nr_iterations = 1000000;
    const size_t eq_steps = nr_iterations / 2;
    mcpele::MC mc(potential, old_coords, temperature);
    std::shared_ptr<mcpele::TakeStep> random_walk = std::make_shared<mcpele::RandomCoordsDisplacementAll>(46, 10);
    std::shared_ptr<mcpele::TakeStep> adaptive_random_walk = std::make_shared<mcpele::AdaptiveTakeStep>(random_walk);
    std::shared_ptr<mcpele::CloudTest> ct = std::make_shared<mcpele::CloudTest>(42, 44, nr_drops, cloud_radius);
    std::shared_ptr<mcpele::ConfTest> disk_test = std::make_shared<mcpele::CheckSphericalContainerConfig>(disk_radius);
    std::shared_ptr<mcpele::RecordCloudR2> cloud_measure_r2 = std::make_shared<mcpele::RecordCloudR2>(eq_steps, old_coords);
    ct->add_conf_test(disk_test);
    mc.set_takestep(adaptive_random_walk);
    mc.add_accept_test(ct);
    mc.add_action(cloud_measure_r2);
    mc.set_report_steps(0);
    mc.run(nr_iterations);
    const double true_mean_r2 = 0.5 * disk_radius * disk_radius;
    const double true_var_r2 = true_mean_r2 * true_mean_r2 / 3;
    EXPECT_NEAR_RELATIVE(true_mean_r2, cloud_measure_r2->get_mean_r2(), 1e-1);
    EXPECT_NEAR_RELATIVE(true_var_r2, cloud_measure_r2->get_var_r2(), 1e-1);
    std::cout << "cloud_measure_r2->get_var_r2(): " << cloud_measure_r2->get_var_r2() << "\n";
    std::cout << "true_var: " << true_var_r2 << "\n";
    std::cout << "accepted fraction: " << mc.get_accepted_fraction() << "\n";
}

TEST(RecordCloudDropsTimeseries , Works)
{
    std::shared_ptr<TrivialPotential> potential = std::make_shared<TrivialPotential>();
    const size_t nr_drops = 10;
    const size_t cloud_radius = 1;
    const size_t disk_radius = 10;
    const size_t ndof = 2;
    pele::Array<double> old_coords(ndof, 0);
    const double temperature = 1;
    const size_t nr_iterations = 1000000;
    const size_t eq_steps = nr_iterations / 2;
    const size_t record_every = 1;
    mcpele::MC mc(potential, old_coords, temperature);
    std::shared_ptr<mcpele::TakeStep> random_walk = std::make_shared<mcpele::RandomCoordsDisplacementAll>(46, 10);
    std::shared_ptr<mcpele::TakeStep> adaptive_random_walk = std::make_shared<mcpele::AdaptiveTakeStep>(random_walk);
    std::shared_ptr<mcpele::CloudTest> ct = std::make_shared<mcpele::CloudTest>(42, 44, nr_drops, cloud_radius);
    std::shared_ptr<mcpele::ConfTest> disk_test = std::make_shared<mcpele::CheckSphericalContainerConfig>(disk_radius);
    std::shared_ptr<mcpele::RecordCloudDropsTimeseries> cloud_drops_ts =
    std::make_shared<mcpele::RecordCloudDropsTimeseries>(old_coords, record_every, eq_steps);
    ct->add_conf_test(disk_test);
    mc.set_takestep(adaptive_random_walk);
    mc.add_accept_test(ct);
    mc.add_action(cloud_drops_ts);
    mc.set_report_steps(0);
    mc.run(nr_iterations);
    const double true_mean_r2 = 0.5 * disk_radius * disk_radius;
    const double true_var_r2 = true_mean_r2 * true_mean_r2 / 3;
    double mean_r2 = 0.;
    std::deque<pele::Array<double>> ts = cloud_drops_ts->get_time_series();
    std::cout<<"deque size: "<<ts.size()<<"\n";
    for(auto& drops: ts){
        double num = 0;
        double den = 0;
        for(auto& x: drops){
            num += x*x;
            if (x > 0){den += 1.;};
        }
        mean_r2 += num/den;
    }
    mean_r2 /= ts.size();
    EXPECT_NEAR_RELATIVE(true_mean_r2, mean_r2, 1e-2);
    std::cout << "true_mean: " << true_mean_r2 << " mean_r2: "<< mean_r2 <<"\n";
    std::cout << "accepted fraction: " << mc.get_accepted_fraction() << "\n";
}
