#ifndef _MCPELE_CLOUD_TEST_H__
#define _MCPELE_CLOUD_TEST_H__

#include "mc.h"
#include "uniform_spherical_sampling.h"

namespace mcpele {
    
struct Drop {
    double bias;
    bool oracle;
    pele::Array<double> x;
};

class Cloud : public std::vector<std::shared_ptr<Drop> > {
    double m_weight;
public:
    Cloud(){vector();} //inialize empty vector
    Cloud(const size_t nr_drops)
    {
        for (size_t i = 0; i < nr_drops; ++i) {
            push_back(std::make_shared<Drop>());
        }
    }

    void compute_weight()
    {
        m_weight = 0;
        for (size_t i = 0; i < size(); ++i) {
            m_weight += at(i)->bias * at(i)->oracle;
        }
    }

    double get_weight() const
    {
        return m_weight;
    }

    void set_bias(pele::Array<double> new_bias){
        assert (new_bias.size() == size());
        for (size_t i = 0; i < size(); ++i) {
            at(i)->bias = new_bias[i];
        }
        compute_weight();
    }
};

class TestHandler : public std::vector<std::shared_ptr<ConfTest> > {
public:
    bool do_conf_tests(pele::Array<double> coords, MC* mc)
    {
        for (std::shared_ptr<ConfTest> t : *this) {
            if (!t->conf_test(coords, mc)) {
                return false;
            }
        }
        return true;
    }
};

class CloudTest : public AcceptTest {
    const size_t m_nr_cloud_points;
    TestHandler m_tests;
    bool m_initialised;
    Cloud m_old_cloud, m_trial_cloud;
    std::shared_ptr<pele::BasePotential> m_bias;
    std::mt19937_64 m_generator;
    UniformSphericalSampling m_n_ball_volume_sampling;
    std::uniform_real_distribution<double> m_uniform_real_distribution;
public:
    CloudTest(const size_t rseed, const size_t cloud_seed,
        const size_t nr_cloud_points, const double cloud_radius,
        std::shared_ptr<pele::BasePotential> bias=NULL)
        : m_nr_cloud_points(nr_cloud_points),
          m_initialised(false),
          m_old_cloud(nr_cloud_points),
          m_trial_cloud(nr_cloud_points),
          m_bias(bias),
          m_generator(rseed),
          m_n_ball_volume_sampling(cloud_seed, cloud_radius),
          m_uniform_real_distribution(0.0, 1.0)
    {}
    bool test(pele::Array<double>& trial_coords, double trial_energy,
        pele::Array<double>& old_coords, double old_energy,
        double temperature, MC* mc)
    {
        if (!m_initialised) {
            sample_cloud(m_old_cloud, old_coords, temperature, mc);
            m_initialised = true;
        }
        sample_cloud(m_trial_cloud, trial_coords, temperature, mc);
        const bool success = get_trial_success();
        if (success) {
            std::swap(m_old_cloud, m_trial_cloud);
            m_old_cloud.compute_weight();
        }
        return success;
    }
    void sample_cloud(Cloud& cloud, const pele::Array<double> center,
        const double temperature, MC* mc)
    {
        for (std::shared_ptr<Drop> drop : cloud) {
            pele::Array<double> coords(center.size());
            m_n_ball_volume_sampling.displace(coords, mc);
            coords += center;
            if (m_bias) {
                drop->bias = std::exp(-m_bias->get_energy(coords) / temperature);
            }
            else {
                drop->bias = 1;
            }
            drop->oracle = m_tests.do_conf_tests(coords, mc);
            drop->x = pele::Array<double>(coords.begin(), coords.end()).copy();
        }
        cloud.compute_weight();
    }
    bool get_trial_success()
    {
        const double r = m_trial_cloud.get_weight() / m_old_cloud.get_weight();
        if (r > 1) {
            return true;
        }
        return m_uniform_real_distribution(m_generator) < r;
    }
    void set_generator_seeds(const size_t rseed, const size_t cloud_seed)
    {
        m_generator.seed(rseed);
        m_n_ball_volume_sampling.set_generator_seed(cloud_seed);
    }
    void add_conf_test(std::shared_ptr<ConfTest> inp)
    {
        m_tests.push_back(inp);
    }
    const Cloud& display_old_cloud() const
    {
        return m_old_cloud;
    }
    const Cloud& display_trial_cloud() const
    {
        return m_trial_cloud;
    }
    const Cloud get_old_cloud() const
    {
        return m_old_cloud;
    }

    void set_cloud(Cloud cloud)
    {
        assert(cloud.size() == m_old_cloud.size());
        m_old_cloud = cloud;
        m_old_cloud.compute_weight();
    }

    void set_cloud_bias(pele::Array<double> new_bias){
        m_old_cloud.set_bias(new_bias);
    }

};    
    
} // namespace mcpele

#endif // #ifndef _MCPELE_CLOUD_TEST_H__

