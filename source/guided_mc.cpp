#include "mcpele/guided_mc.h"

#include "mcpele/progress.h"
#include "pele/eigen_interface.hpp"

namespace mcpele {

GuidedMC::GuidedMC(std::shared_ptr<pele::BasePotential> potential,
                   const pele::Array<double> &coords, const double temperature,
                   const double timestep, const double standard_deviation,
                   const size_t rseed, const bool normalize_conf_gradient,
                   const bool use_hessian, const double max_timestep,
                   const size_t adaptive_interval, const double adaptive_factor,
                   const double adaptive_min_acceptance_ratio,
                   const double adaptive_max_acceptance_ratio)
    : MCBase(std::move(potential), coords, temperature),
      m_E_reject_count(0),
      m_conf_reject_count(0),
      m_displace_count(0),
      m_timestep(timestep),
      m_standard_deviation(standard_deviation),
      m_normalize_conf_gradient(normalize_conf_gradient),
      m_use_hessian(use_hessian),
      m_seed(rseed),
      m_generator(rseed),
      m_normal_distribution(0.0, 1.0),
      m_uniform_distribution(0.0, 1.0),
      m_random_bool_distribution(0, 1),
      m_max_timestep(max_timestep),
      m_adaptive_interval(adaptive_interval),
      m_adaptive_factor(adaptive_factor),
      m_adaptive_min_acceptance_ratio(adaptive_min_acceptance_ratio),
      m_adaptive_max_acceptance_ratio(adaptive_max_acceptance_ratio),
      m_standard_deviation_timestep_ratio(standard_deviation / timestep) {
  if (timestep <= 0.0) {
    throw std::runtime_error(
        "GuidedMC::GuidedMC: initial timestep must be positive, got " +
        std::to_string(timestep));
  }
  if (standard_deviation <= 0.0) {
    throw std::runtime_error(
        "GuidedMC::GuidedMC: standard deviation must be positive, got " +
        std::to_string(standard_deviation));
  }
  if (max_timestep != 0.0 and max_timestep < timestep) {
    throw std::runtime_error(
        "GuidedMC::GuidedMC: max timestep must be greater than or equal to "
        "initial timestep, got " +
        std::to_string(max_timestep));
  }
  if (adaptive_factor <= 0.0 or adaptive_factor >= 1.0) {
    throw std::runtime_error(
        "GuidedMC::GuidedMC: adaptive factor must be between 0 and 1, got " +
        std::to_string(adaptive_factor));
  }
  if (adaptive_min_acceptance_ratio <= 0.0 or
      adaptive_min_acceptance_ratio >= 1.0) {
    throw std::runtime_error(
        "GuidedMC::GuidedMC: adaptive min acceptance ratio must be between 0 "
        "and 1, got " +
        std::to_string(adaptive_min_acceptance_ratio));
  }
  if (adaptive_max_acceptance_ratio <= 0.0 or
      adaptive_max_acceptance_ratio >= 1.0) {
    throw std::runtime_error(
        "GuidedMC::GuidedMC: adaptive max acceptance ratio must be between 0 "
        "and 1, got " +
        std::to_string(adaptive_max_acceptance_ratio));
  }
  m_use_energy_change = false;
  m_forward = m_random_bool_distribution(m_generator);
}

void GuidedMC::run(const size_t max_iter) {
  check_input();
  progress stat(max_iter);
  while (m_niter < max_iter) {
    this->one_iteration();
    if (m_print_progress) {
      stat.next(m_niter);
    }
  }
  m_niter = 0;
  m_forward = m_random_bool_distribution(m_generator);
}

pele::Array<double> GuidedMC::get_conf_gradient(pele::Array<double> &coords) {
  pele::Array conf_gradient(coords.size(), 0.0);
  for (const auto &test : m_conf_tests) {
    conf_gradient += test->gmc_gradient(coords, this);
  }
  for (const auto &test : m_late_conf_tests) {
    conf_gradient += test->gmc_gradient(coords, this);
  }

  if (m_use_hessian) {
    pele::Array conf_hessian(coords.size(), 0.0);
    for (const auto &test : m_conf_tests) {
      conf_hessian += test->gmc_hessian(coords, this);
    }
    for (const auto &test : m_late_conf_tests) {
      conf_hessian += test->gmc_hessian(coords, this);
    }
    // Code following pele/optimizer.hpp.
    auto eigen_gradient = Eigen::VectorXd(conf_gradient.size());
    eig_eq_pele(eigen_gradient, conf_gradient);
    auto eigen_hessian =
        Eigen::MatrixXd(conf_hessian.size(), conf_hessian.size());
    eig_mat_eq_pele(eigen_hessian, conf_hessian);

    const Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> es(eigen_hessian);
    const auto eigenvalues = es.eigenvalues();
    const auto eigenvectors = es.eigenvectors();
    double abs_min_eigenvalue = eigenvalues.minCoeff();
    if (abs_min_eigenvalue < 0.0) {
      abs_min_eigenvalue = -abs_min_eigenvalue;
    } else {
      abs_min_eigenvalue = 0.0;
    }
    const double average_eigenvalue = eigenvalues.mean();
    const double offset = std::max(1.0e-1 * std::abs(average_eigenvalue),
                             2.0 * abs_min_eigenvalue);
    eigen_hessian.diagonal().array() += offset;
    Eigen::VectorXd newton_step = eigen_hessian.ldlt().solve(eigen_gradient);
    pele_eq_eig(conf_gradient, newton_step);
  }
  if (m_normalize_conf_gradient) {
    if (const auto n = norm(conf_gradient); n != 0.0) {
      conf_gradient /= n;
    }
  }
  if (not m_forward) {
    conf_gradient *= -1.0;
  }
  return conf_gradient;
}

void GuidedMC::one_iteration() {
  ++m_niter;
  ++m_nitercount;

  const auto conf_gradient = get_conf_gradient(m_coords);

  auto mean = m_coords + m_timestep * conf_gradient;
  pele::Array<double> normal_vector(m_coords.size());
  for (size_t i = 0; i < m_coords.size(); ++i) {
    normal_vector[i] =
        m_standard_deviation * m_normal_distribution(m_generator);
  }
  m_trial_coords = mean + normal_vector;
  ++m_displace_count;

  // configuration tests
  m_success = std::all_of(m_conf_tests.begin(), m_conf_tests.end(),
                          [this](const auto &test) {
                            return test->conf_test(m_trial_coords, this);
                          });
  if (not m_success) {
    ++m_conf_reject_count;
  } else {
    // Metropolis-Hastings accept test
    ++m_neval;
    m_trial_energy = compute_energy(m_trial_coords);
    const double energy_exp_argument =
        (m_energy - m_trial_energy) / m_temperature;
    const auto trial_conf_gradient = get_conf_gradient(m_trial_coords);
    const auto inverse_mean = m_trial_coords - m_timestep * trial_conf_gradient;
    const auto forward_diff = (m_trial_coords - mean);
    const auto backward_diff = (m_coords - inverse_mean);
    const double proposal_exp_argument =
        (dot(forward_diff, forward_diff) - dot(backward_diff, backward_diff)) /
        (2.0 * m_standard_deviation * m_standard_deviation);
    if (const auto exp_argument = energy_exp_argument + proposal_exp_argument;
        exp_argument < 0.0) {
      if (m_uniform_distribution(m_generator) > exp(exp_argument)) {
        m_success = false;
      }
    }
    if (not m_success) {
      ++m_E_reject_count;
    } else {
      // late configuration tests
      m_success =
          std::all_of(m_late_conf_tests.begin(), m_late_conf_tests.end(),
                      [this](const auto &test) {
                        return test->conf_test(m_trial_coords, this);
                      });
      if (not m_success) {
        ++m_conf_reject_count;
      }
    }
  }

  // adapt stepsize etc.
  if (get_iterations_count() <= m_report_steps) {
    ++m_adaptive_total_steps;
    if (m_success) {
      ++m_adaptive_accepted_steps;
    }
    if (get_iterations_count() % m_adaptive_interval == 0) {
      const double acceptance_fraction =
          static_cast<double>(m_adaptive_accepted_steps) /
          static_cast<double>(m_adaptive_total_steps);
      m_adaptive_total_steps = 0;
      m_adaptive_accepted_steps = 0;
      if (acceptance_fraction < m_adaptive_min_acceptance_ratio) {
        // increase acceptance by decreasing timestep and standard deviation
        m_timestep *= m_adaptive_factor;
        m_standard_deviation *= m_adaptive_factor;
      } else if (acceptance_fraction > m_adaptive_max_acceptance_ratio) {
        // decrease acceptance by increasing timestep and standard deviation
        m_timestep /= m_adaptive_factor;
        m_standard_deviation /= m_adaptive_factor;
        if (m_max_timestep != 0.0 and m_timestep > m_max_timestep) {
          m_timestep = m_max_timestep;
          m_standard_deviation =
              m_standard_deviation_timestep_ratio * m_timestep;
        }
      }
      if (get_iterations_count() + m_adaptive_interval > m_report_steps) {
        std::cout << "GuidedMC: final adaptive acceptance fraction, timestep, "
                     "and standard deviation: "
                  << acceptance_fraction << " " << m_timestep << " "
                  << m_standard_deviation << std::endl;
      }
    }
  }

  // log success to step being taken
  if (m_record_acceptance_rate) {
    m_success_accumulator.add_success(m_success);
  }

  // if the step is accepted, copy the coordinates and energy
  if (m_success) {
    m_coords.assign(m_trial_coords);
    m_energy = m_trial_energy;
    ++m_accept_count;
  } else {
    // otherwise flip direction
    m_forward = not m_forward;
  }

  if (m_coords.get_max() > 1e9) {
    std::cout << "WARNING: this can lead to errors in the energy calculation "
                 "due to overflow";
  }

  // perform the actions on the new configuration
  do_actions(m_coords, m_energy, m_success);

  m_last_success = m_success;
}

void GuidedMC::check_input() {
  if (m_use_energy_change) {
    throw std::runtime_error(
        "GMC::check_input: using energy change is not supported");
  }
  if (m_conf_tests.size() + m_late_conf_tests.size() == 0) {
    throw std::runtime_error(
        "GMC::check_input: there must be at least one "
        "(late) conf test that determines the gradient in "
        "a reflection of Galilean Monte Carlo");
  }
  if (m_enable_input_warnings) {
    if (m_conf_tests.size() == 0 && m_late_conf_tests.size() == 0) {
      std::cout << "warning: no conf tests set"
                << "\n";
    }
    if (m_actions.size() == 0) {
      std::cout << "warning: no actions set"
                << "\n";
    }
  }
}

}  // namespace mcpele