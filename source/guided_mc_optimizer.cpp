#include "mcpele/guided_mc_optimizer.h"

#include "mcpele/progress.h"
#include "pele/combine_potentials.hpp"

namespace mcpele {

GuidedMCOptimizer::GuidedMCOptimizer(
    std::shared_ptr<pele::BasePotential> potential,
    const pele::Array<double> &coords, const double temperature,
    std::shared_ptr<pele::GradientOptimizer> optimizer,
    const int optimizer_niter, const double standard_deviation,
    const size_t rseed, const double max_standard_deviation,
    const size_t adaptive_interval, const double adaptive_factor,
    const double adaptive_min_acceptance_ratio,
    const double adaptive_max_acceptance_ratio)
    : MCBase(std::move(potential), coords, temperature),
      m_E_reject_count(0),
      m_conf_reject_count(0),
      m_displace_count(0),
      m_optimizer(std::move(optimizer)),
      m_optimizer_niter(optimizer_niter),
      m_standard_deviation(standard_deviation),
      m_seed(rseed),
      m_generator(rseed),
      m_normal_distribution(0.0, 1.0),
      m_uniform_distribution(0.0, 1.0),
      m_random_bool_distribution(0, 1),
      m_max_standard_deviation(max_standard_deviation),
      m_adaptive_interval(adaptive_interval),
      m_adaptive_factor(adaptive_factor),
      m_adaptive_min_acceptance_ratio(adaptive_min_acceptance_ratio),
      m_adaptive_max_acceptance_ratio(adaptive_max_acceptance_ratio) {
  if (optimizer_niter <= 0) {
    throw std::runtime_error(
        "GuidedMCOptimizer::GuidedMCOptimizer: optimizer_niter must be "
        "positive, got " +
        std::to_string(optimizer_niter));
  }
  if (standard_deviation <= 0.0) {
    throw std::runtime_error(
        "GuidedMC::GuidedMC: standard deviation must be positive, got " +
        std::to_string(standard_deviation));
  }
  if (max_standard_deviation != 0.0 and
      max_standard_deviation < standard_deviation) {
    throw std::runtime_error(
        "GuidedMCOptimizer::GuidedMCOptimizer: max standard deviation must be "
        "greater than or equal to initial standard deviation, got " +
        std::to_string(max_standard_deviation));
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
  m_descent_potential = m_optimizer->get_potential();
  m_ascent_potential =
      std::make_shared<pele::NegatedPotential>(m_descent_potential);
}

void GuidedMCOptimizer::run(const size_t max_iter) {
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

void GuidedMCOptimizer::one_iteration() {
  ++m_niter;
  ++m_nitercount;

  m_optimizer->set_potential(m_forward ? m_descent_potential
                                       : m_ascent_potential);
  m_optimizer->reset(m_coords);
  m_optimizer->run(m_optimizer_niter);
  m_neval += m_optimizer->get_nfev();
  auto mean = m_optimizer->get_x();

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
    m_optimizer->set_potential(m_forward ? m_ascent_potential
                                         : m_descent_potential);
    m_optimizer->reset(m_trial_coords);
    m_optimizer->run(m_optimizer_niter);
    m_neval += m_optimizer->get_nfev();
    const auto inverse_mean = m_optimizer->get_x();
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
        // increase acceptance by increasing standard deviation
        m_standard_deviation /= m_adaptive_factor;
        if (m_max_standard_deviation != 0.0 and
            m_standard_deviation > m_max_standard_deviation) {
          m_standard_deviation = m_max_standard_deviation;
        }
      } else if (acceptance_fraction > m_adaptive_max_acceptance_ratio) {
        // decrease acceptance by decreasing standard deviation
        m_standard_deviation *= m_adaptive_factor;
      }
      if (get_iterations_count() + m_adaptive_interval > m_report_steps) {
        std::cout << "GuidedMCOptimizer: final adaptive acceptance fraction, "
                     "and standard deviation: "
                  << acceptance_fraction << " "
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

void GuidedMCOptimizer::check_input() {
  if (m_use_energy_change) {
    throw std::runtime_error(
        "GuidedMCOptimizer::check_input: using energy change is not supported");
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