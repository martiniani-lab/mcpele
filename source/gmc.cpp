#include "mcpele/gmc.h"

namespace mcpele {

GMC::GMC(std::shared_ptr<pele::BasePotential> potential,
         const pele::Array<double>& coords, const double temperature,
         const double timestep, const size_t nparticles, const size_t ndim,
         const size_t rseed, const size_t resample_velocity_steps,
         const double max_timestep, const bool use_random_timestep,
         const size_t adaptive_interval, const double adaptive_factor,
         const double adaptive_min_acceptance_ratio,
         const double adaptive_max_acceptance_ratio)
    : MCBase(std::move(potential), coords, temperature),
      m_take_step(std::make_shared<GMCTakeStep>(timestep, nparticles, ndim,
                                                rseed, max_timestep,
                                                use_random_timestep)),
      m_adaptive_step(std::make_shared<AdaptiveTakeStep>(
          m_take_step, adaptive_interval, adaptive_factor,
          adaptive_min_acceptance_ratio, adaptive_max_acceptance_ratio)),
      accumulated_gradient(coords.size()),
      m_E_reject_count(0),
      m_conf_reject_count(0),
      resample_velocity_steps(resample_velocity_steps) {
  m_use_energy_change = false;
  if (coords.size() != nparticles * ndim) {
    throw std::runtime_error(
        "GMC::GMC: coords size does not match nparticles * ndim");
  }
}

bool GMC::check_configuration_short(pele::Array<double>& trial_coords) {
  auto success = do_conf_tests(trial_coords);
  if (success) {
    const auto trial_energy = compute_energy(trial_coords);
    success = do_accept_tests(trial_coords, trial_energy, m_coords, m_energy);
    if (success) {
      success = do_late_conf_tests(trial_coords);
    }
  }
  return success;
}

void GMC::one_iteration() {
  // TODO: It should be possible to store information about a suceeding conf
  // test from the past time step!
  ++m_niter;
  ++m_nitercount;

  m_trial_coords.assign(m_coords);
  m_take_step->set_current_step_name(this);
  m_take_step->displace(m_trial_coords, this);

  accumulated_gradient = pele::Array(m_coords.size(), 0.0);

  bool success_conf = true;
  for (const auto& test : m_conf_tests) {
    if (const bool result = test->conf_test(m_trial_coords, this); !result) {
      success_conf = false;
    }
    const auto gradient = test->gmc_gradient(m_coords, this);
    assert(fabs(norm(gradient) - 1.0) < 1.0e-12);
    accumulated_gradient += gradient;
  }

  bool success_accept = true;
  // TODO: Allow to use compute_energy_change.
  m_trial_energy = compute_energy(m_trial_coords);
  for (const auto& test : m_accept_tests) {
    if (const bool result = test->test(m_trial_coords, m_trial_energy, m_coords,
                                       m_energy, m_temperature, this);
        !result) {
      success_accept = false;
    }
  }

  bool success_late_conf = true;
  for (const auto& test : m_late_conf_tests) {
    if (const bool result = test->conf_test(m_trial_coords, this); !result) {
      success_late_conf = false;
    }
    const auto gradient = test->gmc_gradient(m_coords, this);
    assert(fabs(norm(gradient) - 1.0) < 1.0e-12);
    accumulated_gradient += gradient;
  }

  m_success = success_conf && success_accept && success_late_conf;
  if (!m_success) {
    if (!success_conf || !success_late_conf) ++m_conf_reject_count;
    if (!success_accept) ++m_E_reject_count;

    assert(std::any_of(accumulated_gradient.begin(), accumulated_gradient.end(),
                       [](const double x) { return x != 0.0; }));
    accumulated_gradient /= norm(accumulated_gradient);
    auto old_velocity = m_take_step->get_velocity();
    const auto new_velocity =
        old_velocity -
        2.0 * dot(old_velocity, accumulated_gradient) * accumulated_gradient;
    auto east_pos = m_coords.copy();
    m_take_step->set_velocity(new_velocity);
    m_take_step->displace(east_pos, this);
    auto west_pos = m_coords.copy();
    m_take_step->set_velocity(-1.0 * new_velocity);
    m_take_step->displace(west_pos, this);
    auto south_pos = m_coords.copy();
    m_take_step->set_velocity(-1.0 * old_velocity);
    m_take_step->displace(south_pos, this);

    const auto south = check_configuration_short(south_pos);
    if (!south) {
      m_take_step->set_velocity(-1.0 * old_velocity);  // Go south.
    } else {
      const auto east = check_configuration_short(east_pos);
      const auto west = check_configuration_short(west_pos);
      if (east && !west) {
        m_take_step->set_velocity(new_velocity);  // Go east.
      } else if (west && !east) {
        m_take_step->set_velocity((-1.0) * new_velocity);  // Go west.
      } else {
        m_take_step->set_velocity(-1.0 * old_velocity);  // Go south.
      }
    }
  }

  if (resample_velocity_steps > 0 &&
      get_iterations_count() % resample_velocity_steps == 0) {
    m_take_step->resample_velocity();
  }

  // adapt stepsize etc.
  if (get_iterations_count() <= m_report_steps) {
    m_adaptive_step->report(m_coords, m_energy, m_trial_coords, m_trial_energy,
                            m_success, this);
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
  }

  if (m_coords.get_max() > 1e9) {
    std::cout << "WARNING: this can lead to errors in the energy calculation "
                 "due to overflow";
  }

  // perform the actions on the new configuration
  do_actions(m_coords, m_energy, m_success);

  m_last_success = m_success;
}

void GMC::check_input() {
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
    if (m_accept_tests.size() == 0) {
      std::cout << "warning: no accept tests set"
                << "\n";
    }
  }
}

}  // namespace mcpele