#ifndef _MCPELE_CHECK_HYPERCUBIC_CONTAINER_CONFIG_H__
#define _MCPELE_CHECK_HYPERCUBIC_CONTAINER_CONFIG_H__

#include "gmc.h"
#include "mc.h"

namespace mcpele {

class CheckHypercubicContainerConfig final : public GMCConfTest {
 protected:
  const double side_length_over_two;
  const double side_length;
  const bool use_powered_cosine_sum_gradient;

 public:
  explicit CheckHypercubicContainerConfig(
      const double side_length,
      const bool use_powered_cosine_sum_gradient = false)
      : side_length_over_two(side_length / 2.0),
        side_length(side_length),
        use_powered_cosine_sum_gradient(use_powered_cosine_sum_gradient) {
    if (side_length <= 0.0) {
      throw std::runtime_error("side length must be positive");
    }
  }
  bool conf_test(pele::Array<double> &trial_coords, MCBase *mc) override {
    return std::all_of(
        trial_coords.begin(), trial_coords.end(),
        [this](const double x) { return fabs(x) <= side_length_over_two; });
  }
  pele::Array<double> gmc_gradient(pele::Array<double> &coords,
                                   MCBase *mc) override;
  ~CheckHypercubicContainerConfig() override = default;
};

}  // namespace mcpele

#endif  // #ifndef _MCPELE_CHECK_HYPERCUBIC_CONTAINER_CONFIG_H__
