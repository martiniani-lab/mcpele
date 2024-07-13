#ifndef _MCPELE_CHECK_HYPERCUBIC_CONTAINER_CONFIG_H__
#define _MCPELE_CHECK_HYPERCUBIC_CONTAINER_CONFIG_H__

#include "gmc.h"
#include "mc.h"

namespace mcpele {

class CheckHypercubicContainerConfig final : public GMCConfTest {
 protected:
  double side_length_over_two;

 public:
  explicit CheckHypercubicContainerConfig(const double side_length)
      : side_length_over_two(side_length / 2.0) {
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
