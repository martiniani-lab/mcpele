#ifndef _MCPELE_CHECK_SPHERICAL_CONTAINER_CONFIG_H__
#define _MCPELE_CHECK_SPHERICAL_CONTAINER_CONFIG_H__

#include "mc.h"
#include "gmc.h"

namespace mcpele {

class CheckSphericalContainerConfig : public GMCConfTest {
 protected:
  double m_radius2;

 public:
  CheckSphericalContainerConfig(const double radius)
      : m_radius2(radius * radius) {}
  bool conf_test(pele::Array<double> &trial_coords, MCBase *mc) {
    return pele::dot(trial_coords, trial_coords) <= m_radius2;
  }
  pele::Array<double> gmc_gradient(pele::Array<double> &coords, MCBase *mc) {
    pele::Array<double> grad = coords.copy();
    grad /= norm(grad);
    grad *= -(1.0);
    return grad;
  }
  virtual ~CheckSphericalContainerConfig() {}
};

}  // namespace mcpele

#endif  // #ifndef _MCPELE_CHECK_SPHERICAL_CONTAINER_CONFIG_H__
