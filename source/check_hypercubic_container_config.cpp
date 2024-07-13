#include "mcpele/check_hypercubic_container_config.h"

pele::Array<double> mcpele::CheckHypercubicContainerConfig::gmc_gradient(
    pele::Array<double>& coords, MCBase* mc) {
  pele::Array gradient(coords.size(), 0.0);
  size_t min_index = coords.size();
  double min_distance = std::numeric_limits<double>::infinity();
  bool positive = true;
  for (size_t i = 0; i < coords.size(); ++i) {
    const double positive_distance = side_length_over_two - coords[i];
    const double negative_distance = coords[i] + side_length_over_two;
    assert(positive_distance >= 0.0);
    assert(negative_distance >= 0.0);
    double smaller_distance;
    bool smaller_positive;
    if (positive_distance < negative_distance) {
      smaller_distance = positive_distance;
      smaller_positive = true;
    } else {
      smaller_distance = negative_distance;
      smaller_positive = false;
    }
    if (smaller_distance < min_distance) {
      min_distance = smaller_distance;
      min_index = i;
      positive = smaller_positive;
    }
  }
  assert(min_index < coords.size() && min_distance > 0.0 && !std::isinf(min_distance));
  if (positive) {
    gradient[min_index] = -1.0;
  } else {
    gradient[min_index] = 1.0;
  }
  return gradient;
}
