#ifndef _MCPELE_LOWEST_EIGENVALUE_H
#define _MCPELE_LOWEST_EIGENVALUE_H

#include "pele/base_potential.hpp"
#include "pele/lbfgs.hpp"
#include "pele/lowest_eig_potential.hpp"

namespace mcpele {

class FindLowestEigenvalue {
 private:
  std::shared_ptr<pele::LowestEigPotential> m_lowesteigpot;
  pele::Array<double> m_ranvec;
  pele::LBFGS m_lbfgs;

 public:
  FindLowestEigenvalue(std::shared_ptr<pele::BasePotential> landscape_potential,
                       const size_t boxdimension,
                       const pele::Array<double> ranvec,
                       const size_t lbfgsniter);
  double compute_lowest_eigenvalue(pele::Array<double> const &coords);
};

}  // namespace mcpele

#endif  //#ifndef _MCPELE_LOWEST_EIGENVALUE_H
