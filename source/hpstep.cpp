

/* Wrapper For Monte Carlo move */
/* Rewrite that it fits pele if you want to use this later */
#include "mcpele/mc.h"
#include "mcpele/hpstep.h"






namespace mcpele {
    void HPStep::displace(pele::Array<double> &coords, MC * mc) {
        std::shared_ptr<pele::HPModel> hpmodel = std::dynamic_pointer_cast<pele::HPModel>(mc -> m_potential);
        hpmodel->DoMCMove();
    }
}







