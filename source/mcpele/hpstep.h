#ifndef _MCPELE_HP
#define _MCPELE_HP


/* Wrapper For Monte Carlo move */
/* Rewrite better if you want to use this later */

#include <cmath>
#include <algorithm>
#include <memory>
#include <stdexcept>
#include <random>

#include "mc.h"
#include "pele/array.h"
#include "pele/base_potential.h"
#include "histogram.h"
#include "pele/HPModel.h"





namespace mcpele {
    class HPStep : public TakeStep {
    public:
        virtual ~HPStep() {}
        HPStep() {};
        void displace(pele::Array<double> &coords, MC * mc);
    };
}









#endif