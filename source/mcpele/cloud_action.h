#ifndef _MCPELE_RECORD_CLOUD_ACTION_H__
#define _MCPELE_RECORD_CLOUD_ACTION_H__

#include "cloud_test.h"

namespace mcpele {

class CloudAction : public Action {
public:
    virtual ~CloudAction() {}

    void action(pele::Array<double>& coords, double energy, bool accepted, MC* mc)
    {
        const Cloud& c = static_cast<CloudTest*>(mc->display_accept_tests().at(0).get())->display_old_cloud();
        record_cloud_action(c, mc);
    }

    double get_cloud_weighted_average(const Cloud& c, MC* mc){
        // TODO: Add passing of result to python output
        double weighted_sum = 0;
        double sum_of_weights = 0;
        for (const std::shared_ptr<Drop> drop : c) {
            const double tmp = drop->bias * drop->oracle;
            weighted_sum += tmp * this->do_action(drop, mc);
            sum_of_weights += tmp;
        }
        return weighted_sum / sum_of_weights;
    }

    virtual void record_cloud_action(const Cloud& c, MC* mc) = 0;
    virtual double do_action(const std::shared_ptr<Drop> drop, MC* mc) = 0;
};

} // namespace mcpele

#endif // #ifndef _MCPELE_RECORD_CLOUD_ACTION_H__

