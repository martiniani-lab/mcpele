#ifndef _MCPELE_RECORD_CLOUD_R2_H__
#define _MCPELE_RECORD_CLOUD_R2_H__

#include "cloud_test.h"
#include "cloud_action.h"
#include "histogram.h"

namespace mcpele {

class RecordCloudR2 : public CloudAction {
    const pele::Array<double> m_origin;
    const size_t m_eqsteps;
    Moments m_accumulator_m1, m_accumulator_m2;
    bool m_record_m1;
public:
    RecordCloudR2(const size_t eqsteps, pele::Array<double>& origin)
        : m_origin(origin.copy()),
          m_eqsteps(eqsteps),
          m_accumulator_m1(),
          m_accumulator_m2(),
          m_record_m1(true)
    {}
    void record_cloud_action(const Cloud& c, MC* mc)
    {
        if (mc->get_iterations_count() > m_eqsteps) {
            // record r2
            m_record_m1 = true;
            double weighted_average_m1 = this->get_cloud_weighted_average(c, mc);
            m_accumulator_m1.update(weighted_average_m1);
            // record r2*r2
            m_record_m1 = false;
            double weighted_average_m2 = this->get_cloud_weighted_average(c, mc);
            m_accumulator_m2.update(weighted_average_m2);
        }
    }

    double do_action(const std::shared_ptr<Drop> drop, MC* mc){
        double r2 = get_drop_r2(drop->x);
        if (m_record_m1){
            return r2;
            }
        else{
            return r2*r2;
            }
    }

    double get_drop_r2(const pele::Array<double>& drop_x) const
    {
        double r2 = 0;
        for (size_t i = 0; i < drop_x.size(); ++i) {
            const double tmp = drop_x[i] - m_origin[i];
            r2 += tmp * tmp;
        }
        return r2;
    }
    double get_mean_r2() const
    {
        return m_accumulator_m1.mean();
    }
    double get_var_r2() const
    {
        return m_accumulator_m2.mean() - m_accumulator_m1.mean()*m_accumulator_m1.mean();
    }
};

} // namespace mcpele

#endif // #ifndef _MCPELE_RECORD_CLOUD_R2_H__
