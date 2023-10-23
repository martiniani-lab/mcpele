#include <stdexcept>
#include "mcpele/record_cloud_drops_timeseries.h"

namespace mcpele{
/*
 * m_mcv mean coordinate vector
 * m_mcv2 element wise square mean coordinate vector, useful to compute the variance in each coordinate
 * */
RecordCloudDropsTimeseries::RecordCloudDropsTimeseries(pele::Array<double>& origin, const size_t record_every, const size_t eqsteps)
    : RecordCloudVectorTimeseries(record_every, eqsteps),
    m_origin(origin.copy())
    {}

pele::Array<double>  RecordCloudDropsTimeseries::get_recorded_vector(const Cloud& c, MC* mc)
{
    pele::Array<double> cloud_r_vector(c.size());

    for(size_t i=0; i<c.size(); ++i){
        std::shared_ptr<Drop> d = c[i];
        double r = 0;
        if (d->oracle) {
            double r2 = 0;
            for(size_t j=0; j < m_origin.size(); ++j){
                double dx = d->x[j] - m_origin[j];
                r2 += dx*dx;
            }
            r = std::sqrt(r2);
        }
        // returns 0 if oracle is 0
        cloud_r_vector[i] = d->oracle * r;
    }
    return cloud_r_vector;
}

} //namespace mcpele
