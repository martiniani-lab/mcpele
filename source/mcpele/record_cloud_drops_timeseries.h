#ifndef _MCPELE_RECORD_CLOUD_DROPS_TIMESERIES_H__
#define _MCPELE_RECORD_CLOUD_DROPS_TIMESERIES_H__

#include "record_vector_timeseries.h"

namespace mcpele {

class RecordCloudDropsTimeseries : public RecordCloudVectorTimeseries {
protected:
    pele::Array<double> m_origin;
public:
    RecordCloudDropsTimeseries(pele::Array<double>& origin, const size_t record_every, const size_t eqsteps);
    virtual ~RecordCloudDropsTimeseries(){}
    virtual pele::Array<double> get_recorded_vector(const Cloud& c, MC* mc);
};


} // namespace mcpele

#endif // #ifndef _MCPELE_RECORD_VECTOR_TIMESERIES_H__
