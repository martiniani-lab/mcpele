#ifndef _MCPELE_UNIFORM_RECTANGULAR_SAMPLING_H__
#define _MCPELE_UNIFORM_RECTANGULAR_SAMPLING_H__

#include <random>
#include "pele/array.h"

namespace mcpele {
    
class UniformRectangularSampling : public TakeStep {
protected:
    std::mt19937_64 m_gen;
    std::uniform_real_distribution<double> m_dist05;
    pele::Array<double> m_boxvec;
    pele::Array<double> m_origin;
    bool m_cubic;
public:
    virtual ~UniformRectangularSampling() {}
    UniformRectangularSampling(const size_t seed, const pele::Array<double> boxvec)
        : m_gen(seed),
          m_dist05(-0.5, 0.5),
          m_boxvec(boxvec.copy()),
          m_origin()
    {}
    void set_generator_seed(const size_t inp) { m_gen.seed(inp); }
    void set_origin(const pele::Array<double> origin){m_origin = origin.copy();}
    virtual void displace(pele::Array<double>& coords, MC* mc)
    {
        if (coords.size() % m_boxvec.size()) {
            throw std::runtime_error("UniformRectangularSampling::displace: coords size incompatible with boxvec size");
        }
        const size_t nr_particles = coords.size() / m_boxvec.size();
        const size_t dim = m_boxvec.size();

        if (m_origin.empty()){
            for (size_t i = 0; i < nr_particles; ++i) {
                for (size_t j = 0; j < dim; ++j) {
                    int k = i * dim + j;
                    coords[k] = m_boxvec[j] * m_dist05(m_gen);
                }
            }
        }
        else{
            for (size_t i = 0; i < nr_particles; ++i) {
                for (size_t j = 0; j < dim; ++j) {
                    int k = i * dim + j;
                    coords[k] = m_origin[k] + m_boxvec[j] * m_dist05(m_gen);
                }
            }
        }

    }
};    
    
} // namespace mcpele

#endif // #ifndef _MCPELE_UNIFORM_RECTANGULAR_SAMPLING_H__
