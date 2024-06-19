#include "compute.hpp"
inline void scale_vector(const std::vector<data_type>& input, std::vector<data_type>& output, const data_type scale) {
    std::transform(input.begin(), input.end(), output.begin(),
                   [scale](data_type x) { return x * scale; });
}

inline void multiply_vector(const std::vector<data_type>& input1, const std::vector<data_type>& input2, std::vector<data_type>& output) {
    std::transform(input1.begin(), input1.end(), input2.begin(), output.begin(),
                   [](data_type x, data_type y) { return x * y; });
}
inline void sub_vector(const std::vector<data_type>& input1, const std::vector<data_type>& input2, std::vector<data_type>& output) {
    std::transform(input1.begin(), input1.end(), input2.begin(), output.begin(),
                   [](data_type x, data_type y) { return x - y; });
}
inline void add_vector(const std::vector<data_type>& input1, const std::vector<data_type>& input2, std::vector<data_type>& output) {
    std::transform(input1.begin(), input1.end(), input2.begin(), output.begin(),
                   [](data_type x, data_type y) { return x + y; });
}

std::vector <data_type> gradient_poly6(const data_type &distance, const data_type kh, const std::vector<data_type> &r_ij)
{
    std::vector <data_type> grad(2,0);
    data_type h1 = 1/kh;
    auto fac = (4*h1*h1*h1*h1*h1*h1*h1*h1)*(M_1_PI);
    auto temp = kh*kh - distance*distance;
    if (0 < distance && distance <= kh)
    {
        auto temp_2 = fac*(-6)*temp*temp;
        scale_vector(r_ij, grad, temp);
    }

    return grad;
}
void calc_divergence(const std::vector<std::vector<data_type>> &pos,
                const std::vector<std::vector<data_type>> &vel,
                const std::vector<data_type> &density,
                const std::vector<int> &p_type,
                const std::vector<std::vector<unsigned int>> &nearIndex,
                const std::vector<std::vector<double>> &nearDist,
                std::vector<data_type> &divergence,
                const constants &c)
{
    
#pragma omp parallel for num_threads(10)
    for(unsigned int i = 0; i<pos.size(); i++)
    {
        if (p_type[i] == 1) // if Fluid particle
        {
            for(unsigned int j=0; j<nearIndex[i].size(); j++)
            {
               if (nearDist[i][j]>0 && nearDist[i][j]<=c.radius)
            //    if (nearDist[i][j]>0 && nearDist[i][j]<=c.radius && p_type[nearIndex[i][j]] == 1)
               {
                   std::vector<data_type> r_ij(2,0);
                   std::vector<data_type> weight(2,0);
                   std::vector<data_type> v_ji(2,0);

                   sub_vector(pos[i], pos[nearIndex[i][j]], r_ij);
                   sub_vector(vel[nearIndex[i][j]], vel[i], v_ji); // v_j - v_i and store in temp
                   weight = gradient_poly6(nearDist[i][j], c.radius, r_ij);
                   data_type temp = std::inner_product(weight.begin(), weight.end(), v_ji.begin(), 0.0);
                   temp = temp*c.mass*nearDist[i][j]/(nearDist[i][j]+c.Eta);
                   divergence[i] += temp;
               }
            }
            divergence[i] = divergence[i]/density[i];
        }
    }
}
