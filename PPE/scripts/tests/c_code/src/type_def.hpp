#include <chrono>
#define data_type double // data type for the particles // Need double for Knn, change accordingly. 

#pragma once


struct constants
{
    data_type h;
    data_type dp;
    data_type h_fac;
    data_type mass;
    data_type boundary_size;
    data_type x_y_bn;
    data_type x_y_bp;
    data_type x_y_n;
    data_type x_y_p;
    unsigned int resolution;
    unsigned int n_particles;
    unsigned int mid_idx;
    data_type Eta;
    data_type radius; // kh, radius of influence
};
