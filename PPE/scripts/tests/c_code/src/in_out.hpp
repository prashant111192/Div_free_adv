#ifndef IN_OUT_HPP
#define IN_OUT_HPP
#include <fstream>
#include <iostream>
#include <string>

#include "type_def.hpp"

template <typename T>
void writeMatrixToFile(const MatrixXX &pos, T &x, std::string filename)
    {
    // int precision = 6;
    LOG(INFO)<< "Writing to file: "<< filename;
    std::ofstream file(filename, std::ios::out);
    if (file.is_open()) {
        // file << std::fixed << std::setprecision(precision);
        for (int i = 0; i < x.rows(); ++i) {
            file<< pos(i,0) << "," << pos(i,1) << ",";
            for (int j = 0; j < x.cols(); ++j) {
                file << x(i, j);
                if (j < x.cols() - 1) {
                    file << ",";
                }
            }
            file << "\n";
        }
        file.close();
        LOG(INFO) << "Matrix written to " << filename;
    } else {
        LOG(ERROR) << "Unable to open file " << filename;
    }
}

#endif