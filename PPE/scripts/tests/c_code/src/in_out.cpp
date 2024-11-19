#include "in_out.hpp"

void make_from_dsph(const constants &c, MatrixXX &pos, MatrixXX &vel, MatrixXX &density, Eigen::MatrixXi &p_type, MatrixXX &pressure)
{

    LOG(INFO) << "Reading DSPH file";
    std::string path("./DSPHdata/");
    std::string ext(".csv");
    for (auto &file : std::filesystem::directory_iterator(path))
    {
        if  (file.path().extension() == ext)
        {
            std::string filename = file.path().string();
            // std::string filename = file.path().filename().string();
            std::ifstream inFile(filename);
            LOG(INFO) << filename;
            std::string datLineStr;
            long int current_line = 0;
            while(std::getline(inFile, datLineStr))
            {
                if (current_line > 2)
                {
                    std::stringstream datLine(datLineStr);
                    std::string data;
                    std::vector<data_type> dataVec;
                    while(std::getline(datLine, data, ';'))
                    {
                        dataVec.push_back(std::stod(data));
                    }
                    pos(current_line-3, 0) = dataVec[0];
                    pos(current_line-3, 1) = dataVec[2];

                    // if (dataVec[0] > -.5 && dataVec[0] < 0.5 && dataVec[2] > -10 && dataVec[2] < 0 && dataVec[8] ==3)
                    // {
                    //     vel(current_line-3, 0) = 0;
                    //     vel(current_line-3, 1) = 1.5;
                    // }
                    // else{
                    //     vel(current_line-3, 0) = 0;
                    //     vel(current_line-3, 1) = 0;
                    // }
                    vel(current_line-3, 0) = dataVec[4];
                    vel(current_line-3, 1) = dataVec[6];
                    if (dataVec[8] == 3) // Fluid
                        p_type(current_line-3, 0) = 1;
                    else                 // Boundary
                        p_type(current_line-3, 0) = 0;

                    if (dataVec[7] > 1020)
                        density(current_line-3, 0) = 1020;
                        else
                        density(current_line-3, 0) = dataVec[7];
                    // if (dataVec[0] > -0.1 && dataVec[0] < 0.1 && dataVec[2] > -0.1 && dataVec[2] < 0.1)
                    // {
                    //     std::cout<< "position:" << dataVec[0] << " " << dataVec[2] << std::endl;
                    //     std::cout<< "velocity:" << dataVec[4] << " " << dataVec[6] << std::endl;
                    //     std::cout<< "density:" << dataVec[7] << std::endl;
                    //     std::cout<< "type:" << dataVec[8] << std::endl;
                    // }
                    // if (pos(current_line-3, 0) < 0.05 && pos(current_line-3, 0) > -0.05 && pos(current_line-3, 1) < 0.05 && pos(current_line-3, 1) > -0.05 && 1==0)
                    //     {
                    //         vel(current_line-3, 0) = 0;
                    //         vel(current_line-3, 1) = 0;
                    //         p_type(current_line-3, 0) = 0;

                    //     }
                    //     else{

                    //         vel(current_line-3, 0) = dataVec[4];
                    //         vel(current_line-3, 1) = dataVec[6];
                    //         if (dataVec[8] == 3) // Fluid
                    //             p_type(current_line-3, 0) = 1;
                    //         else                 // Boundary
                    //             p_type(current_line-3, 0) = 0;
                    //     }
                    // if (dataVec[7] > 1020)
                    //     density(current_line-3, 0) = 1020;
                    //     else
                    //     density(current_line-3, 0) = dataVec[7];
                    // pressure(current_line-3, 0) = dataVec[8];
                }
                else{
                    std::cout<< "current line: " << current_line <<":" << datLineStr << std::endl;
                }
                current_line ++;
            }
        }
    }
}


void make_dsph_input(const constants &c, MatrixXX &pos,
                     MatrixXX &vel, MatrixXX &density,
                     Eigen::MatrixXi &p_type,
                     MatrixXX &pressure)
{
    std::string filename = "adv.csv";
    LOG(INFO)<< "Making DSPH input file called: "<< filename;
    auto chrono_start = std::chrono::high_resolution_clock::now();

    std::ofstream file(filename, std::ios::out);
    if (file.is_open()) {
        for (int i = 0; i<3; i++)
        {
            file << "JUST PLACEHOLDERS\n";
        }

        for (int i = 0; i < pos.rows(); ++i) {
            int Type, MK;
            if (p_type(i, 0) == 1) // Fluid
            {
                Type = 3;
                MK = std::experimental::randint(1, 2);
            }
            else // Solids
            {
                Type = 0;
                MK = 10;
            }

            file<< pos(i,0) << ";" << 
            0 << ";" <<
            pos(i,1) << ";" << 
            i << ";" <<
            vel(i,0) << ";" << 
            0 << ";" <<
            vel(i,1) << ";" << 
            density(i,0) << ";" << 
            Type<< ";"<<
            MK << ";" << "\n";
        }
        file.close();
        LOG(INFO) << "Matrix written to " << filename;
    } else {
        LOG(ERROR) << "Unable to open file " << filename;
    }
    auto chrono_end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(chrono_end - chrono_start).count();
    CLOG(INFO, "TIME") << "Writing ADVFILE (csv) "<< filename << " took(s):" << duration/10e6 ;
    
    
    
}