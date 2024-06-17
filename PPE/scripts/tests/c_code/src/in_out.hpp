#include <iostream>
#include <vector>
#include <fstream>
#include <string>   

using namespace std;
template <typename ele_type>
ostream &operator<<(ostream &os, const vector<ele_type> &vect_name)
{
    for (auto itr : vect_name)
    {
        os << itr << " ";
    }
    return os;
}

template <typename T>
void save_data(T x, string filename)
{
    // save the vector as a csv file
    // x: vector to be saved, can be 1D or 2D
    // filename: name of the file to be saved
    ofstream file;
    file.open(filename);
    if (x.size() == 0)
    {
        file << "Empty vector" << endl;
    }
    else if (x[0].size() == 0)
    {
        for (int i = 0; i < x.size(); i++)
        {
            file << x[i] << endl;
        }
    }
    else
    {
        for (int i = 0; i < x.size(); i++)
        {
            for (int j = 0; j < x[i].size(); j++)
            {
                file << x[i][j] << ",";
            }
            file << endl;
        }
    }
}
