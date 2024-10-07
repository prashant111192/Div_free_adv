#include <iostream>
#include <stdio.h>
#include <math.h>

template <typename T>
T pow(T base, int exp)
{
    T result = 1;
    int power;
    T base_ = base;
    power = exp;
    if (power<0)
    {
        base_ = 1/base_;
        power = -1*power;
    }
    while (power!=0)
    {
        result = result*base_;
        power = power-1;
    }
    // while (power)
    // {
    //     if (exp & 1)
    //         result *= base;
    //     exp >>= 1;
    //     base *= base;
    // }
    // std::cout<< "base = " << base << " exp = " << exp << " and the result = " << result << std::endl;
    return result;
}
int main()
{
    std::cout<< std::fixed;
    long double a, b, c;
    long double base = 10;

    for (int i=0; i<100; i++)
    {
        a = pow(base,i);
        b = 0.1;
        // b = pow(base,(-1*i));
        c = a +b;
        std::cout << a <<" + "<<b<<" = " << c <<" for i= "<< i<< std::endl;
    }
    return 0;
}