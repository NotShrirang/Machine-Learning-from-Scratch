#include <iostream>
#include <cmath>
#include <vector>
#include "LinearRegression.h"
using namespace std;

int main()
{
    vector <float> X = { 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0 };
    vector <float> y = { 0.0, 3.0, 5.0, 7.0, 8.0, 10.0, 13.0, 14.0, 16.0, 19.0 };
    vector <float> x_test = { 5.0 };
    LinearRegression linear_reg;
    linear_reg.fit(X, y);
    linear_reg.display();
    vector <float> y_test = linear_reg.predict(x_test);
    cout<<"\n";
    for (auto& it : y_test) {
        cout << it << " ";
    }
    return 0;
}