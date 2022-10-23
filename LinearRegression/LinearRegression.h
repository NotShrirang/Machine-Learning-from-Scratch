#include <iostream>
#include <cmath>
#include <vector>

class LinearRegression
{
private:
    float m, c;
    std::vector<float> xs, ys, regression_line;

public:
    LinearRegression() {
        int a = 0;
    }
    void fit(std::vector<float> X, std::vector<float> y);
    void display();
    std::vector<float> predict(std::vector<float> X);
};

void LinearRegression::fit(std::vector<float> X, std::vector<float> y)
{
    float mean_x, mean_y, mean_Xy, mean_x_sq, mean_sq_X, slope, b, sum = 0, i = 0;
    for (auto &it : X)
    {
        sum = sum + it;
        i++;
    }
    mean_x = sum / i;
    sum = 0;
    i = 0;
    for (auto &it : y)
    {
        sum = sum + it;
        i++;
    }
    mean_y = sum / i;
    std::vector<float> Xy;
    for (int j = 0; j < i; j++)
    {
        Xy.push_back(X.at(j) * y.at(j));
    }
    std::vector<float> X_sq;
    for (int j = 0; j < i; j++)
    {
        X_sq.push_back(X.at(j) * X.at(j));
    }
    mean_x_sq = mean_x * mean_x;
    sum = 0;
    i = 0;
    for (auto &it : Xy)
    {
        sum = sum + it;
        i++;
    }
    mean_Xy = sum / i;
    sum = 0;
    i = 0;
    for (auto &it : X_sq)
    {
        sum = sum + it;
        i++;
    }
    mean_sq_X = sum / i;
    slope = (((mean_x * mean_y) - (mean_Xy)) / ((mean_x_sq) - (mean_sq_X)));
    b = mean_y - (slope * mean_x);
    m = slope;
    c = b;
}

std::vector<float> LinearRegression::predict(std::vector<float> X) {
    std::vector <float> y_predict;
    for (auto& it : X) {
        y_predict.push_back(m*it + c);
    }
    return y_predict;
}

void LinearRegression :: display() {
    std::cout << "Slope : " << m << "\nc : " << c;
}