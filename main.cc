#include <iostream>
#include <vector>
#include <random>
#include <thread>
#include <array>
using namespace std;

struct Matrix {
    Matrix(int n) : n(n), data(n * n) {}
    int n;
    vector<float> data;
    float operator()(int x, int y) const { return data[y * n + x]; }
    float& operator()(int x, int y) { return data[y * n + x]; }
    void clear() { for (auto& e : data) e = 0; }
};

// 3.82s for N=4096 with 4 cores on 2017 imac 4.2GHz i7
// 1.07s for N=4096 with 32 cores on 2019 AMD TR3 4.1GHz 3960
void gemm(const Matrix& a, const Matrix& b, Matrix& c) {
    int n = a.n;
    c.clear();

    array<thread, 8> w;
    for (int e = 0; e < 8; e++) w[e] = thread([e, n, &a, &b, &c]() {
        int js = e * n / 8;
        int je = js + n / 8;
        for (int j = js; j < je; j++)
            for (int k = 0; k < n; k++)
                for (int i = 0; i < n; i++)
                    c(i, j) += a(i, k) * b(k, j);
    });
    for (auto& e : w) e.join();
}

float sum(const Matrix& a) {
    float s = 0;
    for (int i = 0; i < a.n; i++)
        for (int j = 0; j < a.n; j++) {
            s += a(i, j);
        }
    return s;
}

int main(int argc, char** argv) {
    int n = 1024 * 4;
    Matrix a(n), b(n), c(n);
    std::mt19937_64 re(0);
    std::normal_distribution<float> dis(0, 1);

    for (int i = 0; i < a.n; i++)
        for (int j = 0; j < a.n; j++)
            a(i, j) = dis(re);

    for (int i = 0; i < b.n; i++)
        for (int j = 0; j < b.n; j++)
            b(i, j) = dis(re);

    unsigned long ta = __builtin_readcyclecounter();
    gemm(a, b, c);
    unsigned long tb = __builtin_readcyclecounter();
    std::cout << "gemm " << (tb - ta) / 4.2e9 << " " << sum(c) << std::endl;
}
