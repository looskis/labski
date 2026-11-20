#include "mm3_host.hpp"
#include <cmath>
#include <cstdio>
#include <vector>

int main() {
    std::vector<float> a(3075), b(3075), y(3072), ones(3072, 1.0f);
    for (size_t i = 0; i < a.size(); ++i) {
        a[i] = (int(i * 37 % 223) - 111) / 127.0f;
        b[i] = (int(i * 67 % 239) - 119) / 113.0f;
    }
    double max_dot = 0, max_rms = 0;
    for (int n : {0,1,7,8,31,32,33,63,64,128,1024,3072,3075}) {
        double expected = 0;
        for (int j = 0; j < n; ++j) expected += double(a[j]) * b[j];
        const double delta = std::abs(mm3::host::dot(a.data(), b.data(), n) - expected);
        max_dot = std::fmax(max_dot, delta);
        if (delta > 3e-5) return 1;
    }
    mm3::host::rms_rows(a.data(), b.data(), 3, 1024, 1e-6f, y.data());
    for (int r = 0; r < 3; ++r) {
        double ss = 0;
        for (int j = 0; j < 1024; ++j) ss += double(a[r*1024+j])*a[r*1024+j];
        double norm = 1 / std::sqrt(ss/1024 + 1e-6f);
        for (int j = 0; j < 1024; ++j) {
            int i = r*1024+j;
            max_rms = std::fmax(max_rms, std::abs(y[i] - a[i]*norm*b[j]));
        }
    }
    if (max_rms > 1e-5) return 2;
    const float* parts[] = {a.data(), b.data(), ones.data()};
    for (bool use_extra : {false, true}) {
        auto old = y;
        mm3::host::sum_into(y.data(), parts, 3, use_extra ? b.data() : nullptr, y.size());
        for (size_t i = 0; i < y.size(); ++i) {
            float v = (a[i]+b[i])+1.0f;
            if (use_extra) v += b[i];
            if (y[i] != old[i]+v) return 3;
        }
    }
    mm3::host::pause();
    std::printf("{\"dot_shapes\":13,\"dot_max_abs_error_vs_fp64\":%.9g,\"rms_outputs\":3072,\"rms_max_abs_error_vs_fp64\":%.9g,\"sum_outputs_exact\":6144}\n",max_dot,max_rms);
}
