#pragma once

#include <functional>

#include "common.h"


template <std::size_t N>
struct adam_t {
    inline static float c_beta1 = 0.9f;
    inline static float c_beta2 = 0.999f;
    inline static float c_epsilon = 1e-8f;

    // Internal state calculated for convenience
    // If you have a bunch of derivatives, you would probably want to store / calculate these once
    // for the entire gradient, instead of each derivative like this is doing.
    float beta1 = 1.0f;
    float beta2 = 1.0f;
    float m[N] = {0};
    float v[N] = {0};

    /* can be inplace: out can point to derivative */
    void get_adjusted(float derivative[N], float alpha, float out[N]) {
        // bias correction
        beta1 *= c_beta1;
        beta2 *= c_beta2;

        for (std::uint32_t i = 0; i < N; i++) {
            // exponential moving average of first and second moment
            m[i] = c_beta1 * m[i] + (1.0f - c_beta1) * derivative[i];
            v[i] = c_beta2 * v[i] + (1.0f - c_beta2) * derivative[i] * derivative[i];
     
            float mhat = m[i] / (1.0f - beta1);
            float vhat = v[i] / (1.0f - beta2);

            out[i] = alpha * mhat / (std::sqrt(vhat) + c_epsilon);
        }
    }
};

