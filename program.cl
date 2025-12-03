void kernel fft_window(global const float2* in, global float* out, ulong windtype) {
    if (windtype == 0) {

}
}



void kernel comp_utility(global const float2* in, global float2* out, ulong bitrate, ulong bufstart, float gain, float balance, uint muted, uint mute_l, uint mute_r, float width) {
    float al = in[get_global_id(0)].x;
    float ar = in[get_global_id(0)].y;
    float amp = exp10(gain/20);
    float bal = (balance + 1)/2;
    float lrdiff = width*(al-ar)/2.0;
    out[get_global_id(0)].x = (1-bal)*(!muted && !mute_l)*amp*(lrdiff + (al+ar)/2.0);
    out[get_global_id(0)].y = bal*(!muted && !mute_r)*amp*(-lrdiff + (al+ar)/2.0);
}

float2 sinf(ulong bitrate, ulong sample, float hz) {
    int whole = 0;
    float t = fract(hz * sample/((double) bitrate), &whole);
    return (float2) sinpi(t * 2);
}

float2 sawf(ulong bitrate, ulong sample, float hz, float shape) {
    int whole = 0;
    float t = fract(hz * sample/((double) bitrate), &whole);
    if (t < shape) {
        return (float2) clamp(t*2/shape - 1.0, -1.0, 1.0);
    } else {
        return (float2) clamp((t-shape)*(-2)/(1-shape) + 1.0, -1.0, 1.0);
    }
}

void kernel comp_synth(global const float2* in, global float2* out, ulong bitrate, ulong bufstart, uint tablesel, float hz, float shape) {
    float2 o;
    if (tablesel == 0) {
        o = sinf(bitrate, bufstart + get_global_id(0), hz);
    } else if (tablesel == 1) {
        o = sawf(bitrate, bufstart + get_global_id(0), hz, shape);
    }
    out[get_global_id(0)] = o;
}
