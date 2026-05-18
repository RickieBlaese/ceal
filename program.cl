constant float PI = 3.14159265f;
constant float ln2 = 0.693147181f;

int modi(int a, int b) {
    int r = a % b;
    return r < 0 ? r + b : r;
}

void kernel fft_window(global const float2* in, global float* out, ulong windtype) {
}


void kernel comp_utility(global const float2* in, global float2* out, ulong sample_rate, ulong bufstart, ulong groupwidth, float gain, float balance, uint muted, uint mute_l, uint mute_r, float width) {
    for (uint i = 0; i < groupwidth; i++) {
        ulong offset = get_global_id(0)*groupwidth + i;
        float al = in[offset].x;
        float ar = in[offset].y;
        float amp = exp10(gain/20);
        float bal = (balance + 1)/2;
        float lrdiff = width*(al-ar)/2.0;
        out[offset].x = (1-bal)*(!muted && !mute_l)*amp*(lrdiff + (al+ar)/2.0);
        out[offset].y = bal*(!muted && !mute_r)*amp*(-lrdiff + (al+ar)/2.0);
    }
}

float2 sinf(ulong sample_rate, ulong sample, float hz) {
    float whole = 0;
    float t = fract(hz * sample/convert_float(sample_rate), &whole);
    return (float2) sinpi(t * 2);
}

float2 sawf(ulong sample_rate, ulong sample, float hz, float shape) {
    float whole = 0;
    float t = fract(hz * sample/convert_float(sample_rate), &whole);
    if (t < shape) {
        return (float2) clamp(t*2/shape - 1.0, -1.0, 1.0);
    } else {
        return (float2) clamp((t-shape)*(-2)/(1-shape) + 1.0, -1.0, 1.0);
    }
}

float2 gaussianf(ulong sample_rate, ulong sample, float hz) {
    float whole = 0;
    float t = (fract(hz * sample/convert_float(sample_rate), &whole) - 0.5)*4;
    return exp(-PI * t * t);
}

void kernel comp_synth(global const float2* in, global float2* out, ulong sample_rate, ulong bufstart, ulong groupwidth, uint tablesel, float hz, float shape) {
    for (uint i = 0; i < groupwidth; i++) {
        float2 o;
        ulong offset = get_global_id(0)*groupwidth + i;
        if (tablesel == 0) {
            o = sinf(sample_rate, bufstart + offset, hz);
        } else if (tablesel == 1) {
            o = sawf(sample_rate, bufstart + offset, hz, shape);
        } else if (tablesel == 2) {
            o = gaussianf(sample_rate, bufstart + offset, hz);
        } else if (tablesel == 3) { /* impulse */
            o = (bufstart + offset == 0 ? 1.0f : 0.0f);
        }

        out[offset] = o;
    }
}


void kernel comp_lpf(global const float2* incatprev /*sampout+2*/, global float2* outcatprev /*sampout+2*/, ulong sample_rate, ulong bufstart, ulong sampout, global float* f0, global float* q) {
    const int adjlen = sampout+2;
    float Fs = sample_rate;

    for (int i = 0; i < sampout; i++) {
        float omega = 2*PI*f0[i]/Fs;
        float s = sin(omega);
        float c = cos(omega);
        float alpha = s/2/q[i];
        float b0 = (1-c)/2;
        float b1 = (1-c);
        float b2 = b0;
        float a0 = 1+alpha;
        float a1 = -2*c;
        float a2 = 1-alpha;
        outcatprev[i] = b0/a0*incatprev[i] + b1/a0*incatprev[modi(i-1, adjlen)] + b2/a0*incatprev[modi(i-2, adjlen)] - a1/a0*outcatprev[modi(i-1, adjlen)] - a2/a0*outcatprev[modi(i-2, adjlen)];
    }
}

void kernel comp_lpf2(global const float2* incatprev /*sampout+4*/, global float2* outcatprev /*sampout+4*/, ulong sample_rate, ulong bufstart, ulong groupwidth, float f0, float q) {
    const int adjlen = groupwidth*2+4;
    float Fs = sample_rate;

    float omega = 2*PI*f0/Fs;
    float s = sin(omega);
    float c = cos(omega);
    float alpha = s/2/q;
    float b0 = (1-c)/2;
    float b1 = (1-c);
    float b2 = b0;
    float a0 = 1+alpha;
    float a1 = -2*c;
    float a2 = 1-alpha;

    /* float u = (a1/a0 - sqrt((a1/a0)*(a1/a0) + 4*a2/a0))/2; */
    /* float r = a1/a0 - u; */
    float rps = -a1/a0;
    float rs = a2/a0;

    float ja0 = a0;
    float ja2 = a0*rs + a1*rps + a2;
    float ja4 = a2*rs;
    float jb0 = b0;
    float jb1 = b1 + b0*rps;
    float jb2 = b0*rs + b1*rps + b2;
    float jb3 = b1*rs + b2*rps;
    float jb4 = b2*b2/b1;


    for (int i = get_global_id(0); i < groupwidth*2; i+=2) {
        outcatprev[i] = jb0/ja0*incatprev[i] + jb1/ja0*incatprev[modi(i-1, adjlen)] + jb2/ja0*incatprev[modi(i-2, adjlen)] + jb3/ja0*incatprev[modi(i-3, adjlen)] + jb4/ja0*incatprev[modi(i-4, adjlen)] - ja2/ja0*outcatprev[modi(i-2, adjlen)] - ja4/ja0*outcatprev[modi(i-4, adjlen)];
        /* printf("get_global_id:%i, i:%u, out:%.1v2hlf, incat:%.1v2hlf:%.1v2hlf:%.1v2hlf:%.1v2hlf, outcat:%.1v2hlf:%.1v2hlf:%.1v2hlf:%.1v2hlf\n", get_global_id(0), i, outcatprev[i], incatprev[modi(-1, adjlen)], incatprev[modi(-2, adjlen)], incatprev[modi(-3, adjlen)], incatprev[modi(-4, adjlen)], outcatprev[modi(-1, adjlen)], outcatprev[modi(-2, adjlen)], outcatprev[modi(-3, adjlen)], outcatprev[modi(-4, adjlen)]); */
    }
}

void kernel comp_hpf(global const float2* incatprev /*sampout+2*/, global float2* outcatprev /*sampout+2*/, ulong sample_rate, ulong bufstart, ulong sampout, float f0, float q) {
    const int adjlen = sampout+2;
    float Fs = sample_rate;
    float omega = 2*PI*f0/Fs;
    float s = sin(omega);
    float c = cos(omega);
    float alpha = s/2/q;
    float b0 = (1+c)/2;
    float b1 = c-1;
    float b2 = b0;
    float a0 = 1+alpha;
    float a1 = -2*c;
    float a2 = 1-alpha;

    for (int i = 0; i < sampout; i++) {
        outcatprev[i] = b0/a0*incatprev[i] + b1/a0*incatprev[modi(i-1, adjlen)] + b2/a0*incatprev[modi(i-2, adjlen)] - a1/a0*outcatprev[modi(i-1, adjlen)] - a2/a0*outcatprev[modi(i-2, adjlen)];
    }
}

void kernel comp_notchf(global const float2* incatprev /*sampout+2*/, global float2* outcatprev /*sampout+2*/, ulong sample_rate, ulong bufstart, ulong sampout, float f0, float q) {
    const int adjlen = sampout+2;
    float Fs = sample_rate;
    float omega = 2*PI*f0/Fs;
    float s = sin(omega);
    float c = cos(omega);
    float alpha = s/2/q;
    float b0 = 1;
    float b1 = -2*c;
    float b2 = 1;
    float a0 = 1+alpha;
    float a1 = -2*c;
    float a2 = 1-alpha;

    for (int i = 0; i < sampout; i++) {
        outcatprev[i] = b0/a0*incatprev[i] + b1/a0*incatprev[modi(i-1, adjlen)] + b2/a0*incatprev[modi(i-2, adjlen)] - a1/a0*outcatprev[modi(i-1, adjlen)] - a2/a0*outcatprev[modi(i-2, adjlen)];
    }
}

void kernel comp_apf(global const float2* incatprev /*sampout+2*/, global float2* outcatprev /*sampout+2*/, ulong sample_rate, ulong bufstart, ulong sampout, float f0, float q) {
    const int adjlen = sampout+2;
    float Fs = sample_rate;
    float omega = 2*PI*f0/Fs;
    float s = sin(omega);
    float c = cos(omega);
    float alpha = s/2/q;
    float b0 = 1-alpha;
    float b1 = -2*c;
    float b2 = 1+alpha;
    float a0 = 1+alpha;
    float a1 = -2*c;
    float a2 = 1-alpha;

    for (int i = 0; i < sampout; i++) {
        outcatprev[i] = b0/a0*incatprev[i] + b1/a0*incatprev[modi(i-1, adjlen)] + b2/a0*incatprev[modi(i-2, adjlen)] - a1/a0*outcatprev[modi(i-1, adjlen)] - a2/a0*outcatprev[modi(i-2, adjlen)];
    }
}

/* constant 0 peak gain version */
void kernel comp_bpf(global const float2* incatprev /*sampout+2*/, global float2* outcatprev /*sampout+2*/, ulong sample_rate, ulong bufstart, ulong sampout, float f0, float bw) {
    const int adjlen = sampout+2;
    float Fs = sample_rate;
    float omega = 2*PI*f0/Fs;
    float s = sin(omega);
    float c = cos(omega);
    float alpha = s/2 * sinh(ln2/2 * bw * omega/s);

    float b0 = alpha;
    float b1 = 0;
    float b2 = -alpha;
    float a0 = 1+alpha;
    float a1 = -2*c;
    float a2 = 1-alpha;

    for (int i = 0; i < sampout; i++) {
        outcatprev[i] = b0/a0*incatprev[i] + b1/a0*incatprev[modi(i-1, adjlen)] + b2/a0*incatprev[modi(i-2, adjlen)] - a1/a0*outcatprev[modi(i-1, adjlen)] - a2/a0*outcatprev[modi(i-2, adjlen)];
    }
}

void kernel comp_peak(global const float2* incatprev /*sampout+2*/, global float2* outcatprev /*sampout+2*/, ulong sample_rate, ulong bufstart, ulong sampout, float f0, float gain, float q) {
    const int adjlen = sampout+2;
    float Fs = sample_rate;
    float omega = 2*PI*f0/Fs;
    float s = sin(omega);
    float c = cos(omega);
    float A = exp10(gain/40.0f);
    float alpha = s/2/q;
    float b0 = 1+alpha*A;
    float b1 = -2*c;
    float b2 = 1-alpha*A;
    float a0 = 1+alpha/A;
    float a1 = -2*c;
    float a2 = 1-alpha/A;

    for (int i = 0; i < sampout; i++) {
        outcatprev[i] = b0/a0*incatprev[i] + b1/a0*incatprev[modi(i-1, adjlen)] + b2/a0*incatprev[modi(i-2, adjlen)] - a1/a0*outcatprev[modi(i-1, adjlen)] - a2/a0*outcatprev[modi(i-2, adjlen)];
    }
}

void kernel comp_lowshelf(global const float2* incatprev /*sampout+2*/, global float2* outcatprev /*sampout+2*/, ulong sample_rate, ulong bufstart, ulong sampout, float f0, float gain, float slope) {
    const int adjlen = sampout+2;
    float Fs = sample_rate;
    float omega = 2*PI*f0/Fs;
    float s = sin(omega);
    float c = cos(omega);
    float A = exp10(gain/40.0f);
    float alpha = s/2 * sqrt((A+1/A)*(1/slope - 1) + 2);

    float u = (A+1) - (A-1)*c;
    float v = (A-1) - (A+1)*c;
    float up = (A+1) + (A-1)*c;
    float vp = (A-1) + (A+1)*c;
    float sq = 2*sqrt(A)*alpha;

    float b0 = A*(u + sq);
    float b1 = 2*A*v;
    float b2 = A*(u - sq);
    float a0 = up + sq;
    float a1 = -2*vp;
    float a2 = up - sq;

    for (int i = 0; i < sampout; i++) {
        outcatprev[i] = b0/a0*incatprev[i] + b1/a0*incatprev[modi(i-1, adjlen)] + b2/a0*incatprev[modi(i-2, adjlen)] - a1/a0*outcatprev[modi(i-1, adjlen)] - a2/a0*outcatprev[modi(i-2, adjlen)];
    }
}

void kernel comp_highshelf(global const float2* incatprev /*sampout+2*/, global float2* outcatprev /*sampout+2*/, ulong sample_rate, ulong bufstart, ulong sampout, float f0, float gain, float slope) {
    const int adjlen = sampout+2;
    float Fs = sample_rate;
    float omega = 2*PI*f0/Fs;
    float s = sin(omega);
    float c = cos(omega);
    float A = exp10(gain/40.0f);
    float alpha = s/2 * sqrt((A+1/A)*(1/slope - 1) + 2);

    float u = (A+1) - (A-1)*c;
    float v = (A-1) - (A+1)*c;
    float up = (A+1) + (A-1)*c;
    float vp = (A-1) + (A+1)*c;
    float sq = 2*sqrt(A)*alpha;

    float b0 = A*(up + sq);
    float b1 = -2*A*vp;
    float b2 = A*(up - sq);
    float a0 = u + sq;
    float a1 = 2*v;
    float a2 = u - sq;

    for (int i = 0; i < sampout; i++) {
        outcatprev[i] = b0/a0*incatprev[i] + b1/a0*incatprev[modi(i-1, adjlen)] + b2/a0*incatprev[modi(i-2, adjlen)] - a1/a0*outcatprev[modi(i-1, adjlen)] - a2/a0*outcatprev[modi(i-2, adjlen)];
    }
}

