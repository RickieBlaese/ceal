#pragma once

#include <algorithm>
#include <iostream>
#include <numeric>
#include <vector>

#include <cmath>
#include <cstdint>

#include <CL/opencl.hpp>
#include <clFFT.h>

/* negative not fully implemented */
struct rational_t {
    std::uint64_t num = 1, den = 1;
    bool pos = true;

    rational_t(std::int64_t numerator, std::int64_t denominator);

    rational_t(std::uint64_t numerator, std::uint64_t denominator);

    rational_t(float a, std::uint64_t denominator);

    template <typename T>
    T cast() const {
        return (2*pos - 1) * static_cast<T>(num) / static_cast<T>(den);
    }

    /* simplifiy */
    rational_t &smp();

    rational_t &neg();

    rational_t &add(const rational_t &other);

    rational_t &sub(const rational_t &other);

    rational_t &inv();

    rational_t &mul(const rational_t &other);

    rational_t &div(const rational_t &other);

    /* strict less than */
    bool less(const rational_t &other) const;
};

/* [a, b).
 * T is int-like.
 * note: overwrites out */
template <typename T>
void partition(T a, T b, std::size_t n, std::vector<T> &out) {
    long double i = 0;
    for (std::uint32_t ind = 0; ind < n; ind++) {
        out[ind] = static_cast<T>(std::round(i));
        i += static_cast<long double>(b-a)/n;
    }
}

std::size_t next_exp_2(std::size_t n);


template <typename T, std::size_t N>
struct managed_set_clbuf_t {
    std::array<cl_mem_flags, N> flags;
    std::array<cl::Buffer, N> buffers;
    bool init = false;
    std::size_t bflen = 256, sublen = 0;
    cl::Context *context;
    cl::Device *device;
    cl::CommandQueue queue;

    managed_set_clbuf_t(cl::Context *context, cl::Device *device, const std::array<cl_mem_flags, N> flags) : flags(flags), context(context), device(device) {
        queue = cl::CommandQueue(*context, *device);
    }

    void ensure_init() {
        if (!init) {
            for (std::size_t i = 0; i < N; i++) {
                buffers[i] = cl::Buffer(*context, flags[i], bflen * sizeof(T));
            }
            init = true;
        }
    }

    cl::Buffer get_sub(std::size_t n) {
        if (n >= buffers.size()) {
            std::cerr << "fatal: managed_set_clbuf: out of bounds, attempted to access buffer n = " << n << " > size = " << buffers.size() << '\n';
            std::exit(1);
        }
        ensure_init();
        cl_buffer_region region{0, sublen}; /* should we return the actual buf if sublen = bflen? */
        return buffers[n].createSubBuffer(flags[n], CL_BUFFER_CREATE_TYPE_REGION, &region);
    }

    /* always blocks, and overwrites out */
    void read_sub(std::size_t n, std::vector<T> &out) {
        if (n >= buffers.size()) {
            std::cerr << "fatal: managed_set_clbuf: out of bounds, attempted to access buffer n = " << n << " > size = " << buffers.size() << '\n';
            std::exit(1);
        }
        ensure_init();
        out.resize(sublen);
        queue.enqueueReadBuffer(buffers[n], CL_TRUE, 0, sublen * sizeof(T), out.data());
    }

    void resize(std::size_t newsublen) {
        ensure_init();
        if (newsublen > bflen) {
            bflen = static_cast<std::size_t>(std::round(std::exp2(static_cast<std::size_t>(std::ceil(std::log2(newsublen)))))); /* next 2^n */
            for (std::size_t i = 0; i < N; i++) {
                cl::Buffer b(*context, flags[i], bflen * sizeof(T));
                queue.enqueueCopyBuffer(buffers[i], b, 0, 0, sublen * sizeof(T));
                queue.flush();
                buffers[i] = b;
            }
        } else if (newsublen < bflen/8 && bflen > 512) { /* TODO: change? 1/8 is arbitrary. > 512 check is because for small sizes it's not worth to reallocate */
            bflen = bflen/8;
            for (std::size_t i = 0; i < N; i++) {
                cl::Buffer b(*context, flags[i], bflen * sizeof(T));
                queue.enqueueCopyBuffer(buffers[i], b, 0, 0, bflen * sizeof(T));
                queue.flush();
                buffers[i] = b;
            }
        }
        sublen = newsublen;
    }

    /* always blocks */
    void set(std::size_t n, const std::vector<T> &v) {
        if (n >= buffers.size()) {
            std::cerr << "fatal: managed_set_clbuf: out of bounds, attempted to access buffer n = " << n << " > size = " << buffers.size() << '\n';
            std::exit(1);
        }
        resize(v.size());
        queue.enqueueWriteBuffer(buffers[n], CL_TRUE, 0, sublen * sizeof(T), v.data());
    }
};


/* whatever is produced by tracks: midi or audio */
struct content_t {
    virtual ~content_t() = default;
    virtual content_t *copy() const = 0;
};

struct midi_note_t {
    std::uint32_t note = 60;
    rational_t begin, end; /* fractions of 1 beat */
    float balance = 0, velocity = 0.5;

    static float to_hz(std::uint32_t note) {
        return 440 * std::exp2f(static_cast<float>(note-69) / 12.0f);
    }

    static std::uint32_t from_hz(float hz) {
        return static_cast<std::uint32_t>(std::round(std::log2f(hz/440.0f) * 12.0f + 69));
    }
};

/* midi content */
struct midi_t : virtual content_t {
    std::uint64_t bpm;
    std::vector<midi_note_t> notes;

    explicit midi_t(std::uint64_t bpm);
    midi_t(std::uint64_t bpm, std::vector<midi_note_t> notes);

    content_t *copy() const override;

    void sort_by_begin() {
        std::sort(notes.begin(), notes.end(), [&](const midi_note_t &a, const midi_note_t &b) { return a.begin.less(b.begin); });
    }
};

/* audio content */
struct audio_t : virtual content_t {
    std::uint64_t sample_rate;
    std::vector<cl_float2> samples;

    explicit audio_t(std::uint64_t sample_rate);
    audio_t(std::uint64_t sample_rate, std::vector<cl_float2> samples);

    content_t *copy() const override;
};

struct comp_t;

struct param_tag_t {
    comp_t* parent;
    std::uint32_t param_id;
};

using proc_output_cache_t = std::unordered_map<comp_t*, std::unique_ptr<content_t>>;

using deriv_fn_t = std::function<float (const proc_output_cache_t& /* cache */, const std::vector<comp_t*>& /* chain */, std::uint32_t /* self_ind */, std::uint64_t /* sample */, const param_tag_t& /* tag */)>;

float estimate_deriv(const proc_output_cache_t &cache, const std::vector<comp_t*> &chain, std::uint32_t self_ind, std::uint64_t sample, const param_tag_t &tag);

float default_zero_deriv(const proc_output_cache_t &cache, const std::vector<comp_t*> &chain, std::uint32_t self_ind, std::uint64_t sample, const param_tag_t &tag);

/* component */
struct comp_t {
    bool midi_out = false;
    std::uint64_t sample_rate;
    deriv_fn_t proc_deriv;
    virtual ~comp_t() = default;
    virtual std::uint64_t ind_latency(std::uint64_t samples, std::uint64_t bufstart) const = 0;
    /* does not handle allocation of out, expects correct out type
     * also note that midi components need to convert from sampout to time using sample_rate
     */
    virtual bool proc(const content_t *in, std::uint64_t sampout, std::uint64_t bufstart, content_t *out) = 0;

    void set_sample_rate(std::uint64_t newsample_rate);
};

struct comp_cplx_t : virtual comp_t {
    std::vector<std::vector<comp_t>> child_groups;
};

template <typename ...Ts>
struct comp_cl_t : virtual comp_t {
    cl::Context *context = nullptr;
    cl::Device *device = nullptr;
    cl::CommandQueue queue;
    cl::KernelFunctor<Ts...> f;
    managed_set_clbuf_t<cl_float2, 2> bufs; /* in, out */
    managed_set_clbuf_t<cl_float2, 2> reserve_bufs; /* in, out | for leftovers */

    comp_cl_t(cl::Context *context, cl::Device *device, cl::KernelFunctor<Ts...> f, const std::array<cl_mem_flags, 2> flags) : context(context), device(device), f(std::move(f)), bufs(context, device, flags), reserve_bufs(context, device, flags) {
        queue = cl::CommandQueue(*context, *device);
    }

    /* assumes first args are buf in, buf out, sample_rate, bufstart, groupwidth */
    template <typename ...Args>
    void run_with_args(const audio_t *a, audio_t *b, std::uint64_t sampout, std::uint64_t catprev, std::uint64_t bufstart, std::uint64_t groupcount, Args... args) {
        std::uint64_t adjsampout = sampout - (sampout % groupcount);
        bufs.set(0, a->samples);
        bufs.resize(adjsampout + catprev); /* cuts off some */
        std::size_t groupwidth = adjsampout/groupcount;
        cl::NDRange global(groupcount);
        f(cl::EnqueueArgs(queue, global), bufs.get_sub(0), bufs.get_sub(1), sample_rate, bufstart, groupwidth, args...).wait();
        bufs.read_sub(1, b->samples); /* b->samples.size() is adjsampout + catprev after this */
        if (sampout == adjsampout) {
            return;
        }
        /* WARNING BELOW DOES NOT RESPECT CATPREV */
        std::uint64_t leftcount = sampout - adjsampout;
        std::vector<cl_float2> leftover(leftcount);
        std::copy(a->samples.begin() + static_cast<std::int64_t>(adjsampout), a->samples.end(), leftover.begin());
        reserve_bufs.set(0, leftover);
        cl::NDRange reserve_global(1);
        f(cl::EnqueueArgs(queue, reserve_global), reserve_bufs.get_sub(0), reserve_bufs.get_sub(1), sample_rate, bufstart + adjsampout, leftcount, args...).wait();
        reserve_bufs.read_sub(1, leftover);
        b->samples.resize(sampout);
        std::copy(leftover.begin(), leftover.end(), b->samples.end() - static_cast<std::int64_t>(leftcount));
    }

    /* assumes first args are buf in, buf out, sample_rate, bufstart, bpm, groupwidth */
    /* template <typename ...Args>
    void run_with_args(const midi_t *m, audio_t *b, std::uint64_t sampout, std::uint64_t bufstart, std::uint64_t bpm, std::uint64_t groupcount, Args... args) {
        std::size_t groupwidth = adjsampout/groupcount;
        cl::NDRange global(groupcount);
        f(cl::EnqueueArgs(queue, global), bufs.get_sub(0), bufs.get_sub(1), sample_rate, bufstart, groupwidth, args...).wait(); */
        // bufs.read_sub(1, b->samples); /* b->samples.size() is adjsampout after this */
        /* if (sampout == adjsampout) {
            return;
        }
        std::uint64_t leftcount = sampout - adjsampout;
        std::vector<cl_float2> leftover(leftcount);
        std::copy(a->samples.begin() + static_cast<std::int64_t>(adjsampout), a->samples.end(), leftover.begin());
        reserve_bufs.set(0, leftover);
        cl::NDRange reserve_global(1);
        f(cl::EnqueueArgs(queue, reserve_global), reserve_bufs.get_sub(0), reserve_bufs.get_sub(1), sample_rate, bufstart + adjsampout, leftcount, args...).wait();
        reserve_bufs.read_sub(1, leftover);
        b->samples.resize(sampout);
        std::copy(leftover.begin(), leftover.end(), b->samples.end() - static_cast<std::int64_t>(leftcount)); */
    // }

    /* assumes first args are buf in, buf out, sample_rate, bufstart, sampout */
    template <typename ...Args>
    void run_single_with_args(const audio_t *a, audio_t *b, std::uint64_t sampout, std::uint64_t bufstart, Args... args) {
        bufs.set(0, a->samples);
        cl::NDRange global(1);
        f(cl::EnqueueArgs(queue, global), bufs.get_sub(0), bufs.get_sub(1), sample_rate, bufstart, sampout, args...).wait();
        bufs.read_sub(1, b->samples);
    }

};
