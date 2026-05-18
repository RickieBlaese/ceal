#include <algorithm>
#include <atomic>
#include <iomanip>
#include <fstream>
#include <optional>
#include <span>
#include <sstream>
#include <iostream>
#include <string>
#include <utility>

#include <pulse/pulseaudio.h>

#include "common.h"
#include "diffrender.h"

cl::Program program; /* NOLINT */



using track_id_t = std::uint64_t;

track_id_t get_new_track_id() {
    static track_id_t count = 0;
    return count++;
}

/* ideally, whole track is on the gpu as one opencl kernel since organization of it shouldn't change much.
 * however, unsure how slow recompiling is going to be / how much we want to assume it will be changed on average
 * need recompiling to be on the order of < 50ms
 * or we can develop two forms of freeze: a kernel freeze which compiles everything together in 1 kernel, and then a full freeze which renders it completely
 * that way, we offer either and don't have to give guarantees about how fast the kernel freeze would be */
/* 1 sample is a cl_float2 : both L and R channels in one sample */
struct track_t {
    std::uint64_t sample_rate, bpm;
    std::vector<std::unique_ptr<comp_t>> components;
    std::uint64_t id = get_new_track_id();
    std::vector<track_id_t> sends;


    void set_sample_rate(std::uint64_t newsamplerate) {
        sample_rate = newsamplerate;
        for (std::unique_ptr<comp_t> &comp : components) {
            comp->set_sample_rate(newsamplerate);
        }
    }

    /* handles allocation of out, ignores initial value of out.
     * note that bflen is in samples */
    bool get(std::uint64_t bflen, std::uint64_t bufstart, content_t *&out) {
        content_t *empty = new audio_t(bflen);
        content_t *a = empty, *b = nullptr, *c = nullptr;
        /* find needed buf length */
        for (const std::unique_ptr<comp_t> &comp : components) {
            bflen += comp->ind_latency(bflen, bufstart);
        }
        bool last_midi_out = false;
        for (std::unique_ptr<comp_t> &comp : components) {
            b = new audio_t(sample_rate);
            c = new midi_t(bpm);
            bool res = false;
            if (!comp->midi_out) {
                res = comp->proc(a, bflen, bufstart, b);
                a = b;
            } else {
                res = comp->proc(a, bflen, bufstart, c);
                a = c;
            }
            if (!res) {
                return false;
            }
            last_midi_out = comp->midi_out;
        }
        if (last_midi_out) {
            out = c;
        } else {
            out = b;
        }
        return true;
    }
};


template <typename T, typename ...Args>
struct param_controller_t {
    std::function<T (float, Args...)> f;

    explicit param_controller_t(T constant) : f([constant](float, Args...) { return constant; }) {}

    param_controller_t() = default;

    void set_constant(const T &constant) {
        f = [constant](float, Args...) { return constant; };
    }

    T get(float t, Args... args) {
        return f(t, args...);
    }
};

/* 0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0 */
/* built-in components */

/* utility component */
float comp_utility_deriv(const proc_output_cache_t &cache, const std::vector<comp_t*> &chain, std::uint32_t self_ind, std::uint64_t sample, const param_tag_t &tag);

struct comp_utility_t : virtual comp_cl_t<cl::Buffer, cl::Buffer, cl_ulong, cl_ulong, cl_ulong, cl_float, cl_float, cl_uint, cl_uint, cl_uint, cl_float>  {
    /* functor: first 2 params are LR input audio, last 2 are LR out, and rest are settings below */
    cl_float gain = 0; /* in db */
    cl_float balance = 0; /* -1 is full left, 1 is full right */
    bool muted = false, mute_l = false, mute_r = false;
    cl_float width = 0; /* 0 for mono, > 1 for widened */

    comp_utility_t(cl::Context *context, cl::Device *device) : comp_cl_t(context, device, cl::Kernel(program, "comp_utility"), {CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY}) {
        midi_out = false;
        proc_deriv = comp_utility_deriv;
    }

    std::uint64_t ind_latency(std::uint64_t samples, std::uint64_t bufstart) const override {
        return 0;
    }

    bool proc(const content_t *in, std::uint64_t sampout, std::uint64_t bufstart, content_t *out) override {
        const auto *a = dynamic_cast<const audio_t*>(in);
        auto *b = dynamic_cast<audio_t*>(out);
        if (a == nullptr || b == nullptr) {
            return false;
        }
        /* BAD! */
        /* realloc(reinterpret_cast<void*>(inl.getInfo(CL_MEM_HOST_PTR, static_cast<void*>(nullptr))), sampout * sizeof(float));
        inl.get(); */
        run_with_args(a, b, sampout, 0, bufstart, 8, gain, balance, static_cast<cl_uint>(muted), static_cast<cl_uint>(mute_l), static_cast<cl_uint>(mute_r), width);
        return true;
    }
};

float comp_utility_deriv(const proc_output_cache_t &cache, const std::vector<comp_t*> &chain, std::uint32_t self_ind, std::uint64_t sample, const param_tag_t &tag) {
    auto self_ut = dynamic_cast<comp_utility_t*>(chain[self_ind]);
    if ((tag.parent == self_ut && tag.param_id == 0 /* gain id */) || self_ind == 0) {
        return self_ut->gain; /* *gain_control_deriv */
    } else if (self_ind > 0) {
        return self_ut->gain * chain[self_ind]->proc_deriv(cache, chain, self_ind-1, sample, tag);
    }
}


/* synth component */
struct comp_synth_t : virtual comp_cl_t<cl::Buffer, cl::Buffer, cl_ulong, cl_ulong, cl_ulong, cl_uint, cl_float, cl_float> {
    cl_uint tablesel = 0;
    cl_float hz = 1;
    cl_float shape = 0;

    comp_synth_t(cl::Context *context, cl::Device *device) : comp_cl_t(context, device, cl::Kernel(program, "comp_synth"), {CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY}) {
        midi_out = false;
    }

    std::uint64_t ind_latency(std::uint64_t samples, std::uint64_t bufstart) const override {
        return 0;
    }

    bool proc(const content_t *in /* should be midi!*/, std::uint64_t sampout, std::uint64_t bufstart, content_t* out) override {
        auto *m = dynamic_cast<const midi_t*>(in);
        auto *b = dynamic_cast<audio_t*>(out);
        if (b == nullptr) {
            return false;
        }
        std::vector<cl_float2> zeros(sampout);
        auto *a = new audio_t(sample_rate, zeros);
        run_with_args(a, b, sampout, 0, bufstart, 8, tablesel, hz, shape);
        return true;
    }
};


struct comp_noteroll_t : comp_t {
    midi_t midi;

    explicit comp_noteroll_t(std::uint64_t bpm) : midi(bpm) {
        midi_out = true;
    }

    std::uint64_t ind_latency(std::uint64_t samples, std::uint64_t bufstart) const override {
        return 0;
    }

    bool proc(const content_t *in /* expect empty audio */, std::uint64_t sampout, std::uint64_t bufstart, content_t *out) override {
        auto *a = dynamic_cast<const audio_t*>(in);
        auto *b = dynamic_cast<midi_t*>(out);
        if (a == nullptr || b == nullptr) {
            return false;
        }
        for (const midi_note_t &note : midi.notes) {
            auto nbegin = note.begin.cast<float>();
            auto nend = note.end.cast<float>();
            float rangebegin = bufstart/static_cast<float>(sample_rate)*midi.bpm; /* NOLINT */
            float rangeend = (bufstart+sampout)/static_cast<float>(sample_rate)*midi.bpm; /* NOLINT */
            if ((rangebegin <= nbegin && nbegin < rangeend) || (rangebegin <= nend && nend < rangeend)) {
                b->notes.push_back(note);
            }
        }
        b->sort_by_begin();
        return true;
    }
};

template <typename T>
void shift_catprevs(std::vector<T> &insamples, std::vector<T> &outsamples, const std::vector<T> &newinput, std::size_t sampout, std::uint32_t n) {
    assert(insamples.size() >= 2*n && outsamples.size() >= 2*n); /* pray bruh */
    assert(newinput.size() == sampout);
    std::vector<T> inprev(n), outprev(n);
    std::copy(insamples.end() - n, insamples.end(), inprev.begin());
    std::copy(outsamples.end() - n, outsamples.end(), inprev.begin());
    insamples.resize(sampout+n, {{0.0f, 0.0f}});
    outsamples.resize(sampout+n, {{0.0f, 0.0f}});
    std::copy(newinput.begin(), newinput.end(), insamples.begin());
    /* std::copy(b-.begin(), b-.end(), outsamples.begin()); */
    std::copy(inprev.begin(), inprev.end(), insamples.end() - n);
    std::copy(outprev.begin(), outprev.end(), outsamples.end() - n);
}


struct comp_lpf_t : virtual comp_cl_t<cl::Buffer, cl::Buffer, cl_ulong, cl_ulong, cl_ulong, cl::Buffer, cl::Buffer> {
    audio_t incatprev;
    audio_t outcatprev;

    param_controller_t<cl_float> f0;
    managed_set_clbuf_t<cl_float, 1> f0_send;
    std::vector<cl_float> f0_send_v;
    param_controller_t<cl_float> q;
    managed_set_clbuf_t<cl_float, 1> q_send;
    std::vector<cl_float> q_send_v;

    comp_lpf_t(cl::Context *context, cl::Device *device) : comp_cl_t(context, device, cl::Kernel(program, "comp_lpf"), {CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY}), incatprev(sample_rate), outcatprev(sample_rate), f0(400), f0_send(context, device, {CL_MEM_READ_ONLY}), q(0.707), q_send(context, device, {CL_MEM_READ_ONLY}) {
        incatprev.samples.resize(4, {{0.0f, 0.0f}});
        outcatprev.samples.resize(4, {{0.0f, 0.0f}});
        midi_out = false;
    }

    void set_sample_rate(std::uint64_t sample_rate) {
        incatprev.sample_rate = sample_rate;
        outcatprev.sample_rate = sample_rate;
    }

    std::uint64_t ind_latency(std::uint64_t samples, std::uint64_t bufstart) const override {
        return 0;
    }

    bool proc(const content_t *in, std::uint64_t sampout, std::uint64_t bufstart, content_t *out) override {
        const auto *a = dynamic_cast<const audio_t*>(in);
        auto *b = dynamic_cast<audio_t*>(out);
        if (a == nullptr || b == nullptr) {
            return false;
        }
        /* BAD! */
        /* realloc(reinterpret_cast<void*>(inl.getInfo(CL_MEM_HOST_PTR, static_cast<void*>(nullptr))), sampout * sizeof(float));
        inl.get(); */


        shift_catprevs(incatprev.samples, outcatprev.samples, a->samples, sampout, 2);
        f0_send_v.resize(sampout);
        for (std::uint32_t i = 0; i < sampout; i++) {
            f0_send_v[i] = f0.f((bufstart+i)/static_cast<float>(sample_rate)); /* NOLINT */
        }
        q_send_v.resize(sampout);
        for (std::uint32_t i = 0; i < sampout; i++) {
            q_send_v[i] = q.f((bufstart+i)/static_cast<float>(sample_rate)); /* NOLINT */
        }
        f0_send.set(0, f0_send_v);
        q_send.set(0, q_send_v);

        run_single_with_args(&incatprev, &outcatprev, sampout, bufstart, f0_send.get_sub(0), q_send.get_sub(0));
        b->samples.resize(sampout);
        std::copy(outcatprev.samples.begin(), outcatprev.samples.end() - 2, b->samples.begin());
        return true;
    }
};

struct comp_lpf2_t : virtual comp_cl_t<cl::Buffer, cl::Buffer, cl_ulong, cl_ulong, cl_ulong, cl_float, cl_float> {
    cl_float f0 = 2000;
    cl_float q = 0.707;
    audio_t incatprev;
    audio_t outcatprev;

    comp_lpf2_t(cl::Context *context, cl::Device *device) : comp_cl_t(context, device, cl::Kernel(program, "comp_lpf2"), {CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY}), incatprev(sample_rate), outcatprev(sample_rate) {
        incatprev.samples.resize(8, {{0.0f, 0.0f}});
        outcatprev.samples.resize(8, {{0.0f, 0.0f}});
        midi_out = false;
    }

    void set_sample_rate(std::uint64_t sample_rate) {
        incatprev.sample_rate = sample_rate;
        outcatprev.sample_rate = sample_rate;
    }

    std::uint64_t ind_latency(std::uint64_t samples, std::uint64_t bufstart) const override {
        return 0;
    }

    bool proc(const content_t *in, std::uint64_t sampout, std::uint64_t bufstart, content_t *out) override {
        const auto *a = dynamic_cast<const audio_t*>(in);
        auto *b = dynamic_cast<audio_t*>(out);
        if (a == nullptr || b == nullptr) {
            return false;
        }

        shift_catprevs(incatprev.samples, outcatprev.samples, a->samples, sampout, 4);
        run_with_args(&incatprev, &outcatprev, sampout, 4, bufstart, 2, f0, q);
        b->samples.resize(sampout);
        std::copy(outcatprev.samples.begin(), outcatprev.samples.end() - 4, b->samples.begin());
        return true;
    }
};

struct comp_hpf_t : virtual comp_cl_t<cl::Buffer, cl::Buffer, cl_ulong, cl_ulong, cl_ulong, cl_float, cl_float> {
    cl_float f0 = 2000;
    cl_float q = 0.707;
    audio_t incatprev;
    audio_t outcatprev;

    comp_hpf_t(cl::Context *context, cl::Device *device) : comp_cl_t(context, device, cl::Kernel(program, "comp_hpf"), {CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY}), incatprev(sample_rate), outcatprev(sample_rate) {
        incatprev.samples.resize(4, {{0.0f, 0.0f}});
        outcatprev.samples.resize(4, {{0.0f, 0.0f}});
        midi_out = false;
    }

    void set_sample_rate(std::uint64_t sample_rate) {
        incatprev.sample_rate = sample_rate;
        outcatprev.sample_rate = sample_rate;
    }

    std::uint64_t ind_latency(std::uint64_t samples, std::uint64_t bufstart) const override {
        return 0;
    }

    bool proc(const content_t *in, std::uint64_t sampout, std::uint64_t bufstart, content_t *out) override {
        const auto *a = dynamic_cast<const audio_t*>(in);
        auto *b = dynamic_cast<audio_t*>(out);
        if (a == nullptr || b == nullptr) {
            return false;
        }

        shift_catprevs(incatprev.samples, outcatprev.samples, a->samples, sampout, 2);
        run_single_with_args(&incatprev, &outcatprev, sampout, bufstart, f0, q);
        b->samples.resize(sampout);
        std::copy(outcatprev.samples.begin(), outcatprev.samples.end() - 2, b->samples.begin());
        return true;
    }
};

struct comp_notchf_t : virtual comp_cl_t<cl::Buffer, cl::Buffer, cl_ulong, cl_ulong, cl_ulong, cl_float, cl_float> {
    cl_float f0 = 1000;
    cl_float q = 0.707;
    audio_t incatprev;
    audio_t outcatprev;

    comp_notchf_t(cl::Context *context, cl::Device *device) : comp_cl_t(context, device, cl::Kernel(program, "comp_notchf"), {CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY}), incatprev(sample_rate), outcatprev(sample_rate) {
        incatprev.samples.resize(4, {{0.0f, 0.0f}});
        outcatprev.samples.resize(4, {{0.0f, 0.0f}});
        midi_out = false;
    }

    void set_sample_rate(std::uint64_t sample_rate) {
        incatprev.sample_rate = sample_rate;
        outcatprev.sample_rate = sample_rate;
    }

    std::uint64_t ind_latency(std::uint64_t samples, std::uint64_t bufstart) const override {
        return 0;
    }

    bool proc(const content_t *in, std::uint64_t sampout, std::uint64_t bufstart, content_t *out) override {
        const auto *a = dynamic_cast<const audio_t*>(in);
        auto *b = dynamic_cast<audio_t*>(out);
        if (a == nullptr || b == nullptr) {
            return false;
        }

        shift_catprevs(incatprev.samples, outcatprev.samples, a->samples, sampout, 2);
        run_single_with_args(&incatprev, &outcatprev, sampout, bufstart, f0, q);
        b->samples.resize(sampout);
        std::copy(outcatprev.samples.begin(), outcatprev.samples.end() - 2, b->samples.begin());
        return true;
    }
};

struct comp_apf_t : virtual comp_cl_t<cl::Buffer, cl::Buffer, cl_ulong, cl_ulong, cl_ulong, cl_float, cl_float> {
    cl_float f0 = 1000;
    cl_float q = 0.707;
    audio_t incatprev;
    audio_t outcatprev;

    comp_apf_t(cl::Context *context, cl::Device *device) : comp_cl_t(context, device, cl::Kernel(program, "comp_apf"), {CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY}), incatprev(sample_rate), outcatprev(sample_rate) {
        incatprev.samples.resize(4, {{0.0f, 0.0f}});
        outcatprev.samples.resize(4, {{0.0f, 0.0f}});
        midi_out = false;
    }

    void set_sample_rate(std::uint64_t sample_rate) {
        incatprev.sample_rate = sample_rate;
        outcatprev.sample_rate = sample_rate;
    }

    std::uint64_t ind_latency(std::uint64_t samples, std::uint64_t bufstart) const override {
        return 0;
    }

    bool proc(const content_t *in, std::uint64_t sampout, std::uint64_t bufstart, content_t *out) override {
        const auto *a = dynamic_cast<const audio_t*>(in);
        auto *b = dynamic_cast<audio_t*>(out);
        if (a == nullptr || b == nullptr) {
            return false;
        }

        shift_catprevs(incatprev.samples, outcatprev.samples, a->samples, sampout, 2);
        run_single_with_args(&incatprev, &outcatprev, sampout, bufstart, f0, q);
        b->samples.resize(sampout);
        std::copy(outcatprev.samples.begin(), outcatprev.samples.end() - 2, b->samples.begin());
        return true;
    }
};

struct comp_bpf_t : virtual comp_cl_t<cl::Buffer, cl::Buffer, cl_ulong, cl_ulong, cl_ulong, cl_float, cl_float> {
    cl_float f0 = 1000;
    cl_float bw = 10;
    audio_t incatprev;
    audio_t outcatprev;

    comp_bpf_t(cl::Context *context, cl::Device *device) : comp_cl_t(context, device, cl::Kernel(program, "comp_bpf"), {CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY}), incatprev(sample_rate), outcatprev(sample_rate) {
        incatprev.samples.resize(4, {{0.0f, 0.0f}});
        outcatprev.samples.resize(4, {{0.0f, 0.0f}});
        midi_out = false;
    }

    void set_sample_rate(std::uint64_t sample_rate) {
        incatprev.sample_rate = sample_rate;
        outcatprev.sample_rate = sample_rate;
    }

    std::uint64_t ind_latency(std::uint64_t samples, std::uint64_t bufstart) const override {
        return 0;
    }

    bool proc(const content_t *in, std::uint64_t sampout, std::uint64_t bufstart, content_t *out) override {
        const auto *a = dynamic_cast<const audio_t*>(in);
        auto *b = dynamic_cast<audio_t*>(out);
        if (a == nullptr || b == nullptr) {
            return false;
        }

        shift_catprevs(incatprev.samples, outcatprev.samples, a->samples, sampout, 2);
        run_single_with_args(&incatprev, &outcatprev, sampout, bufstart, f0, bw);
        b->samples.resize(sampout);
        std::copy(outcatprev.samples.begin(), outcatprev.samples.end() - 2, b->samples.begin());
        return true;
    }
};


struct comp_peak_t : virtual comp_cl_t<cl::Buffer, cl::Buffer, cl_ulong, cl_ulong, cl_ulong, cl_float, cl_float, cl_float> {
    cl_float f0 = 1000;
    cl_float gain = 0;
    cl_float q = 0.707;
    audio_t incatprev;
    audio_t outcatprev;

    comp_peak_t(cl::Context *context, cl::Device *device) : comp_cl_t(context, device, cl::Kernel(program, "comp_peak"), {CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY}), incatprev(sample_rate), outcatprev(sample_rate) {
        incatprev.samples.resize(4, {{0.0f, 0.0f}});
        outcatprev.samples.resize(4, {{0.0f, 0.0f}});
        midi_out = false;
    }

    void set_sample_rate(std::uint64_t sample_rate) {
        incatprev.sample_rate = sample_rate;
        outcatprev.sample_rate = sample_rate;
    }

    std::uint64_t ind_latency(std::uint64_t samples, std::uint64_t bufstart) const override {
        return 0;
    }

    bool proc(const content_t *in, std::uint64_t sampout, std::uint64_t bufstart, content_t *out) override {
        const auto *a = dynamic_cast<const audio_t*>(in);
        auto *b = dynamic_cast<audio_t*>(out);
        if (a == nullptr || b == nullptr) {
            return false;
        }

        shift_catprevs(incatprev.samples, outcatprev.samples, a->samples, sampout, 2);
        run_single_with_args(&incatprev, &outcatprev, sampout, bufstart, f0, gain, q);
        b->samples.resize(sampout);
        std::copy(outcatprev.samples.begin(), outcatprev.samples.end() - 2, b->samples.begin());
        return true;
    }
};

struct comp_lowshelf_t : virtual comp_cl_t<cl::Buffer, cl::Buffer, cl_ulong, cl_ulong, cl_ulong, cl_float, cl_float, cl_float> {
    cl_float f0 = 400;
    cl_float gain = 0;
    cl_float slope = 1; /* idk */
    audio_t incatprev;
    audio_t outcatprev;

    comp_lowshelf_t(cl::Context *context, cl::Device *device) : comp_cl_t(context, device, cl::Kernel(program, "comp_lowshelf"), {CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY}), incatprev(sample_rate), outcatprev(sample_rate) {
        incatprev.samples.resize(4, {{0.0f, 0.0f}});
        outcatprev.samples.resize(4, {{0.0f, 0.0f}});
        midi_out = false;
    }

    void set_sample_rate(std::uint64_t sample_rate) {
        incatprev.sample_rate = sample_rate;
        outcatprev.sample_rate = sample_rate;
    }

    std::uint64_t ind_latency(std::uint64_t samples, std::uint64_t bufstart) const override {
        return 0;
    }

    bool proc(const content_t *in, std::uint64_t sampout, std::uint64_t bufstart, content_t *out) override {
        const auto *a = dynamic_cast<const audio_t*>(in);
        auto *b = dynamic_cast<audio_t*>(out);
        if (a == nullptr || b == nullptr) {
            return false;
        }

        shift_catprevs(incatprev.samples, outcatprev.samples, a->samples, sampout, 2);
        run_single_with_args(&incatprev, &outcatprev, sampout, bufstart, f0, gain, slope);
        b->samples.resize(sampout);
        std::copy(outcatprev.samples.begin(), outcatprev.samples.end() - 2, b->samples.begin());
        return true;
    }
};

struct comp_highshelf_t : virtual comp_cl_t<cl::Buffer, cl::Buffer, cl_ulong, cl_ulong, cl_ulong, cl_float, cl_float, cl_float> {
    cl_float f0 = 1000;
    cl_float gain = 0;
    cl_float slope = 1; /* idk */
    audio_t incatprev;
    audio_t outcatprev;

    comp_highshelf_t(cl::Context *context, cl::Device *device) : comp_cl_t(context, device, cl::Kernel(program, "comp_highshelf"), {CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY}), incatprev(sample_rate), outcatprev(sample_rate) {
        incatprev.samples.resize(4, {{0.0f, 0.0f}});
        outcatprev.samples.resize(4, {{0.0f, 0.0f}});
        midi_out = false;
    }

    void set_sample_rate(std::uint64_t sample_rate) {
        incatprev.sample_rate = sample_rate;
        outcatprev.sample_rate = sample_rate;
    }

    std::uint64_t ind_latency(std::uint64_t samples, std::uint64_t bufstart) const override {
        return 0;
    }

    bool proc(const content_t *in, std::uint64_t sampout, std::uint64_t bufstart, content_t *out) override {
        const auto *a = dynamic_cast<const audio_t*>(in);
        auto *b = dynamic_cast<audio_t*>(out);
        if (a == nullptr || b == nullptr) {
            return false;
        }

        shift_catprevs(incatprev.samples, outcatprev.samples, a->samples, sampout, 2);
        run_single_with_args(&incatprev, &outcatprev, sampout, bufstart, f0, gain, slope);
        b->samples.resize(sampout);
        std::copy(outcatprev.samples.begin(), outcatprev.samples.end() - 2, b->samples.begin());
        return true;
    }
};


std::string get_file_content(const std::string &filename) {
    std::ifstream file(filename);
    std::stringstream ss;
    ss << file.rdbuf();
    std::string text = ss.str();
    file.close();
    return text;
}

/* 0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0 */
/* pulseaudio */

std::mutex tracks_lock;

namespace pa_inst { /* basically a namespace */
    static std::int32_t latency; // start latency in micro seconds
    static std::int32_t sampleoffs;
    static pa_buffer_attr bufattr;
    static std::int32_t underflows;
    static pa_sample_spec ss;
    static pa_mainloop *ml;
    static pa_mainloop_api *mlapi;
    static pa_context *ctx;
    static pa_stream *playstream;
    static track_t *source;

    // This callback gets called when our context changes state.  We really only
    // care about when it's ready or if it has failed
    static void pa_state_cb(pa_context *c, void *userdata) {
        pa_context_state_t state{};
        int *pa_ready = reinterpret_cast<int*>(userdata); /* NOLINT */
        state = pa_context_get_state(c);
        switch  (state) { // These are just here for reference
            case PA_CONTEXT_UNCONNECTED:
            case PA_CONTEXT_CONNECTING:
            case PA_CONTEXT_AUTHORIZING:
            case PA_CONTEXT_SETTING_NAME:
            default:
                break;
            case PA_CONTEXT_FAILED:
            case PA_CONTEXT_TERMINATED:
                *pa_ready = 2;
                break;
            case PA_CONTEXT_READY:
                *pa_ready = 1;
                break;
        }
    }

    static void stream_request_cb(pa_stream *s, size_t nbytes, void *userdata) {
        static content_t *m = nullptr;
        static std::vector<float> af;
        static std::uint8_t counter = 0;
        if (counter % 8 == 0) {
            pa_usec_t usec;
            int neg;
            pa_stream_get_latency(s,&usec,&neg);
            std::cout << "INFO: latency " << std::setw(8) << usec << " us\n";
            counter = 0;
        }
        counter++;
        /* if (sampleoffs*2 + length > sizeof(sampledata)) {
            sampleoffs = 0;
        } */
        
        if (source == nullptr) {
            return;
        }
        std::size_t samplecount = nbytes/sizeof(float)/2;
        {
            std::lock_guard track(tracks_lock);
            source->get(samplecount, sampleoffs, m);
        }
        auto *a = dynamic_cast<audio_t*>(m);
        if (a == nullptr) {
            return;
        }
        af.resize(a->samples.size()*2);
        for (std::uint32_t i = 0; i < a->samples.size(); i++) {
            af[i*2] = a->samples[i].x;
            af[i*2+1] = a->samples[i].y;
        }
        pa_stream_write(s, af.data(), nbytes, NULL, 0LL, PA_SEEK_RELATIVE);
        sampleoffs += samplecount;
    }

    static void stream_underflow_cb(pa_stream *s, void *userdata) {
        // We increase the latency by 50% if we get 2 underflows and latency is under 2s
        // This is very useful for over the network playback that can't handle low latencies
        printf("underflow\n");
        underflows++;
        if (underflows >= 1 && latency < 2000000) {
            latency = latency*2;
            bufattr.maxlength = pa_usec_to_bytes(latency,&ss);
            bufattr.tlength = pa_usec_to_bytes(latency,&ss);  
            pa_stream_set_buffer_attr(s, &bufattr, NULL, NULL);
            underflows = 0;
            std::cout << "INFO: latency increased to " << latency << '\n';
        }
    }

    static void destroy() {
        pa_context_disconnect(ctx);
        pa_context_unref(ctx);
        pa_mainloop_free(ml);
    }

}  // namespace pa_inst

/* HAVE TO USE BECAUSE OF C ENUMS */
template <typename T>
T int_to_flags(int i) {
    return *reinterpret_cast<T*>(&i); /* NOLINT */
}

const std::uint32_t maxu32 = std::numeric_limits<std::uint32_t>::max();


void write_to_pcm(const std::vector<float> &samples, const std::string &filename) {
    std::ofstream ofile(filename, std::ios_base::binary);
    for (std::size_t i = 0; i < samples.size(); i++) {
        ofile.write(reinterpret_cast<const char*>(&samples[i]), sizeof(float)); /* NOLINT */
    }
    ofile.close();
}

int main() {
    /* pulseaudio setup */
    std::uint64_t globalsamplerate = 48000;
    pa_inst::source = nullptr;
    pa_inst::sampleoffs = 0;
    pa_inst::latency = 300000; /* in us */
    pa_inst::underflows = 0;
    std::atomic_int32_t paready = 0;
    int r = 0;
    std::string mediarole = "production";


    pa_inst::ml = pa_mainloop_new();
    pa_inst::mlapi = pa_mainloop_get_api(pa_inst::ml);
    pa_proplist *proplist = pa_proplist_new();
    pa_proplist_set(proplist, PA_PROP_MEDIA_ROLE, mediarole.c_str(), mediarole.size()+1);
    pa_inst::ctx = pa_context_new_with_proplist(pa_inst::mlapi, "ceal", proplist);
    pa_context_connect(pa_inst::ctx, NULL, PA_CONTEXT_NOFLAGS, NULL);

    // This function defines a callback so the server will tell us it's state.
    // Our callback will wait for the state to be ready.  The callback will
    // modify the variable to 1 so we know when we have a connection and it's
    // ready.
    // If there's an error, the callback will set paready to 2
    pa_context_set_state_callback(pa_inst::ctx, pa_inst::pa_state_cb, &paready);

    while (paready == 0) {
        pa_mainloop_iterate(pa_inst::ml, 1, NULL);
    }

    /* now we have a ready value */
    if (paready == 2) {
        pa_inst::destroy();
        return -1;
    }
    
    pa_inst::ss.rate = globalsamplerate;
    pa_inst::ss.channels = 2;
    pa_inst::ss.format = PA_SAMPLE_FLOAT32;
    pa_inst::playstream = pa_stream_new(pa_inst::ctx, "playback", &pa_inst::ss, NULL);
    if (!pa_inst::playstream) {
        std::cerr << "fatal: could not create pulseaudio play stream\n";
        return -1;
    }

    pa_stream_set_write_callback(pa_inst::playstream, pa_inst::stream_request_cb, NULL);
    pa_stream_set_underflow_callback(pa_inst::playstream, pa_inst::stream_underflow_cb, NULL);
    pa_inst::bufattr.fragsize = maxu32;
    pa_inst::bufattr.maxlength = pa_usec_to_bytes(pa_inst::latency, &pa_inst::ss);
    pa_inst::bufattr.minreq = pa_usec_to_bytes(0, &pa_inst::ss);
    pa_inst::bufattr.prebuf = maxu32;
    pa_inst::bufattr.tlength = pa_usec_to_bytes(pa_inst::latency, &pa_inst::ss);

    r = pa_stream_connect_playback(pa_inst::playstream, NULL, &pa_inst::bufattr,
        int_to_flags<pa_stream_flags>(PA_STREAM_INTERPOLATE_TIMING | PA_STREAM_AUTO_TIMING_UPDATE | PA_STREAM_ADJUST_LATENCY), NULL, NULL);

    if (r < 0) {
        /* sometimes PA_STREAM_ADJUST_LATENCY isn't supported on older pusle servers, so we try without */
        r = pa_stream_connect_playback(pa_inst::playstream, NULL, &pa_inst::bufattr,
            int_to_flags<pa_stream_flags>(PA_STREAM_INTERPOLATE_TIMING | PA_STREAM_AUTO_TIMING_UPDATE), NULL, NULL);
    }

    if (r < 0) {
        std::cerr << "fatal: could not connect stream for playback, pa_stream_connect_playback returned " << r << '\n';
        pa_inst::destroy();
        return -1;
    }



    std::vector<cl::Platform> platforms;
    cl::Platform::get(&platforms);
    if (platforms.empty()) {
        std::cerr << "fatal: no OpenCL platforms found\n";
        return 1;
    }
    for (const auto &p : platforms) {
        std::cout << "info: platform " << p.getInfo<CL_PLATFORM_NAME>() << '\n';
    }
    cl::Platform default_platform = platforms[0];
    std::cout << "info: using platform " << default_platform.getInfo<CL_PLATFORM_NAME>() << '\n';

    std::vector<cl::Device> devices;
    default_platform.getDevices(CL_DEVICE_TYPE_ALL, &devices);
    if (devices.empty()) {
        std::cerr << "fatal: no devices found on platform\n";
        return 1;
    }
    for (const auto &d : devices) {
        std::cout << "info: device " << d.getInfo<CL_DEVICE_NAME>() << '\n';
    }
    cl::Device default_device = devices[0];
    std::cout << "info: using device " << default_device.getInfo<CL_DEVICE_NAME>() << '\n';
    std::cout << "info: using OpenCL version " << default_device.getInfo<CL_DRIVER_VERSION>() << '\n';
    
    cl::Context context({default_device});
    /* clfftSetupData clfftsetupdata;
    clfftInitSetupData(&clfftsetupdata);
    const clfftStatus ffts = clfftSetup(&clfftsetupdata);
    if (ffts != CLFFT_SUCCESS) {
        std::cerr << "fatal: could not initialize clFFT: error code " << ffts << '\n';
        return 1;
    } */

    cl::Program::Sources sources;
    std::string programstr = get_file_content("program.cl");
    sources.emplace_back(programstr.c_str(), programstr.length());
    
    program = cl::Program(context, sources);

    if (program.build({default_device}) != CL_SUCCESS) {
        std::cerr << "fatal: could not build pocl:\n\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << '\n';
        return 1;
    }

    cl::CommandQueue queue(context, default_device);


#ifdef clfft
    std::ifstream wavin("in.pcm", std::ios_base::binary | std::ios_base::ate);
    size_t insz = wavin.tellg()/sizeof(float);
    std::vector<float> fftinput(insz);
    /* srand(time(nullptr));
    for (std::uint32_t i = 0; i < insz; i++) {
        fftinput[i] = 2.0f*std::clamp<float>(static_cast<float>(std::sin(2*std::numbers::pi*(static_cast<float>(i)/16.0f))) + 0.1f*static_cast<float>(rand())/RAND_MAX - 0.05f, -1.0, 1.0);
    } */
    wavin.seekg(0);
    wavin.read(reinterpret_cast<char*>(fftinput.data()), insz*sizeof(float)); /* NOLINT */



    clfftPlanHandle clfplan = 0;
    const size_t len[1] = {insz};
    clfftCreateDefaultPlan(&clfplan, context.get(), clfftDim::CLFFT_1D, len);
    clfftSetResultLocation(clfplan, CLFFT_OUTOFPLACE);
    clfftSetLayout(clfplan, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
    clfftSetPlanPrecision(clfplan, CLFFT_SINGLE);
    size_t instrides[] = {1};
    size_t outstrides[] = {1};
    clfftSetPlanInStride(clfplan, CLFFT_1D, instrides);
    clfftSetPlanOutStride(clfplan, CLFFT_1D, outstrides);

    constexpr std::uint32_t nq = 1;
    cl_command_queue queues[nq] = {queue.get()};
    clfftBakePlan(clfplan, nq, queues, nullptr, nullptr);


    cl::Buffer fftinbuf(context, CL_MEM_READ_WRITE | CL_MEM_USE_HOST_PTR, insz*sizeof(float), fftinput.data(), nullptr);
    cl::Buffer fftoutbuf(context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, (1+insz/2)*sizeof(float)*2, nullptr, nullptr);
    cl_mem fftinbufs[] = {fftinbuf.get()};
    cl_mem fftoutbufs[] = {fftoutbuf.get()};
    clfftStatus status = clfftEnqueueTransform(clfplan, CLFFT_FORWARD, nq, queues, 0, nullptr, nullptr, fftinbufs, fftoutbufs, nullptr);
    if (status != CLFFT_SUCCESS) {
        std::cerr << "fatal: could not enqueue transform: clfft error code " << status << '\n';
        return 1;
    }
    queue.finish(); /* wait for fft to complete */

    size_t outsz = 0;
    fftoutbuf.getInfo(CL_MEM_SIZE, &outsz);
    size_t outn = outsz/sizeof(float);
    std::vector<float> fftoutput(outn);
    queue.enqueueReadBuffer(fftoutbuf, true, 0, outsz, fftoutput.data());

    std::cout << "\noutn: " << outn << '\n';
    std::ofstream ofile("outfft.pcm", std::ios_base::binary);
    std::vector<float> mags(outn/2);
    float max = 0.0f;
    for (std::uint32_t i = 0; i < outn/2; i++) {
        float a = fftoutput[2*i], b = fftoutput[2*i + 1];
        mags[i] = std::sqrt(a*a + b*b);
        max = std::max(max, mags[i]);
    }
    std::cout << "max: " << max << '\n';
    if (max == 0.0f) {
        max = 1.0f;
    }
    for (std::uint32_t i = 0; i < mags.size(); i++) {
        float normmag = mags[i]/max;
        ofile.write(reinterpret_cast<char*>(&normmag), sizeof(float)); /* NOLINT */
    }
    ofile.close();

    clfftDestroyPlan(&clfplan);


#endif




    track_t main;
    auto wave = std::make_unique<comp_synth_t>(&context, &default_device);
    wave->tablesel = 1;
    wave->hz = 100.01;
    main.components.push_back(std::move(wave));
    auto ut = std::make_unique<comp_utility_t>(&context, &default_device);
    ut->gain = -6;
    main.components.push_back(std::move(ut));
    // auto lpf = std::make_unique<comp_lpf2_t>(&context, &default_device);
    auto lpf = std::make_unique<comp_lpf_t>(&context, &default_device);
    lpf->f0.f = [](float t) { return 50+5000*(sin(2*std::numbers::pi*5*t)/2 + 0.5f); };
    lpf->q.set_constant(2);
    main.components.push_back(std::move(lpf));

    main.set_sample_rate(globalsamplerate);

    content_t *out;
    main.get(globalsamplerate, 0, out);
    auto a = dynamic_cast<audio_t*>(out);
    std::vector<float> s(a->samples.size()*2);
    for (std::uint32_t i = 0; i < a->samples.size(); i++) {
        s[2*i] = a->samples[i].x;
        s[2*i+1] = a->samples[i].y;
    }
    write_to_pcm(s, "out.pcm");


    /* pa_inst::source = &main;

    int retval = 0;
    pa_mainloop_run(pa_inst::ml, &retval);

    while (true) {
        std::string s;
        std::cin >> s;
        if (s == "q") {
            break;
        }
    }

    pa_mainloop_quit(pa_inst::ml, 0);
    pa_inst::destroy(); */

    return 0;
}
