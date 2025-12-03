#include <atomic>
#include <iostream>
#include <iomanip>
#include <fstream>
#include <numeric>
#include <span>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <cmath>
#include <cstdint>

#include <CL/opencl.hpp>
#include <clFFT.h>
#include <pulse/pulseaudio.h>


cl::Program program; /* NOLINT */

struct rational_t {
    std::uint64_t num = 1, den = 1;
    bool pos = true;

    rational_t(std::int64_t numerator, std::int64_t denominator) : num(std::abs(numerator)), den(std::abs(denominator)),
        pos(((numerator >= 0) && (denominator >= 0)) || ((numerator < 0) && (denominator < 0))) {}

    rational_t(std::uint64_t numerator, std::uint64_t denominator) : num(numerator), den(denominator) {}

    rational_t(float a, std::uint64_t denominator) : num(static_cast<std::uint64_t>(std::round(std::abs(a)*denominator))), den(denominator), pos(!std::signbit(a)) {} /* NOLINT */

    template <typename T>
    T cast() const {
        return (2*pos - 1) * static_cast<T>(num) / static_cast<T>(den);
    }

    /* simplifiy */
    rational_t &smp() {
        const std::uint64_t g = std::gcd(num, den); 
        num /= g;
        den /= g;
        return *this;
    }

    rational_t &neg() {
        pos = !pos;
        return *this;
    }

    rational_t &add(const rational_t &other) {
        std::uint64_t l = std::lcm(den, other.den);
        num = l/den*num + l/other.den * other.num;
        den = l;
        return *this;
    }

    rational_t &sub(const rational_t &other) {
        std::uint64_t l = std::lcm(den, other.den);
        num = l/den*num - l/other.den * other.num;
        den = l;
        return *this;
    }

    rational_t &inv() {
        std::uint64_t tmp = num;
        num = den;
        den = tmp;
        return *this;
    }

    rational_t &mul(const rational_t &other) {
        num *= other.num;
        den *= other.den;
        return *this;
    }

    rational_t &div(const rational_t &other) {
        num *= other.den;
        den *= other.num;
        return *this;
    }
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

std::size_t next_exp_2(std::size_t n) {
    return static_cast<std::size_t>(std::exp2(std::ceil(std::log2(n))));
}

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

    explicit midi_t(std::uint64_t bpm) : bpm(bpm), notes() {}
    midi_t(std::uint64_t bpm, std::vector<midi_note_t> notes) : bpm(bpm), notes(std::move(notes)) {}

    content_t *copy() const override {
        auto *m = new midi_t(bpm, notes);
        return m;
    }
};

/* audio content */
struct audio_t : virtual content_t {
    std::uint64_t bitrate;
    std::vector<cl_float2> samples;

    explicit audio_t(std::uint64_t bitrate) : bitrate(bitrate), samples() {}
    audio_t(std::uint64_t bitrate, std::vector<cl_float2> samples) : bitrate(bitrate), samples(std::move(samples)) {}

    content_t *copy() const override {
        auto *a = new audio_t(bitrate, samples);
        return a;
    }
};

/* component */
struct comp_t {
    bool midi_out = false;
    std::uint64_t bitrate;
    virtual ~comp_t() = default;
    virtual std::uint64_t ind_latency(std::uint64_t samples, std::uint64_t bufstart) const = 0;
    /* does not handle allocation of out, expects correct out type
     * also note that midi components need to convert from sampout to time using bitrate
     */
    virtual bool proc(const content_t *in, std::uint64_t sampout, std::uint64_t bufstart, content_t *out) = 0;

    void set_bitrate(std::uint64_t newbitrate) {
        bitrate = newbitrate;
    }

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

    /* assumes first args are buf in, buf out, bitrate, bufstart */
    template <typename ...Args>
    void run_with_args(const audio_t *a, audio_t *b, std::uint64_t sampout, std::uint64_t bufstart, std::uint64_t adjn, Args... args) {
        std::uint64_t adjsampout = sampout - (sampout % adjn);
        bufs.set(0, a->samples);
        bufs.resize(adjsampout); /* cuts off some */
        std::size_t perL = adjsampout/adjn;
        cl::NDRange global(adjsampout);
        cl::NDRange local(perL);
        f(cl::EnqueueArgs(queue, global, local), bufs.get_sub(0), bufs.get_sub(1), bitrate, bufstart, args...).wait();
        bufs.read_sub(1, b->samples); /* b->samples.size() is adjsampout after this */
        if (sampout == adjsampout) {
            return;
        }
        std::uint64_t leftcount = sampout - adjsampout;
        std::vector<cl_float2> leftover(leftcount);
        std::copy(a->samples.begin() + static_cast<std::int64_t>(adjsampout), a->samples.end(), leftover.begin());
        reserve_bufs.set(0, leftover);
        cl::NDRange reserve_global(leftcount);
        cl::NDRange reserve_local(leftcount);
        f(cl::EnqueueArgs(queue, reserve_global, reserve_local), reserve_bufs.get_sub(0), reserve_bufs.get_sub(1), bitrate, bufstart + adjsampout, args...).wait();
        reserve_bufs.read_sub(1, leftover);
        b->samples.resize(sampout);
        std::copy(leftover.begin(), leftover.end(), b->samples.end() - static_cast<std::int64_t>(leftcount));
    }
};

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
    std::uint64_t bitrate, bpm;
    std::vector<std::unique_ptr<comp_t>> components;
    std::uint64_t id = get_new_track_id();
    std::vector<track_id_t> sends;


    void set_bitrate(std::uint64_t newbitrate) {
        bitrate = newbitrate;
        for (std::unique_ptr<comp_t> &comp : components) {
            comp->set_bitrate(newbitrate);
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
            b = new audio_t(bitrate);
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


/* 0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0-0 */
/* built-in components */

/* utility component */
struct comp_utility_t : virtual comp_cl_t<cl::Buffer, cl::Buffer, cl_ulong, cl_ulong, cl_float, cl_float, cl_uint, cl_uint, cl_uint, cl_float> {
    /* functor: first 2 params are LR input audio, last 2 are LR out, and rest are settings below */
    cl_float gain = 0; /* in db */
    cl_float balance = 0; /* -1 is full left, 1 is full right */
    bool muted = false, mute_l = false, mute_r = false;
    cl_float width = 0; /* 0 for mono, > 1 for widened */

    comp_utility_t(cl::Context *context, cl::Device *device) : comp_cl_t(context, device, cl::Kernel(program, "comp_utility"), {CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY}) {
        midi_out = false;
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
        run_with_args(a, b, sampout, bufstart, 16, gain, balance, static_cast<cl_uint>(muted), static_cast<cl_uint>(mute_l), static_cast<cl_uint>(mute_r), width);
        return true;
    }
};


struct comp_synth_t : virtual comp_cl_t<cl::Buffer, cl::Buffer, cl_ulong, cl_ulong, cl_uint, cl_float, cl_float> {
    cl_uint tablesel = 0;
    cl_float hz = 1;
    cl_float shape = 0;

    comp_synth_t(cl::Context *context, cl::Device *device) : comp_cl_t(context, device, cl::Kernel(program, "comp_synth"), {CL_MEM_READ_WRITE, CL_MEM_WRITE_ONLY}) {
        midi_out = false;
    }

    std::uint64_t ind_latency(std::uint64_t samples, std::uint64_t bufstart) const override {
        return 0;
    }

    bool proc(const content_t * /*in*/ /* should be midi! and use it */, std::uint64_t sampout, std::uint64_t bufstart, content_t* out) override {
        auto *b = dynamic_cast<audio_t*>(out);
        if (b == nullptr) {
            return false;
        }
        std::vector<cl_float2> zeros(sampout);
        auto *a = new audio_t(bitrate, zeros);
        run_with_args(a, b, sampout, bufstart, 16, tablesel, hz, shape);
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
        if (underflows >= 2 && latency < 2000000) {
            latency = (latency*3)/2;
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
};

/* HAVE TO USE BECAUSE OF C ENUMS */
template <typename T>
T int_to_flags(int i) {
    return *reinterpret_cast<T*>(&i); /* NOLINT */
}

const std::uint32_t maxu32 = std::numeric_limits<std::uint32_t>::max();

int main() {
    /* pulseaudio setup */
    std::uint64_t globalbitrate = 44100;
    pa_inst::source = nullptr;
    pa_inst::sampleoffs = 0;
    pa_inst::latency = 10000; /* in us */
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
    
    pa_inst::ss.rate = globalbitrate;
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
    const clfftStatus ffts = clfftSetup(nullptr);
    if (ffts != CLFFT_SUCCESS) {
        std::cerr << "fatal: could not initialize clFFT: error code " << ffts << '\n';
        return 1;
    }

    cl::Program::Sources sources;
    std::string programstr = get_file_content("program.cl");
    sources.emplace_back(programstr.c_str(), programstr.length());
    
    program = cl::Program(context, sources);

    if (program.build({default_device}) != CL_SUCCESS) {
        std::cerr << "fatal: could not build pocl:\n\n" << program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(default_device) << '\n';
        return 1;
    }

    cl::CommandQueue queue(context, default_device);

    /* clfftPlanHandle clfplan = 0;
    const size_t len[1] = {10};
    clfftCreateDefaultPlan(&clfplan, context.get(), clfftDim::CLFFT_1D, len);
    clfftSetLayout(clfplan, CLFFT_REAL, CLFFT_HERMITIAN_INTERLEAVED);
    constexpr std::uint32_t nq = 1;
    cl_command_queue queues[nq] = {queue.get()};
    clfftBakePlan(clfplan, nq, queues, nullptr, nullptr);
    clfftEnqueueTransform(clfplan, CLFFT_FORWARD, nq, queues, 0, nullptr,  */




    track_t main;
    auto wave = std::make_unique<comp_synth_t>(&context, &default_device);
    wave->tablesel = 1;
    wave->hz = 240;
    wave->shape = 0.5;
    main.components.push_back(std::move(wave));
    auto ut = std::make_unique<comp_utility_t>(&context, &default_device);
    ut->gain = -12;
    main.components.push_back(std::move(ut));
    main.set_bitrate(globalbitrate);
    pa_inst::source = &main;

    int retval = 0;
    pa_mainloop_run(pa_inst::ml, &retval);

    std::string s;
    std::cin >> s;

    pa_mainloop_quit(pa_inst::ml, 0);
    pa_inst::destroy();

    return 0;
}
