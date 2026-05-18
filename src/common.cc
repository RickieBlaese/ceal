#include "common.h"

rational_t::rational_t(std::int64_t numerator, std::int64_t denominator) : num(std::abs(numerator)), den(std::abs(denominator)),
    pos(((numerator >= 0) && (denominator >= 0)) || ((numerator < 0) && (denominator < 0))) {}

rational_t::rational_t(std::uint64_t numerator, std::uint64_t denominator) : num(numerator), den(denominator) {}

rational_t::rational_t(float a, std::uint64_t denominator) : num(static_cast<std::uint64_t>(std::round(std::abs(a)*denominator))), den(denominator), pos(!std::signbit(a)) {} /* NOLINT */


/* simplifiy */
rational_t &rational_t::smp() {
    const std::uint64_t g = std::gcd(num, den); 
    num /= g;
    den /= g;
    return *this;
}

rational_t &rational_t::neg() {
    pos = !pos;
    return *this;
}

rational_t &rational_t::add(const rational_t &other) {
    std::uint64_t l = std::lcm(den, other.den);
    num = l/den*num + l/other.den * other.num;
    den = l;
    return *this;
}

rational_t &rational_t::sub(const rational_t &other) {
    std::uint64_t l = std::lcm(den, other.den);
    num = l/den*num - l/other.den * other.num;
    den = l;
    return *this;
}

rational_t &rational_t::inv() {
    std::uint64_t tmp = num;
    num = den;
    den = tmp;
    return *this;
}

rational_t &rational_t::mul(const rational_t &other) {
    num *= other.num;
    den *= other.den;
    return *this;
}

rational_t &rational_t::div(const rational_t &other) {
    num *= other.den;
    den *= other.num;
    return *this;
}

bool rational_t::less(const rational_t &other) const {
    return num*other.den < den*other.num;
}

std::size_t next_exp_2(std::size_t n) {
    return static_cast<std::size_t>(std::exp2(std::ceil(std::log2(n))));
}


midi_t::midi_t(std::uint64_t bpm) : bpm(bpm), notes() {}
midi_t::midi_t(std::uint64_t bpm, std::vector<midi_note_t> notes) : bpm(bpm), notes(std::move(notes)) {}

content_t *midi_t::copy() const {
    auto *m = new midi_t(bpm, notes);
    return m;
}

audio_t::audio_t(std::uint64_t sample_rate) : sample_rate(sample_rate), samples() {}
audio_t::audio_t(std::uint64_t sample_rate, std::vector<cl_float2> samples) : sample_rate(sample_rate), samples(std::move(samples)) {}

content_t *audio_t::copy() const {
    auto *a = new audio_t(sample_rate, samples);
    return a;
}


void comp_t::set_sample_rate(std::uint64_t newsamplerate) {
    sample_rate = newsamplerate;
}

float estimate_deriv(const proc_output_cache_t &cache, const std::vector<comp_t*> &chain, std::uint32_t self_ind, std::uint64_t sample, const param_tag_t &tag) {
    return 0;
}

float default_zero_deriv(const proc_output_cache_t &cache, const std::vector<comp_t*> &chain, std::uint32_t self_ind, std::uint64_t sample, const param_tag_t &tag) {
    return 0;
}
