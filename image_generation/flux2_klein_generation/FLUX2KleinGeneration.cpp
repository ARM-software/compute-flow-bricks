//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#include "FLUX2KleinGeneration.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <random>
#include <string>

namespace compute_flow::bricks::image_generation {

namespace {

void fill_gaussian_noise(float* output, int64_t num_values, uint64_t seed) {
    constexpr double k_full_circle = 6.283185307179586476925286766559;
    std::mt19937_64 rng(seed);
    int64_t position = 0;

    while (position < num_values) {
        // Explicit uniform conversion and Box-Muller avoid implementation-defined distributions.
        // The survival probability lies in (0, 1], so log(0) cannot occur.
        const double survival  = 1.0 - static_cast<double>(rng() >> 11) * 0x1.0p-53;
        const double angle     = k_full_circle * (static_cast<double>(rng() >> 11) * 0x1.0p-53);
        const double amplitude = std::sqrt(-2.0 * std::log(survival));

        output[position++] = static_cast<float>(amplitude * std::cos(angle));

        if (position < num_values) {
            output[position++] = static_cast<float>(amplitude * std::sin(angle));
        }
    }
}

std::vector<float> make_sigmas(int64_t latent_tokens, int64_t num_steps) {
    // FLUX.2 Klein's empirical shift and exponential FlowMatch schedule.
    const double m_200 = 0.00016927 * static_cast<double>(latent_tokens) + 0.45666666;
    double mu          = m_200;

    if (latent_tokens <= 4300) {
        const double m_10      = 8.73809524e-05 * static_cast<double>(latent_tokens) + 1.89833333;
        const double slope     = (m_200 - m_10) / 190.0;
        const double intercept = m_200 - 200.0 * slope;

        mu = slope * static_cast<double>(num_steps) + intercept;
    }

    // Match the upstream float32 sigma arithmetic after evaluating exp(mu) in double.
    const float shift = static_cast<float>(std::exp(mu));
    std::vector<float> sigmas(num_steps + 1, 0.0F);

    for (int64_t step = 0; step < num_steps; ++step) {
        const float sigma = static_cast<float>(1.0 - static_cast<double>(step) / static_cast<double>(num_steps));

        sigmas[step] = shift / (shift + (1.0F / sigma - 1.0F));
    }

    // The final zero completes the last Euler step at the clean endpoint.
    return sigmas;
}

}  // namespace

ValidationResult FLUX2KleinGeneration::validate(const ComputeFlowManifest& manifest) const {
    const auto& info = manifest.get_brick_info();

    COMPUTE_FLOW_VALIDATE(info.type == "image_generation");
    COMPUTE_FLOW_VALIDATE(info.name == "FLUX2KleinGeneration");
    COMPUTE_FLOW_VALIDATE(info.version.major == 1);

    // Check modules presence
    COMPUTE_FLOW_VALIDATE(manifest.num_modules() == 4);
    COMPUTE_FLOW_VALIDATE(manifest.has_module("tokenizer"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("text_encoder"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("denoiser"));
    COMPUTE_FLOW_VALIDATE(manifest.has_module("vae_decode"));

    // Check valid file and accelerator for each module
    for (const auto& name : {"tokenizer", "text_encoder", "denoiser", "vae_decode"}) {
        const auto& model = manifest[name];

        COMPUTE_FLOW_VALIDATE(!model.model_path.empty());
        COMPUTE_FLOW_VALIDATE(model.acc_config.priorities.size() == 1);
        COMPUTE_FLOW_VALIDATE(model.acc_config.has_accelerator(AcceleratorType::Cpu));
    }

    // Check valid runner for each module
    COMPUTE_FLOW_VALIDATE(manifest["tokenizer"].framework_type == RunnerType::HuggingFaceTokenizer);
    COMPUTE_FLOW_VALIDATE(manifest["text_encoder"].framework_type == RunnerType::LiteRT);
    COMPUTE_FLOW_VALIDATE(manifest["denoiser"].framework_type == RunnerType::LiteRT);
    COMPUTE_FLOW_VALIDATE(manifest["vae_decode"].framework_type == RunnerType::LiteRT ||
                          manifest["vae_decode"].framework_type == RunnerType::ExecuTorch);

    // Check module inputs, outputs and recurrent state
    COMPUTE_FLOW_VALIDATE(manifest["tokenizer"].has_input("text"));
    COMPUTE_FLOW_VALIDATE(manifest["tokenizer"].has_output("tokens"));
    COMPUTE_FLOW_VALIDATE(manifest["text_encoder"].has_input("input_ids"));
    COMPUTE_FLOW_VALIDATE(manifest["text_encoder"].has_input("attention_mask"));
    COMPUTE_FLOW_VALIDATE(manifest["text_encoder"].has_output("context"));
    COMPUTE_FLOW_VALIDATE(manifest["denoiser"].has_input("context"));
    COMPUTE_FLOW_VALIDATE(manifest["denoiser"].has_input("timestep"));
    COMPUTE_FLOW_VALIDATE(manifest["denoiser"].has_input("sigma_delta"));
    COMPUTE_FLOW_VALIDATE(manifest["denoiser"].has_state("latent"));
    COMPUTE_FLOW_VALIDATE(manifest["denoiser"].state("latent").first == 0);
    COMPUTE_FLOW_VALIDATE(manifest["denoiser"].state("latent").second == 0);
    COMPUTE_FLOW_VALIDATE(manifest["vae_decode"].has_input("latent"));
    COMPUTE_FLOW_VALIDATE(manifest["vae_decode"].has_output("image"));

    // Check configs presence; values can change through update_config_impl().
    COMPUTE_FLOW_VALIDATE(manifest.has_config("seed"));
    COMPUTE_FLOW_VALIDATE(manifest.has_config("num_steps"));

    // Check constants presence
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("pad_token_id"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("prompt_prefix"));
    COMPUTE_FLOW_VALIDATE(manifest.has_constant("prompt_suffix"));

    // Validate constants values
    COMPUTE_FLOW_VALIDATE(manifest.get_constant<int64_t>("pad_token_id") >= 0);

    return {};
}

void FLUX2KleinGeneration::load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    _tokenizer = make_runner(manifest["tokenizer"].framework_type);

    _tokenizer->setup(manifest["tokenizer"], manifest.get_brick_info(), profiler);

    _text_encoder = make_runner(manifest["text_encoder"].framework_type);
    _denoiser     = make_runner(manifest["denoiser"].framework_type);
    _vae_decode   = make_runner(manifest["vae_decode"].framework_type);

    // Retain prepared runners so each generation only performs inference.
    _text_encoder->setup(manifest["text_encoder"], manifest.get_brick_info(), profiler);
    _denoiser->setup(manifest["denoiser"], manifest.get_brick_info(), profiler);
    _vae_decode->setup(manifest["vae_decode"], manifest.get_brick_info(), profiler);
}

void FLUX2KleinGeneration::prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    const auto ids_format    = _text_encoder->get_input_format("input_ids");
    const auto mask_format   = _text_encoder->get_input_format("attention_mask");
    const auto latent_format = _denoiser->get_input_format("latent");

    COMPUTE_FLOW_CHECK(ids_format.dt == DataType::I64);
    COMPUTE_FLOW_CHECK(ids_format.shape.size() == 2);
    COMPUTE_FLOW_CHECK(ids_format.shape[0] == 1 && ids_format.shape[1] > 0);
    COMPUTE_FLOW_CHECK(mask_format.dt == ids_format.dt && mask_format.shape == ids_format.shape);
    COMPUTE_FLOW_CHECK(latent_format.dt == DataType::F32);
    COMPUTE_FLOW_CHECK(latent_format.shape.size() == 3);
    COMPUTE_FLOW_CHECK(latent_format.shape[0] == 1 && latent_format.shape[1] > 0 && latent_format.shape[2] > 0);

    _text_sequence_length = ids_format.shape[1];
    _latent_tokens        = latent_format.shape[1];
    _pad_token_id         = manifest.get_constant<int64_t>("pad_token_id");
    _prompt_prefix        = manifest.get_constant<std::string>("prompt_prefix");
    _prompt_suffix        = manifest.get_constant<std::string>("prompt_suffix");

    _memory.allocate("input_ids", ids_format);
    _memory.allocate("attention_mask", mask_format);
    _memory.allocate("initial_noise", latent_format);
    _memory.allocate("timestep", {DataType::F32, {1}});
    _memory.allocate("sigma_delta", {DataType::F32, {1}});

    _seed      = -1;
    _num_steps = 0;

    update_config_impl(manifest, profiler);
}

void FLUX2KleinGeneration::update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) {
    (void)profiler;

    const int64_t seed      = manifest.get_config<int64_t>("seed");
    const int64_t num_steps = manifest.get_config<int64_t>("num_steps");

    COMPUTE_FLOW_CHECK(seed >= 0);
    COMPUTE_FLOW_CHECK(num_steps == 2 || num_steps == 3 || num_steps == 4);

    if (seed != _seed) {
        fill_gaussian_noise(_memory.get_data_as<float>("initial_noise"),
                            _memory.get_tensor("initial_noise").format.get_num_elems(), static_cast<uint64_t>(seed));
        _seed = seed;
    }

    if (num_steps != _num_steps) {
        _sigmas    = make_sigmas(_latent_tokens, num_steps);
        _num_steps = num_steps;
    }
}

TensorDict FLUX2KleinGeneration::run_impl(const TensorDict& input, ProfilerView profiler) {
    (void)profiler;

    COMPUTE_FLOW_CHECK(!_is_eog);

    const auto& prompt = input["prompt"];

    COMPUTE_FLOW_CHECK(prompt.format.dt == DataType::String);
    COMPUTE_FLOW_CHECK(prompt.format.shape.size() == 1);
    COMPUTE_FLOW_CHECK(prompt.format.shape[0] >= 0);
    COMPUTE_FLOW_CHECK(prompt.data != nullptr || prompt.format.shape[0] == 0);

    const std::string instruction =
        prompt.format.shape[0] == 0 ? "" : std::string(static_cast<const char*>(prompt.data), prompt.format.shape[0]);
    std::string text = _prompt_prefix + instruction + _prompt_suffix;

    {
        TensorData text_input{{DataType::String, {static_cast<int64_t>(text.size())}}, text.data(), true};

        _tokenizer->set_input("text", text_input);
        _tokenizer->run();

        const auto tokens        = _tokenizer->get_output("tokens");
        const auto* token_ids    = static_cast<const int32_t*>(tokens.data);
        const int64_t num_tokens = std::min(tokens.format.get_num_elems(), _text_sequence_length);
        auto* ids                = _memory.get_data_as<int64_t>("input_ids");
        auto* mask               = _memory.get_data_as<int64_t>("attention_mask");

        std::fill_n(ids, _text_sequence_length, _pad_token_id);
        std::fill_n(mask, _text_sequence_length, 0);

        for (int64_t index = 0; index < num_tokens; ++index) {
            ids[index]  = token_ids[index];
            mask[index] = 1;
        }
    }

    {
        auto& runner = _text_encoder;

        runner->set_input("input_ids", _memory.get_tensor("input_ids"));
        runner->set_input("attention_mask", _memory.get_tensor("attention_mask"));
        runner->run();
    }

    {
        auto& runner = _denoiser;

        runner->set_input("latent", _memory.get_tensor("initial_noise"));
        runner->set_input("context", _text_encoder->get_output("context"));

        for (int64_t step = 0; step < _num_steps; ++step) {
            *_memory.get_data_as<float>("timestep")    = _sigmas[step];
            *_memory.get_data_as<float>("sigma_delta") = _sigmas[step + 1] - _sigmas[step];

            runner->set_input("timestep", _memory.get_tensor("timestep"));
            runner->set_input("sigma_delta", _memory.get_tensor("sigma_delta"));
            runner->run();
            // The manifest state carries the latent between steps inside the runner.
        }
    }

    {
        auto& runner = _vae_decode;

        runner->set_input("latent", _denoiser->get_output("latent"));
        runner->run();
    }

    TensorDict output;

    // The decoder owns this output until the next run, reset or setup.
    output.declare_tensor("image") = _vae_decode->get_output("image");
    _is_eog                        = true;
    return output;
}

void FLUX2KleinGeneration::warmup_impl(ProfilerView profiler) {
    std::string prompt;
    TensorDict input;

    input.declare_tensor("prompt") = {{DataType::String, {0}}, prompt.data(), true};
    (void)run_impl(input, profiler);
}

void FLUX2KleinGeneration::reset_states_impl(ProfilerView profiler) {
    (void)profiler;
    _tokenizer->reset_states();

    _text_encoder->reset_states();
    _denoiser->reset_states();
    _vae_decode->reset_states();
    _is_eog = false;
}

bool FLUX2KleinGeneration::is_generation_complete() const { return _is_eog; }

}  // namespace compute_flow::bricks::image_generation
