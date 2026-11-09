//
// SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
//
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "compute_flow/bricks/brick_base_v1.h"
#include "compute_flow/memory_planner.h"
#include "compute_flow/runners/runner.h"

namespace compute_flow::bricks::image_generation {

/** @brief Generate an RGB image from a text prompt. */
class FLUX2KleinGeneration : public BrickBaseV1 {
public:
    std::string_view api() const noexcept override { return "image_generation.v1"; }
    ValidationResult validate(const ComputeFlowManifest& manifest) const override;
    bool is_generation_complete() const override;

protected:
    void load_models_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    void prepare_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    void warmup_impl(ProfilerView profiler) override;
    void update_config_impl(const ComputeFlowManifest& manifest, ProfilerView profiler) override;
    TensorDict run_impl(const TensorDict& input, ProfilerView profiler) override;
    void reset_states_impl(ProfilerView profiler) override;

private:
    std::unique_ptr<IRunner> _tokenizer;
    std::unique_ptr<IRunner> _text_encoder;
    std::unique_ptr<IRunner> _denoiser;
    std::unique_ptr<IRunner> _vae_decode;
    MemoryPlanner _memory;
    std::string _prompt_prefix;
    std::string _prompt_suffix;
    std::vector<float> _sigmas;
    int64_t _pad_token_id         = 0;
    int64_t _text_sequence_length = 0;
    int64_t _latent_tokens        = 0;
    int64_t _seed                 = -1;
    int64_t _num_steps            = 0;
    bool _is_eog                  = false;
};

}  // namespace compute_flow::bricks::image_generation
