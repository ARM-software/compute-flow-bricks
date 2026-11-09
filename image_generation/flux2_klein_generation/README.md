<!--
SPDX-FileCopyrightText: Copyright 2026 Arm Limited and/or its affiliates <open-source-office@arm.com>
SPDX-License-Identifier: Apache-2.0
-->

# Configuration

- `seed`: Nonnegative integer controlling the initial Gaussian noise. Change it to produce a
  different variation of the same prompt.
- `num_steps`: Number of denoising steps, accepting `2`, `3`, or `4`. More steps increase
  computation and can improve image quality.

Apply changes with `update_config()` between images, then call `reset_states()` before the next run.
