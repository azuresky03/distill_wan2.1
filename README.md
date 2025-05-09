# Distillation and Post-Training for Wan2.1

This project focuses on the distillation and post-training processes for the Wan2.1 model, aiming to enhance its efficiency and performance.

This work is based on:
*   [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)
*   [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo)

## Data Preprocessing
*   Assuming your video data has been uniformly resized, run `scripts/data_preprocess/preprocess.sh` to preprocess the features required for model input.
*   An example directory structure can be found under `./data`.

## Distillation Techniques
The following distillation techniques are utilized in this project:

*   **CFG Distillation**
    *   This technique eliminates the need for unconditional generation by fusing guidance information with temporal information. This is achieved using sinusoidal positional encoding and a Multi-Layer Perceptron (MLP).
    *   It achieves approximately 2X acceleration with minimal performance degradation.
    *   To train, run the script: `scripts/train/sh/distill_cfg_i2v.sh`.
    *   Comparative Results:
        *   `assets/cfg_distill_26_SuperVi_step40_shift3_guide5.mp4` (distilled) vs. `assets/org_26_SuperVi_step40_shift3_guide5.mp4` (original)
        *   `assets/cfg_distill_38_The wom_step40_shift5_guide5.mp4` (distilled) vs. `assets/org_The wom_step40_shift5_guide5.mp4` (original)

*   **Step Distillation**
    *   To train, run the script: `scripts/train/sh/distill_step.sh`.
    *   Defaults to text-to-video (t2v). For image-to-video (i2v) training, add the `--i2v` flag.
    *   **Consistency Distillation**:
        *   Run with the `--distill_mode consistency` flag.
    *   **Half Step Distillation**:
        *   Achieves 2X acceleration with minimal performance loss.
        *   Aims to consolidate the original two prediction steps into a single step.
        *   Run with the `--distill_mode half` flag.
    *   Example Results for Half Step Distillation:
        *   `assets/cfg_一位留着黑长直_step30_shift3_guide5.mp4` (baseline, 30 steps) vs `assets/halfed_一位留着黑长直_step20_shift7_guide8.mp4` (distilled, 20 steps)
        *   `assets/cfg_一个宇航员走在_step30_shift3_guide5.mp4` (baseline, 30 steps) vs `assets/halfed_一个宇航员走在_step15_shift9_guide8.mp4` (distilled, 15 steps)


## Reinforcement Learning (RL)
*   This section will detail the Reinforcement Learning methodologies, training scripts, and results. (Further information will be added here.)