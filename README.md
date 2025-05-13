# Distillation and Post-Training for Wan2.1

This project focuses on the distillation and post-training processes for the Wan2.1 model, aiming to enhance its efficiency and performance.

## References
This work is based on:
* [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)
* [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo)

## Data Preprocessing
* Assuming your video data has been uniformly resized, run `scripts/data_preprocess/preprocess.sh` to preprocess the features required for model input.
* An example directory structure can be found under `./data`.

## Distillation Techniques
The following distillation techniques are utilized in this project:

### CFG Distillation
* **Description**: Eliminates the need for unconditional generation by fusing guidance information with temporal information using sinusoidal positional encoding and a Multi-Layer Perceptron (MLP).
* **Performance**: Achieves 2X acceleration with minimal performance degradation.
* **Training**: Run the script `scripts/train/sh/distill_cfg_i2v.sh`.
* **Comparative Results**:
  
distilled

https://github.com/user-attachments/assets/0ee4a260-0dee-47d2-a080-9f7a60d83825

original

https://github.com/user-attachments/assets/62a62cc3-ad78-4477-95d2-ca711d569b5e


distilled

https://github.com/user-attachments/assets/18981a28-a7d7-4008-8653-4aed35ffe052

original

https://github.com/user-attachments/assets/f84b27c4-048b-4f05-941a-f974936d7a10

### Step Distillation
* **Training**: Run the script `scripts/train/sh/distill_step.sh`.
* **Mode**: Defaults to text-to-video (t2v). For image-to-video (i2v) training, add the `--i2v` flag.
* **Types**:
  * **Consistency Distillation**:
    * Run with the `--distill_mode consistency` flag.
  * **Half Step Distillation**:
    * **Performance**: Achieves 2X acceleration with minimal performance loss.
    * **Description**: Consolidates the original two prediction steps into a single step.
    * **Training**: Run with the `--distill_mode half` flag.
    * **Example Results**:
      * `assets/halfed_38_The wom_step20_shift10_guide5.mp4` (halfed, 20 steps)
      * `assets/halfed_26_SuperVi_step20_shift10_guide5.mp4` (halfed, 20 steps)
      * `assets/cfg_一位留着黑长直_step30_shift3_guide5.mp4` (baseline, 30 steps) vs `assets/halfed_一位留着黑长直_step15_shift9_guide8.mp4` (distilled, 15 steps)
      * `assets/cfg_一个宇航员走在_step30_shift3_guide5.mp4` (baseline, 30 steps) vs `assets/halfed_一个宇航员走在_step15_shift9_guide8.mp4` (distilled, 15 steps)

## Reinforcement Learning (RL)
* **Method**: Implemented according to the concept from [DRaFT](https://arxiv.org/pdf/2309.17400).
* **Reward**: HPSReward V2.1, with implementation from Easyanimate.
* **Best Practices**: Our experiments found it best to train with LoRA and apply reward on the first frame.
* **Training**: Run the script `RL/sh/debug.sh`.
* **Inference**: Set `lora_alpha` to a smaller value than during training for more natural-looking videos.
* **Example Results**:
  * Original: `assets/org_In the _step7_shift13_guide8.mp4` vs After RL: `assets/RL_In the _step7_shift13_guide8.mp4`

## Inference
* **Text-to-Video (t2v)**
  * Run `scripts/inference/inference.sh` for text-to-video generation tasks.
  * Sample prompts are provided in `test_prompts.txt` and `moviibench_2.0_prompts.txt`.

* **Image-to-Video (i2v)**
  * Run `scripts/inference/i2v.sh` for image-to-video generation tasks.
  * Sample images and corresponding prompts are available in the `examples/i2v` directory.

* **Configuration**
  * Update the `transformer_dir` variable in the scripts to point to your model checkpoint directory.
  * Adjust LoRA-related settings in `generate.py` if using LoRA models.

## Combined Techniques
By combining distillation (based on DMD2) and RL, we can achieve high-quality video generation in just 5 steps:
* `assets/A coffe_step5_shift7_guide8.mp4`
* `assets/A littl_step5_shift15_guide8.mp4`
* `assets/After j_step5_shift15_guide8.mp4`
* `assets/In the _step5_shift7_guide8.mp4`
* `assets/The aft_step5_shift7_guide8.mp4`
