# Distillation and Post-Training for Wan2.1

**TL;DR —** High-quality video generation in just **five denoising steps**.  
See the results in the [Combined Techniques](#combined-techniques) section.

This project focuses on the distillation and post-training processes for the Wan2.1 model, aiming to enhance its efficiency and performance. All data processing, training and inference code along with model weights are open-sourced.

## Table of Contents
- [References](#references)
- [Data Preprocessing](#data-preprocessing)
- [Distillation Techniques](#distillation-techniques)
  - [CFG Distillation](#cfg-distillation)
  - [Step Distillation](#step-distillation)
  - [DMD2](#dmd2)
- [Reinforcement Learning](#reinforcement-learning-rl)
- [Inference](#inference)
- [Combined Techniques](#combined-techniques)
- [Model Weights](#model-weights)
- [Environment Setup](#environment-setup)

## References
This work is based on:
* [Wan-Video/Wan2.1](https://github.com/Wan-Video/Wan2.1)
* [hao-ai-lab/FastVideo](https://github.com/hao-ai-lab/FastVideo)

The work is done at [Zulution AI](https://huggingface.co/ZuluVision)

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
<table>
<tr>
<td align="center">
<video src="https://github.com/user-attachments/assets/0ee4a260-0dee-47d2-a080-9f7a60d83825" width="320" controls loop muted></video>
</td>
<td align="center">
<video src="https://github.com/user-attachments/assets/62a62cc3-ad78-4477-95d2-ca711d569b5e" width="320" controls loop muted></video>
</td>
</tr>
<tr>
<td align="center">
<video src="https://github.com/user-attachments/assets/18981a28-a7d7-4008-8653-4aed35ffe052" width="320" controls loop muted></video>
</td>
<td align="center">
<video src="https://github.com/user-attachments/assets/f84b27c4-048b-4f05-941a-f974936d7a10" width="320" controls loop muted></video>
</td>
</tr>
</table>


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
    halfed, 20 steps

https://github.com/user-attachments/assets/bff12f8b-f7f6-48b9-b55a-3efc25aa5b09


https://github.com/user-attachments/assets/4ac47433-11e4-4e1e-a7f4-a0c8893e579d

orignial cfg-distilled 30 steps vs. further step-distilled 15 steps
<table>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/1cf1d74c-4a7a-4ab1-83ac-fb20ffea8ff3" width="320" controls loop muted></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/8528ebb7-2958-4888-ab9b-cc0767f63e1e" width="320" controls loop muted></video>
    </td>
  </tr>
</table>

### DMD2
* **Description**: Distribution Matching Distillation is applied to further optimize the model.
* **Implementation**: For detailed implementation, see [DMD2_wanx](https://github.com/azuresky03/DMD2_wanx).

## Reinforcement Learning (RL)
* **Method**: Implemented according to the concept from [DRaFT](https://arxiv.org/pdf/2309.17400).
* **Reward**: HPSReward V2.1, with implementation from Easyanimate.
* **Best Practices**: Our experiments found it best to train with LoRA and apply reward on the first frame.
* **Training**: Run the script `RL/sh/debug.sh`.
* **Inference**: Set `lora_alpha` to a smaller value than during training for more natural-looking videos.
* **Example Results**:
<table>
  <tr>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/6d2f61d3-a0a0-4dbe-92bb-5e8e42ca19bb" width="320" controls loop muted></video>
    </td>
    <td align="center">
      <video src="https://github.com/user-attachments/assets/dbc66d5f-8af6-4b7e-9bb1-d03740397ac9" width="320" controls loop muted></video>
    </td>
  </tr>
</table>

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


https://github.com/user-attachments/assets/aeaf967f-f17e-44d2-9809-501fc93d66b3


https://github.com/user-attachments/assets/3a5f57b8-0815-40f5-b427-d1b08abf335d


https://github.com/user-attachments/assets/1e5c46df-c794-4533-aa7f-7365c65c8a2d


https://github.com/user-attachments/assets/24e1213b-4617-46eb-b5d5-cf4fb08a80dd


https://github.com/user-attachments/assets/6e11c8cd-45c6-47f2-9b19-f5831a75e710


## Model Weights
All pretrained model weights are available for download:
* **Baidu Pan**: [https://pan.baidu.com/s/1wUCrRY9Fu8GdDMTZXdc7tw?pwd=m9kn](https://pan.baidu.com/s/1wUCrRY9Fu8GdDMTZXdc7tw?pwd=m9kn)
* **Access Code**: `m9kn`

## Environment Setup

* **Dependencies**: All required packages are listed in the `environment.yml` file.
* **FastVideo**: This project requires FastVideo. Please install our forked version from:
  ```bash
  git clone https://github.com/azuresky03/FastVideo.git
  cd FastVideo
  pip install -e .
  ```
