import argparse
import json
import os

import torch
import torch.distributed as dist
from accelerate.logging import get_logger
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from scripts.dataset import getdataset

from wan.modules.vae import WanVAE
from wan.modules.t5 import T5EncoderModel
from wan.modules.clip import CLIPModel

logger = get_logger(__name__)


def main(args):
    local_rank = int(os.getenv("RANK", 0))
    world_size = int(os.getenv("WORLD_SIZE", 1))
    print("world_size", world_size, "local rank", local_rank)
    train_dataset = getdataset(args)
    assert args.train_batch_size==1, "only support batch size 1"
    sampler = DistributedSampler(train_dataset, rank=local_rank, num_replicas=world_size, shuffle=True)
    train_dataloader = DataLoader(
        train_dataset,
        sampler=sampler,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(local_rank)
    if not dist.is_initialized():
        dist.init_process_group(backend="nccl", init_method="env://", world_size=world_size, rank=local_rank)

    autocast_type = torch.bfloat16

    vae = WanVAE(vae_pth=os.path.join(args.model_path, "Wan2.1_VAE.pth"))
    vae.model = vae.model.to(device).to(autocast_type)
    vae.model.eval()
    vae.model.requires_grad_(False)

    text_encoder = T5EncoderModel(
        text_len=512,
        dtype=autocast_type,
        device=device,
        checkpoint_path=os.path.join(args.model_path, "models_t5_umt5-xxl-enc-bf16.pth"),
        tokenizer_path=os.path.join(args.model_path, "google/umt5-xxl"),
        shard_fn= None,
    )

    clip = CLIPModel(
            dtype=autocast_type,
            device=device,
            checkpoint_path=os.path.join(args.model_path,'models_clip_open-clip-xlm-roberta-large-vit-huge-14.pth'),
            tokenizer_path=os.path.join(args.model_path, 'xlm-roberta-large'))

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "latent"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "prompt_embed"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "y"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "clip_feature"), exist_ok=True)

    json_data = []
    for _, data in tqdm(enumerate(train_dataloader), disable=local_rank != 0):
        with torch.inference_mode():
            with torch.autocast("cuda", dtype=autocast_type):
                input = data["pixel_values"].to(device)
                latents = vae.encode(input)[0]

                txt = data["text"]
                text_embed = text_encoder(txt,device)[0]

                img = data["img"][0]
                h, lat_h = args.max_height, args.max_height // 8
                w, lat_w = args.max_width, args.max_width // 8
                assert lat_h == latents.size(2) and lat_w == latents.size(3)
                msk = torch.ones(1, 81, lat_h, lat_w, device=device)
                msk[:, 1:] = 0
                msk = torch.concat([
                    torch.repeat_interleave(msk[:, 0:1], repeats=4, dim=1), msk[:, 1:]
                ],dim=1)
                msk = msk.view(1, msk.shape[1] // 4, 4, lat_h, lat_w)
                msk = msk.transpose(1, 2)[0]
                y = vae.encode([
                    torch.concat([
                        torch.nn.functional.interpolate(
                        img[None].cpu(), size=(h, w), mode='bicubic').transpose(0, 1),
                        torch.zeros(3, 80, h, w)
                    ],dim=1).to(device)
                ])[0]
                y = torch.concat([msk, y])
                clip_context = clip.visual([img[:, None, :, :].to(device)])[0]

            video_name = os.path.basename(data["path"][0])
            video_suffix = video_name.split(".")[-1]
            video_name = video_name[: -len(video_suffix) - 1]
            
            item = {}

            for name, tensor in zip(["latent", "prompt_embed", "y", "clip_feature"], [latents, text_embed, y, clip_context]):
                tensor_path = os.path.join(args.output_dir, name, video_name + ".pt")
                torch.save(tensor.to(autocast_type), tensor_path)
                item[name + "_path"] = tensor_path
            
            item["length"] = latents.shape[1]
            item["caption"] = data["text"][0]
            json_data.append(item)
    dist.barrier()
    local_data = json_data
    gathered_data = [None] * world_size
    dist.all_gather_object(gathered_data, local_data)
    if local_rank == 0:
        all_json_data = [item for sublist in gathered_data for item in sublist]
        with open(os.path.join(args.output_dir, "videos2caption.json"), "w") as f:
            json.dump(all_json_data, f, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # dataset & dataloader
    parser.add_argument("--model_path", type=str, default="data/mochi")
    parser.add_argument("--model_type", type=str, default="mochi")
    parser.add_argument("--data_merge_path", type=str, required=True)
    parser.add_argument("--num_frames", type=int, default=163)
    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=1,
        help="Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process.",
    )
    parser.add_argument(
        "--train_batch_size",
        type=int,
        default=16,
        help="Batch size (per device) for the training dataloader.",
    )
    parser.add_argument("--num_latent_t", type=int, default=28, help="Number of latent timesteps.")
    parser.add_argument("--max_height", type=int, default=480)
    parser.add_argument("--max_width", type=int, default=848)
    parser.add_argument("--video_length_tolerance_range", type=int, default=2) #defualt 2
    parser.add_argument("--group_frame", action="store_true")  # TODO
    parser.add_argument("--group_resolution", action="store_true")  # TODO
    parser.add_argument("--dataset", default="t2v")
    parser.add_argument("--train_fps", type=int, default=30)
    parser.add_argument("--use_image_num", type=int, default=0)
    parser.add_argument("--text_max_length", type=int, default=512)
    parser.add_argument("--speed_factor", type=float, default=1.0)
    parser.add_argument("--drop_short_ratio", type=float, default=1.0)
    # text encoder & vae & diffusion model
    parser.add_argument("--text_encoder_name", type=str, default="google/t5-v1_1-xxl")
    parser.add_argument("--cache_dir", type=str, default="./cache_dir")
    parser.add_argument("--cfg", type=float, default=0.0)
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=("[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
              " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."),
    )

    args = parser.parse_args()
    main(args)
