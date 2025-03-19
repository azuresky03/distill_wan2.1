# CUDA_VISIBLE_DEVICES=1,2,3,4 
torchrun --nproc_per_node=8 --master-port 29516 generate.py --task t2v-14B --size 1280*720 --sample_steps 20 --ckpt_dir /vepfs-zulution/models/Wan2.1-T2V-14B --dit_fsdp --t5_fsdp --ulysses_size 4 --ring_size 2 --sample_shift 17\
 --sample_guide_scale 6 --prompt "A cat walks on the grass, realistic style." --base_seed 42 --frame_num 81