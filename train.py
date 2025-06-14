import os
import subprocess

# --- Set user variables ---
STAGE_PATH = "Efficient-Large-Model/NVILA-Lite-8B"
DATA_MIXTURE = "BDD"
OUTPUT_DIR = "runs/BDD_trainset"
RUN_NAME = "NVILA-Lite-8B-test"

# --- Multi-GPU settings (single node) ---
GPUS_PER_NODE = 2  # set to 2, 4, or however many GPUs you want to use

GLOBAL_TRAIN_BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 1
PER_DEVICE_TRAIN_BATCH_SIZE = GLOBAL_TRAIN_BATCH_SIZE // (GPUS_PER_NODE * GRADIENT_ACCUMULATION_STEPS)

os.environ["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"

# --- Torchrun command ---
cmd = [
    "torchrun",
    f"--nproc_per_node={GPUS_PER_NODE}",  # only this is needed for multi-GPU single-node
    "llava/train/train_mem.py",
    "--deepspeed", "scripts/zero3.json",
    "--model_name_or_path", STAGE_PATH,
    "--data_mixture", DATA_MIXTURE,
    "--vision_tower", "Efficient-Large-Model/paligemma-siglip-so400m-patch14-448",
    "--mm_vision_select_feature", "cls_patch",
    "--mm_projector", "mlp_downsample_3x3_fix",
    "--tune_language_model", "True", #! key parameter
    "--mm_vision_select_layer", "-2",
    "--mm_use_im_start_end", "False",
    "--mm_use_im_patch_token", "False",
    "--image_aspect_ratio", "dynamic",
    "--bf16", "True",
    "--output_dir", f"{OUTPUT_DIR}/model",
    "--num_train_epochs", "1",
    "--per_device_train_batch_size", str(PER_DEVICE_TRAIN_BATCH_SIZE),
    "--gradient_accumulation_steps", str(GRADIENT_ACCUMULATION_STEPS),
    "--evaluation_strategy", "no",
    "--save_strategy", "steps",
    "--save_steps", "100",
    "--save_total_limit", "1",
    "--learning_rate", "1e-4", #! key parameter
    "--weight_decay", "0.",
    "--warmup_ratio", "0.03",
    "--lr_scheduler_type", "cosine",
    "--logging_steps", "1",
    "--model_max_length", "2048", #! key parameter
    "--gradient_checkpointing", "True",
    "--dataloader_num_workers", "0", #! key parameter
    "--vflan_no_system_prompt", "True",
    "--report_to", "tensorboard",
    "--tune_mm_projector", "False", #! key parameter
    "--tune_vision_tower", "False", #! key parameter
    "--lora_enable", #! key parameter
    "--lora_r", "16", #! key parameter
    "--lora_alpha", "16",
    "--lora_dropout", "0.05",
    "--lora_bias", "none", 
    "--lora_llm", "True", #! key parameter
    "--lora_vt", "False" #! key parameter
]

# --- Run the multi-GPU training ---
subprocess.run(cmd)
