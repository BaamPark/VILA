# Note for team
Finetuning whole parts of NVILA is not avilable with our limited hardware settings (2 x 3090 Ti). To avoid out-of-memory (OOM) issue, we should only fine-tune the LLM with LoRA, leaving ViT and the projector frozen. 

## Requirements
The training process takes a lot of DRAM as well as VRAM. I ran into killed-process many times even though I have 64 GB RAM. To avoid OOM, you need to add swap space. I recommend 64 GB for SWAP. Follow the below instructions.
```bash
sudo swapoff -a
sudo rm -f /swapfile  # optional, in case it exists
sudo fallocate -l 64G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile
```

Then, run `free -h`. You will see something like this. See total column.
```bash
(vila) (base) beomseok@lab731-X299X-AORUS-MASTER:~/AICityTrack2/VILA$ free -h
              total        used        free      shared  buff/cache   available
Mem:           62Gi        22Gi        18Gi       427Mi        21Gi        38Gi
Swap:          63Gi        30Gi        33Gi                          
```
## Instructions
1. In the AI city project dir, run `tools/make_vila_dataset.py`
    - This will generate the datasets under `data` that VILA understand
2. In the VILA dir, open `llava/data/registry/datasets/default.yaml` and register the datasets - [ref](https://github.com/NVlabs/VILA/tree/main/finetuning#:~:text=video%20data%0A%5D-,SampleVideo%3A,-_target_%3A%20llava)
3. Open `train.py` and set `DATA_MIXTURE` to the registered the dataset name

4. Lastly, run `train.py` to finetune the NVILA.

## Parameters
These are the key parameters we should consider:
- `GLOBAL_TRAIN_BATCH_SIZE`: Total batchsize for all gpus.
- `tune_language_model`: whether to update the LLM parameters directly.
- `num_train_epochs`: number of epoch.
- `learning_rate`: global learning rate;
- `model_max_length`: maximum input sequence length; higher values increase memory usage quadratically.
- `dataloader_num_workers`: Number of workers for pytorch dataloader.
- `tune_mm_projector`: whether to fine-tune the multi-modal projector.
- `tune_vision_tower`: whether to fine-tune the entire vision encoder (ViT).
- `lora_enable`: enables LoRA.
- `lora_r`: LoRA rank; controls the size of low-rank adaptation matrices.
- `lora_alpha`: it's common practice to set `lora_alpha = 2 * lora_r`.
- `lora_llm`: apply LoRA to the LLM component (set to True for our setup).
- `lora_vt`: apply LoRA to the vision encoder (set to False unless you're tuning the visual backbone).

### Rationales
These are my rationales for parameter values
- `GLOBAL_TRAIN_BATCH_SIZE=2`: This is lowest we can do to reduce the VRAM usage.
- `num_train_epochs = 1`: Top1 fine-tune with one epoch. Top2 fine-tune with three epochs. In LLM it's common to set small epochs.
- `learning_rate = 1e-4`: Top1 and Top2 did the same. 
- `model_max_length=4096`: The default value is 4096. It might shorten the VRAM usage if you set 2048 but when inputting single image, we require 4096 context window due to dynamic_s2 method. Therefore, we will use default value.
- `dataloader_num_workers=0`: The default is 16. Higher values may accelarate the batch loading on GPU but takes a lot of DRAM.
- `tune_mm_projector=False`: I set it false because there is no point of finetuning when freezing ViT
- `lora_r=16`: The default is 64. Lower the rank, lower the VRAM usage.

## Error you might encounter
If you encounter these runtime error, you can simply redo `train.py`. It will exactly redo the finetuning where it left off keeping the decayed learning rate.

### Out of VRAM
```bash
OutOfMemoryError: CUDA out of memory. Tried to allocate 3.78 GiB. GPU  has a total capacity of 23.69 GiB of which 875.62 MiB is free. Process 492814 has 3.21 GiB memory in use. Including non-PyTorch memory, this process has 19.40 GiB memory in use. Of the allocated memory 17.97 GiB is allocated by PyTorch, and 1.02 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation. 
```

### Out of DRAM
```bash
llava/train/train_mem.py FAILED
------------------------------------------------------
Failures:
  <NO_OTHER_FAILURES>
------------------------------------------------------
Root Cause (first observed failure):
[0]:
  time      : 2025-06-12_22:45:44
  host      : lab731-X299X-AORUS-MASTER
  rank      : 0 (local_rank: 0)
  exitcode  : -9 (pid: 79468)
  error_file: <N/A>
  traceback : Signal 9 (SIGKILL) received by PID 79468
```

### Distributed learning failure (timeout)
```bash
failed (exitcode: 124) local_rank: 0 (pid: 94384) of binary: /home/beomseok/anaconda3/envs/vila/bin/python3.10
Traceback (most recent call last):
  File "/home/beomseok/anaconda3/envs/vila/bin/torchrun", line 8, in <module>
    sys.exit(main())
  File "/home/beomseok/anaconda3/envs/vila/lib/python3.10/site-packages/torch/distributed/elastic/multiprocessing/errors/__init__.py", line 347, in wrapper
    return f(*args, **kwargs)
  File "/home/beomseok/anaconda3/envs/vila/lib/python3.10/site-packages/torch/distributed/run.py", line 879, in main
    run(args)
  File "/home/beomseok/anaconda3/envs/vila/lib/python3.10/site-packages/torch/distributed/run.py", line 870, in run
    elastic_launch(
  File "/home/beomseok/anaconda3/envs/vila/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 132, in __call__
    return launch_agent(self._config, self._entrypoint, list(args))
  File "/home/beomseok/anaconda3/envs/vila/lib/python3.10/site-packages/torch/distributed/launcher/api.py", line 263, in launch_agent
  llava/train/train_mem.py FAILED
------------------------------------------------------------
Failures:
[1]:
  time      : 2025-06-13_18:33:16
  host      : lab731-X299X-AORUS-MASTER
  rank      : 1 (local_rank: 1)
  exitcode  : 124 (pid: 94385)
  error_file: <N/A>
  traceback : To enable traceback see: https://pytorch.org/docs/stable/elastic/errors.html

```