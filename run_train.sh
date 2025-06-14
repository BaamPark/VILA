# Define your output directory here
output_dir="runs/BDD_trainset"

# Run the fine-tuning script with:
# $1 = pretrained model path
# $2 = dataset name
# $3 = output directory
bash scripts/NVILA-Lite/sft.sh Efficient-Large-Model/NVILA-Lite-8B BDD "$output_dir"