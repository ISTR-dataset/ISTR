accelerate launch \
  --config_file accelerate_config.yaml \
  --num_cpu_threads_per_process=8 \
  ./train.py \
  --sample_prompts=""  \
  --config_file=""