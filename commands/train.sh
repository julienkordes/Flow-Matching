uv run main.py \
   --output_dir "outputs" \
   --checkpoint_dir "checkpoints/DIT_800_epochs" \
   --checkpoint_path "checkpoints/DIT_800_epochs/FM_epoch_0700.pt" \
   --save_every 50 \
   --sample_every 25 \
   --num_epochs 800 \
   --model "dit" \
   --sample_steps 40 \