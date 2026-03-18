uv run main.py \
   --output_dir "outputs" \
   --checkpoint_dir "checkpoints/DIT_1350_epochs" \
   --checkpoint_path "checkpoints/DIT_1350_epochs/FM_epoch_0400.pt" \
   --save_every 100 \
   --sample_every 102 \
   --num_epochs 1350 \
   --model "dit" \
   --sample_steps 40 \