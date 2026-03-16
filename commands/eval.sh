uv run eval.py \
   --checkpoint_path "checkpoints/DIT_800_epochs/FM_epoch_0800.pt"\
   --model "dit" \
   --sample_steps 40 \
   --guidance_scale 2 \
   --num_samples_FID 40000 \
