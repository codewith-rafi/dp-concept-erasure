# models/ — model checkpoints (not committed)

This folder holds the pretrained model weights used for the reproduce run.
They are **ignored by git** (≈24 GB) and must be downloaded/placed manually.

## Expected contents

| Subfolder | What goes here |
|---|---|
| `stable-diffusion-v1-4/` | Stable Diffusion v1.4 checkpoint, e.g. from `runwayml/stable-diffusion-v1-4` on Hugging Face (huggingface.co/runwayml/stable-diffusion-v1-4) |
| `sd14-lean/` | Pruned / lean variant of SD 1.4 used for the faster runs |

Download the weights and place them in the corresponding subfolder so the
scripts under `../source_code/SD/` can find them.
