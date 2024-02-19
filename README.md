# DLIP V1 (Distilling BLIP)

Distilled BLIP model (VIT Small, BERT Small, and BLIP Base) achieves similar performance to BLIP with 4x the speed in captioning and retrieval. This repo replicates the performance achieved in this [paper](https://arxiv.org/abs/2308.12956).

## Note
I would recommend using [DLIP V2](https://github.com/jfan1256/dlip-v2) for officially training. However, this repo experimented with a novel distillation loss, which can be referenced in /model_help/distill.py.
