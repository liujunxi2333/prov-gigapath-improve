from __future__ import annotations

import torch
import timm

import gigapath.slide_encoder as gigapath_slide_encoder


def build_encoders(
    tile_weight: str,
    slide_weight: str,
    device: torch.device,
    split_models_across_two_gpus: bool = True,
    *,
    tile_parallel: str = "single",
):
    n_gpu = torch.cuda.device_count() if device.type == "cuda" else 0
    use_split = bool(split_models_across_two_gpus and device.type == "cuda" and n_gpu >= 2)
    tile_parallel = (tile_parallel or "single").strip().lower()
    if tile_parallel not in ("single", "dataparallel"):
        raise ValueError('tile_parallel must be "single" or "dataparallel"')

    tile_device = torch.device("cuda:0") if use_split else device
    slide_device = torch.device("cuda:1") if use_split else device

    tile_enc = timm.create_model(
        "vit_giant_patch14_dinov2",
        pretrained=True,
        img_size=224,
        in_chans=3,
        pretrained_cfg_overlay=dict(file=tile_weight),
    ).to(tile_device)
    tile_enc.eval()

    dp_tile = False
    if use_split and tile_parallel == "dataparallel" and n_gpu >= 2:
        tile_enc = torch.nn.DataParallel(tile_enc, device_ids=[0, 1])
        dp_tile = True
    elif (not use_split) and device.type == "cuda" and n_gpu >= 2:
        tile_enc = torch.nn.DataParallel(tile_enc, device_ids=[0, 1])
        dp_tile = True

    slide_enc = gigapath_slide_encoder.create_model(
        pretrained=slide_weight,
        model_arch="gigapath_slide_enc12l768d",
        in_chans=1536,
    ).to(slide_device)
    slide_enc.eval()
    return tile_enc, slide_enc, tile_device, slide_device, use_split, dp_tile, tile_parallel
