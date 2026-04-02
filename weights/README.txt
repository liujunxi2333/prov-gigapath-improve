请将 GigaPath 预训练权重放置于此目录（或任意路径并通过 TILE_WEIGHT / SLIDE_WEIGHT 指定）：

- pytorch_model.bin  —— ViT tile encoder（timm vit_giant_patch14_dinov2）
- slide_encoder.pth  —— slide encoder（gigapath_slide_enc12l768d）

本仓库默认参数指向：
  weights/pytorch_model.bin
  weights/slide_encoder.pth
