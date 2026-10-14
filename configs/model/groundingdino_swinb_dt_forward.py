import os

# Model config: GroundingDINO Swin-B, detectron2-style ("dtgroundingdino")
# INFERENCE forward -- no LoRA, no training hyperparameters.
#
# Why this exists alongside groundingdino_swinb_demo_forward.py: the ovdbench
# ``GroundingDinoAdapter`` drives models through the detectron2 batched-inputs
# contract (``model([{"image", "height", "width", "captions"}]) ->
# [{"instances": ...}]``), which the plain ``groundingdino`` demo-forward
# class does not implement (its forward requires ``captions=`` kwargs and
# returns raw logits). Keeping ``modelname = "dtgroundingdino"`` (inherited
# from the SwinT dt base) gives the exact same eval path every other adapter
# variant already uses.
#
# Backbone/checkpoint-flag overrides are VERBATIM the verified key-by-key
# diff documented in groundingdino_swinb_demo_forward.py (official HF
# GroundingDINO_SwinB.cfg.py vs the SwinT dt family): backbone,
# use_checkpoint, use_transformer_ckpt -- and nothing else.
_base_ = [
    os.path.join(os.environ["UC_GDINO_REPO"],
                 "groundingdino/config/GroundingDINO_SwinT_OGC_dt.py")
]

backbone = "swin_B_384_22k"
use_checkpoint = True
use_transformer_ckpt = True
