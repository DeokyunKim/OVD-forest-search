
# Model config: GroundingDINO Swin-B (detectron2-style "dtgroundingdino"
# TRAINING forward) under the FROZEN Phase 2 spec.
# Driven by train.py.
#
# WHY A NEW FILE INSTEAD OF AN EDIT
# ---------------------------------
# `scaling = lora_alpha / r` is rebuilt FROM THIS CONFIG at load time and is
# never stored beside the weights, so one shared config file serving several
# alphas is a silent-corruption machine: on 2026-08-05 exactly that halved
# alpha for a whole family of arms with no error anywhere. `_dt_lora_hinge.py`
# is shared by 14 legacy variants and `_dt_lora_a16.py` by the 2026-08-05
# 6-class arm; editing either would retroactively change what those
# checkpoints mean. So alpha=32 gets its own file, and
# utils/train/checkpoint.py writes r/alpha/scaling into the manifest and
# re-verifies them against `module.scaling` on every load.
#
# DELTAS FROM configs/model/groundingdino_swinb_dt_lora_a16.py
# ------------------------------------------------------------
#   lora_alpha        16 -> 32     spec §1 (alpha/r = 2, i.e. scaling 2.0)
#   lora_dropout     0.0 -> 0.05   spec §1
#   lora_lr        0.001 -> 1e-4   spec §3 (the trainer owns the LRs; this
#                                   field is documentation only)
#   use_checkpoint   False -> True         spec §4.1 gradient checkpointing
#   use_transformer_ckpt False -> True     spec §4.1 gradient checkpointing
#
# Everything else — architecture, `modelname = "dtgroundingdino"` (the
# training forward that computes the cls + L1 + GIoU loss dict with Hungarian
# matching), Swin-B backbone — is inherited unchanged.
#
# GRADIENT CHECKPOINTING IS NOT OPTIONAL HERE
# -------------------------------------------
# `transformer_for_adapter.py` wraps `encoder.fusion_layers[i]` (use_checkpoint)
# and `encoder.layers[i]` (use_transformer_ckpt) in
# `torch.utils.checkpoint.checkpoint(...)` WITHOUT `use_reentrant=`. Under
# torch 2.4 that defaults to the reentrant implementation, which produces NO
# gradient at all when none of the checkpointed function's *input tensors*
# require grad — which is exactly the situation here, since the image features
# entering encoder layer 0 come from a frozen backbone and a frozen input_proj.
# Every LoRA adapter inside the encoder would then train on a zero gradient,
# silently. train.py therefore forces `use_reentrant=False`
# (spec §4.1) and asserts after the first backward that encoder LoRA actually
# received gradient.
# The base config lives in the upstream checkout, whose location is
# UC_GDINO_REPO. Naming it absolutely here is what made this file
# unusable on any machine but the one it was written on.
# SLConfig resolves this against the directory holding *this* file, so the
# base travels with the repository. It used to be named inside the external
# checkout, which the fork never committed it to -- see
# configs/model/groundingdino_dt_lora_plain.py.
_base_ = ["groundingdino_dt_lora_plain.py"]

backbone = "swin_B_384_22k"

# Spec §4.1: gradient checkpointing on.
use_checkpoint = True
use_transformer_ckpt = True

# ── LoRA hyperparameters (spec §1) ──
# Restated rather than inherited so the effective values of a run are readable
# in one file. r=16 / alpha=32 -> scaling 2.0, verified on the live modules by
# models/shared/lora_spec_v6line.py::verify_lora_scaling before step 1.
lora_r = 16
lora_alpha = 32
lora_dropout = 0.05

# Documentation only — train.py builds three parameter groups
# from utils/train/training_schedule.py (§3: 1e-4 LoRA, 1e-5 heads, 1e-5
# sampling_offsets LoRA) and never reads these two fields.
lora_lr = 0.0001
lora_weight_decay = 0.01
