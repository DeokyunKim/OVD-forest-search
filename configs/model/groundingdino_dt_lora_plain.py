# ─────────────────────────────────────────────────────────────────────────────
# NEW (additive) model config: plain/vanilla LoRA on GroundingDINO attention
# layers, base model fully frozen. Built for the "Ground Without Forgetting"
# representation-drift analysis (NeurIPS 2026 VLM4RWD workshop): we need a
# LoRA-adapted checkpoint that is architecturally DIFFERENT from ZiRa
# (GroundingDINO_SwinT_OGC_rep.py, dual-branch / Zero-Interference Loss) and
# from DitHub's per-class LoRA pool (GroundingDINO_SwinT_OGC_dt_dithub.py,
# continual-learning LinearPool with per-class A / shared B / task routing).
#
# Architecture hyperparameters below are copied verbatim from the existing
# GroundingDINO_SwinT_OGC_dt_finetuning.py (plain "dtgroundingdino" model,
# no CET/adapter/prompt-tuning machinery, no TaskMemory dependency), with:
#   - freeze_all = True             (base frozen; the trainer unfreezes
#                                     only the LoRA A/B params it adds)
#   - use_checkpoint / use_transformer_ckpt = True (gradient checkpointing,
#                                     to keep peak memory within the shared-GPU
#                                     budget of a single card)
#   - lora_* fields (read by the trainer; NOT
#                                     consumed by build_model itself) documenting
#                                     the exact LoRA hyperparameters used, matching
#                                     the repo's own DitHub defaults
#                                     (train_dithub_odinw13.sh: r=16, alpha=8,
#                                     dropout=0.0, lora_lr=0.001, wd=0.01).
# ─────────────────────────────────────────────────────────────────────────────
batch_size = 1
modelname = "dtgroundingdino"
backbone = "swin_T_224_1k"
position_embedding = "sine"
pe_temperatureH = 20
pe_temperatureW = 20
return_interm_indices = [1, 2, 3]
backbone_freeze_keywords = None
enc_layers = 6
dec_layers = 6
pre_norm = False
dim_feedforward = 2048
hidden_dim = 256
dropout = 0.0
nheads = 8
num_queries = 900
query_dim = 4
num_patterns = 0
num_feature_levels = 4
enc_n_points = 4
dec_n_points = 4
two_stage_type = "standard"
two_stage_bbox_embed_share = False
two_stage_class_embed_share = False
transformer_activation = "relu"
dec_pred_bbox_embed_share = True
dn_box_noise_scale = 1.0
dn_label_noise_ratio = 0.5
dn_label_coef = 1.0
dn_bbox_coef = 1.0
embed_init_tgt = True
dn_labelbook_size = 2000
max_text_len = 256
text_encoder_type = "bert-base-uncased"
use_text_enhancer = True
use_fusion_layer = True
# Gradient checkpointing disabled: LoRA is only ever attached to
# transformer.encoder.text_layers[*].self_attn (see load_lora_model() in
# the trainer's LoRA placement), which is NEVER wrapped in
# torch.utils.checkpoint (only encoder.fusion_layers and encoder.layers are,
# and get_lora_modules() skips those entirely since their weights are frozen
# by freeze_all=True before LoRA is applied — verified empirically: 0 LoRA
# params inside checkpointed modules). Checkpointing those frozen-only
# modules gives no memory benefit for a LoRA-only backward and only adds
# recompute overhead + spurious "None of the inputs have requires_grad"
# warnings, so it is kept off here.
use_checkpoint = False
use_transformer_ckpt = False
use_text_cross_attention = True
text_dropout = 0.0
fusion_dropout = 0.0
fusion_droppath = 0.1
sub_sentence_present = True

# train
aux_loss = True
# False, exactly like GroundingDINO_SwinT_OGC_dt_dithub.py ("we freeze anyways
# all when we instantiate the loras"): get_lora_modules() only considers
# Linear layers whose weight.requires_grad is still True at scan time (see
# models/gdino/lora/utils.py), so freeze_all=True here would
# silently shrink LoRA's footprint to only the freshly-created
# custom-attention Q/K/V/O layers (whose requires_grad defaults to True on
# creation), excluding the encoder's fusion_layers / deformable self-attn+FFN
# Linear layers that DitHub's own default DOES cover. train_lora_forest_
# persons.py freezes every non-LoRA param explicitly (mark_only_lora_as_
# trainable) AFTER LoRA is applied, so the end state is still base-frozen /
# adapter-only regardless of this flag — it only affects which layers
# get_lora_modules is allowed to see.
freeze_all = False

# test
select_box_nums_for_evaluation = 200

# adapter (disabled — this config uses plain LoRA instead, applied externally
# by the trainer through models/gdino/lora)
use_adapter = False
use_self_kd = False
encoder_gate_base_scale = 0.1
decoder_gate_base_scale = 0.1

# task agnostic
use_add_names = True
use_learned_names = False

# cet (disabled)
use_cet = False
cet_middle_dim = 1024
use_prompt_memory = False
use_prompt_memory_output = False
cet_type = "Adapter"

# zero loss
use_zero_inter_loss = False
loss_adapter_weight = 0.1

# moe
num_experts = 1
num_topk_experts = 1

# bert
use_bert_tuning = False

# linear probing
use_cls_linear = False

# prompt tuning
use_prompt_tuning = False

# project adapter
use_project_adapter = False
use_zero_inter_loss_for_conv = False

# -- Plain LoRA hyperparameters (read by the trainer) --
# Same rank/alpha/dropout convention as DitHub's default
# (train_dithub_odinw13.sh: --lora-r 16 --lora-alpha 8 --lora-dropout 0.0
# --lora-out-min 128 --lora-lr 0.001 --lora-weight-decay 0.01), but applied as
# a single vanilla lora.Linear adapter per targeted layer (no per-class pool,
# no TaskMemory, no cross-task B merging).
lora_r = 16
lora_alpha = 8
lora_dropout = 0.0
lora_out_min = 128
lora_lr = 0.001
lora_weight_decay = 0.01
