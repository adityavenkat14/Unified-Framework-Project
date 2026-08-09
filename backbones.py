"""
Backbone loader for the unified GOAL -> ABS/WCA -> LaZSL pipeline.

Two backbones are supported:
  - "openai": the standard pretrained CLIP weights (what ABS/WCA/LaZSL ship with).
  - "goal":   a GOAL-finetuned checkpoint. GOAL finetunes HuggingFace's
              `transformers.CLIPModel`, while this codebase (forked from
              OpenAI's own clip.py) uses a different state_dict key layout
              and a Parameter-based (not nn.Linear) projection head. This
              module converts GOAL's HF-style checkpoint into the key
              layout `clip/model.py`'s CLIP class expects, so it can be
              loaded as a drop-in backbone.

Usage:
    from backbones import load_backbone
    model, preprocess = load_backbone("goal", ckpt_path="goal_vitb16_docci.pt",
                                       openai_name="ViT-B/16", device="cuda")
"""
import torch
from clip import clip as openai_clip


def _convert_qkv(hf_sd, prefix_hf, prefix_openai, num_layers):
    """Fold HF's separate q_proj/k_proj/v_proj into OpenAI's fused in_proj_weight/bias,
    and rename the rest of each transformer block."""
    out = {}
    for i in range(num_layers):
        hf_l = f"{prefix_hf}.encoder.layers.{i}"
        oc_l = f"{prefix_openai}.{i}"

        q_w = hf_sd[f"{hf_l}.self_attn.q_proj.weight"]
        k_w = hf_sd[f"{hf_l}.self_attn.k_proj.weight"]
        v_w = hf_sd[f"{hf_l}.self_attn.v_proj.weight"]
        q_b = hf_sd[f"{hf_l}.self_attn.q_proj.bias"]
        k_b = hf_sd[f"{hf_l}.self_attn.k_proj.bias"]
        v_b = hf_sd[f"{hf_l}.self_attn.v_proj.bias"]
        out[f"{oc_l}.attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        out[f"{oc_l}.attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)

        out[f"{oc_l}.attn.out_proj.weight"] = hf_sd[f"{hf_l}.self_attn.out_proj.weight"]
        out[f"{oc_l}.attn.out_proj.bias"] = hf_sd[f"{hf_l}.self_attn.out_proj.bias"]

        out[f"{oc_l}.ln_1.weight"] = hf_sd[f"{hf_l}.layer_norm1.weight"]
        out[f"{oc_l}.ln_1.bias"] = hf_sd[f"{hf_l}.layer_norm1.bias"]
        out[f"{oc_l}.ln_2.weight"] = hf_sd[f"{hf_l}.layer_norm2.weight"]
        out[f"{oc_l}.ln_2.bias"] = hf_sd[f"{hf_l}.layer_norm2.bias"]

        out[f"{oc_l}.mlp.c_fc.weight"] = hf_sd[f"{hf_l}.mlp.fc1.weight"]
        out[f"{oc_l}.mlp.c_fc.bias"] = hf_sd[f"{hf_l}.mlp.fc1.bias"]
        out[f"{oc_l}.mlp.c_proj.weight"] = hf_sd[f"{hf_l}.mlp.fc2.weight"]
        out[f"{oc_l}.mlp.c_proj.bias"] = hf_sd[f"{hf_l}.mlp.fc2.bias"]
    return out


def convert_hf_to_openai(hf_sd: dict, num_vision_layers: int, num_text_layers: int,
                          context_length: int = 77) -> dict:
    """Convert a HuggingFace transformers.CLIPModel state_dict (what GOAL saves)
    into the key layout used by clip/model.py's CLIP class.

    GOAL finetunes with an extended text context length (248 tokens, to fit
    DOCCI's long dense captions) vs. the standard 77 this codebase's CLIP was
    built with. We truncate the positional embedding to the first
    `context_length` rows -- safe here since every prompt this pipeline
    generates (class names / short template sentences) is well under 77
    tokens; we're not trying to support 248-token inputs, just reuse GOAL's
    learned representations for short text.
    """
    out = {}

    # ---- vision tower ----
    out["visual.class_embedding"] = hf_sd["vision_model.embeddings.class_embedding"]
    out["visual.conv1.weight"] = hf_sd["vision_model.embeddings.patch_embedding.weight"]
    out["visual.positional_embedding"] = hf_sd["vision_model.embeddings.position_embedding.weight"]
    out["visual.ln_pre.weight"] = hf_sd["vision_model.pre_layrnorm.weight"]
    out["visual.ln_pre.bias"] = hf_sd["vision_model.pre_layrnorm.bias"]
    out["visual.ln_post.weight"] = hf_sd["vision_model.post_layernorm.weight"]
    out["visual.ln_post.bias"] = hf_sd["vision_model.post_layernorm.bias"]
    # HF: nn.Linear(width, embed_dim), no bias -> weight is [embed_dim, width]
    # OpenAI: nn.Parameter [width, embed_dim] -> transpose
    out["visual.proj"] = hf_sd["visual_projection.weight"].t().contiguous()
    out.update(_convert_qkv(hf_sd, "vision_model", "visual.transformer.resblocks", num_vision_layers))

    # ---- text tower ----
    out["token_embedding.weight"] = hf_sd["text_model.embeddings.token_embedding.weight"]
    pos_emb = hf_sd["text_model.embeddings.position_embedding.weight"]
    if pos_emb.shape[0] != context_length:
        print(f"[backbones] GOAL checkpoint has context_length={pos_emb.shape[0]}, "
              f"this model uses {context_length} -- truncating positional_embedding "
              f"to the first {context_length} rows (fine for short prompts).")
        pos_emb = pos_emb[:context_length]
    out["positional_embedding"] = pos_emb
    out["ln_final.weight"] = hf_sd["text_model.final_layer_norm.weight"]
    out["ln_final.bias"] = hf_sd["text_model.final_layer_norm.bias"]
    out["text_projection"] = hf_sd["text_projection.weight"].t().contiguous()
    out.update(_convert_qkv(hf_sd, "text_model", "transformer.resblocks", num_text_layers))

    if "logit_scale" in hf_sd:
        out["logit_scale"] = hf_sd["logit_scale"]

    return out


def load_backbone(kind: str, openai_name: str = "ViT-B/16", ckpt_path: str = None,
                   device: str = "cuda", strict: bool = False):
    """
    kind: "openai" (vanilla pretrained CLIP) or "goal" (GOAL-finetuned checkpoint).
    ckpt_path: required when kind == "goal"; path to the .pt file GOAL saved
               (a plain HF CLIPModel state_dict, see goal.py's save code).
    """
    model, preprocess = openai_clip.load(openai_name, device=device, jit=False)

    if kind == "openai":
        return model, preprocess

    if kind == "goal":
        assert ckpt_path is not None, "ckpt_path is required for the 'goal' backbone"
        hf_sd = torch.load(ckpt_path, map_location="cpu")
        num_vision_layers = model.visual.transformer.layers
        num_text_layers = model.transformer.layers
        context_length = model.positional_embedding.shape[0]
        converted = convert_hf_to_openai(hf_sd, num_vision_layers, num_text_layers, context_length)

        missing, unexpected = model.load_state_dict(converted, strict=strict)
        if missing:
            print(f"[backbones] {len(missing)} keys missing after GOAL load "
                  f"(expected: buffers/attn_mask, etc.) e.g. {missing[:5]}")
        if unexpected:
            print(f"[backbones] {len(unexpected)} unexpected keys ignored, e.g. {unexpected[:5]}")

        # Hard fail on any resblocks (transformer layer) key mismatch -- these should
        # NEVER show up as missing/unexpected if the conversion worked. This is the
        # exact class of bug that silently no-op'd the transformer block weights before
        # (a doubled "resblocks.resblocks" path), so treat it as fatal, not a warning.
        bad_missing = [k for k in missing if "resblocks" in k]
        bad_unexpected = [k for k in unexpected if "resblocks" in k]
        assert not bad_missing and not bad_unexpected, (
            f"GOAL weight conversion has a key mismatch in the transformer blocks -- "
            f"this means the transformer layers did NOT actually get GOAL's weights. "
            f"missing e.g. {bad_missing[:3]}, unexpected e.g. {bad_unexpected[:3]}"
        )

        # Sanity check on a couple of tensors so a silent shape mismatch doesn't
        # slip through as a "successful" load.
        assert model.visual.conv1.weight.shape == converted["visual.conv1.weight"].shape
        return model.to(device), preprocess

    raise ValueError(f"Unknown backbone kind: {kind}")
