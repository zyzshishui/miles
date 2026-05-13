import os
import re

import torch


def _should_apply_ape_hotfix_mirror():
    return (
        os.getenv("SGLANG_DSV4_MODE", "2604") != "2604"
        and os.getenv("SGLANG_ENABLE_APE_HOTFIX", "1").lower() in ("1", "true")
    )


def _apply_ape_hotfix_mirror(param):
    # Mirror SGLang Compressor.apply_ape_hotfix. The default 2604 path marks APE
    # converted without reordering data, so this is only for non-2604 checkpoints.
    assert param.shape[0] == 4
    ape = torch.chunk(param, 2, dim=-1)
    return torch.cat([ape[1], ape[0]], dim=-1).view_as(param).contiguous()


def convert_deepseekv4_to_hf(args, name, param):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    if name == "module.module.decoder.hc_head_params.hc_head_fn":
        return [("model.hc_head_fn", param)]
    if name == "module.module.decoder.hc_head_params.hc_head_base":
        return [("model.hc_head_base", param)]
    if name == "module.module.decoder.hc_head_params.hc_head_scale":
        return [("model.hc_head_scale", param)]

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        if rest == "hc_attn_fn":
            return [(f"model.layers.{layer_idx}.hc_attn_fn", param)]
        elif rest == "hc_attn_base":
            return [(f"model.layers.{layer_idx}.hc_attn_base", param)]
        elif rest == "hc_attn_scale":
            return [(f"model.layers.{layer_idx}.hc_attn_scale", param)]
        elif rest == "hc_ffn_fn":
            return [(f"model.layers.{layer_idx}.hc_ffn_fn", param)]
        elif rest == "hc_ffn_base":
            return [(f"model.layers.{layer_idx}.hc_ffn_base", param)]
        elif rest == "hc_ffn_scale":
            return [(f"model.layers.{layer_idx}.hc_ffn_scale", param)]

        expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
        match = re.match(expert_pattern, rest)
        if match:
            rest, expert_idx = match.groups()
            if rest == "linear_fc1":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2":
                return [(f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param)]
            else:
                raise ValueError(f"Unknown expert parameter name: {name}")

        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [(f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight", param)]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

        if rest == "self_attention.wq_a.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wq_a.weight", param)]
        elif rest == "self_attention.q_norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.wq_b.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wq_b.weight", param)]
        elif rest == "self_attention.wkv.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wkv.weight", param)]
        elif rest == "self_attention.kv_norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_norm.weight", param)]
        elif rest == "self_attention.wo_a.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wo_a.weight", param)]
        elif rest == "self_attention.wo_b.weight":
            return [(f"model.layers.{layer_idx}.self_attn.wo_b.weight", param)]
        elif rest == "self_attention.attn_sink":
            return [(f"model.layers.{layer_idx}.self_attn.attn_sink", param)]

        elif rest == "self_attention.compressor.ape":
            if param.shape[0] == 4 and _should_apply_ape_hotfix_mirror():
                param = _apply_ape_hotfix_mirror(param)
            return [(f"model.layers.{layer_idx}.self_attn.compressor.ape", param)]
        elif rest == "self_attention.compressor.wkv.weight":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.wkv.weight", param)]
        elif rest == "self_attention.compressor.wgate.weight":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.wgate.weight", param)]
        elif rest == "self_attention.compressor.norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.compressor.norm.weight", param)]

        elif rest == "self_attention.indexer.linear_wq_b.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wq_b.weight", param)]
        elif rest == "self_attention.indexer.linear_wk.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.wk.weight", param)]
        elif rest == "self_attention.indexer.k_norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.weight", param)]
        elif rest == "self_attention.indexer.k_norm.bias":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.k_norm.bias", param)]
        elif rest == "self_attention.indexer.linear_weights_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.weights_proj.weight", param)]

        elif rest == "self_attention.indexer.compressor.ape":
            if param.shape[0] == 4 and _should_apply_ape_hotfix_mirror():
                param = _apply_ape_hotfix_mirror(param)
            return [(f"model.layers.{layer_idx}.self_attn.indexer.compressor.ape", param)]
        elif rest == "self_attention.indexer.compressor.wkv.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.compressor.wkv.weight", param)]
        elif rest == "self_attention.indexer.compressor.wgate.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.compressor.wgate.weight", param)]
        elif rest == "self_attention.indexer.compressor.norm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.indexer.compressor.norm.weight", param)]

        elif rest == "input_layernorm.weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]
        elif rest == "mlp.router.tid2eid":
            return [(f"model.layers.{layer_idx}.mlp.topk.tid2eid", param)]

        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
