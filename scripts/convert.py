#!/usr/bin/env python
"""Convert PP-DocLayoutV3 safetensors -> GGUF (F32, BN folded into convs).

Tensor renaming produces compact names (<64 chars, ggml limit).
Conv+BN pairs are folded: w' = w * g/sqrt(v+eps), b' = beta - g*m/sqrt(v+eps).
"""
import re
import sys
import numpy as np
from safetensors import safe_open
import gguf

SRC = "PP-DocLayoutV3_safetensors/model.safetensors"
DST = "model-f32.gguf"
BN_EPS = 1e-5

RENAMES = [
    (r"^model\.backbone\.model\.embedder\.(stem[1234ab]+)$", r"\1"),
    (r"^model\.backbone\.model\.encoder\.stages\.(\d)\.downsample$", r"st\1.ds"),
    (r"^model\.backbone\.model\.encoder\.stages\.(\d)\.blocks\.(\d)\.layers\.(\d)\.conv1$", r"st\1.b\2.l\3.c1"),
    (r"^model\.backbone\.model\.encoder\.stages\.(\d)\.blocks\.(\d)\.layers\.(\d)\.conv2$", r"st\1.b\2.l\3.c2"),
    (r"^model\.backbone\.model\.encoder\.stages\.(\d)\.blocks\.(\d)\.layers\.(\d)$", r"st\1.b\2.l\3"),
    (r"^model\.backbone\.model\.encoder\.stages\.(\d)\.blocks\.(\d)\.aggregation\.(\d)$", r"st\1.b\2.agg\3"),
    (r"^model\.encoder_input_proj\.(\d)$", r"enc_in\1"),
    (r"^model\.decoder_input_proj\.(\d)$", r"dec_in\1"),
    (r"^model\.encoder\.encoder\.0\.layers\.0\.self_attn\.q_proj$", r"aifi.q"),
    (r"^model\.encoder\.encoder\.0\.layers\.0\.self_attn\.k_proj$", r"aifi.k"),
    (r"^model\.encoder\.encoder\.0\.layers\.0\.self_attn\.v_proj$", r"aifi.v"),
    (r"^model\.encoder\.encoder\.0\.layers\.0\.self_attn\.out_proj$", r"aifi.o"),
    (r"^model\.encoder\.encoder\.0\.layers\.0\.self_attn_layer_norm$", r"aifi.ln1"),
    (r"^model\.encoder\.encoder\.0\.layers\.0\.final_layer_norm$", r"aifi.ln2"),
    (r"^model\.encoder\.encoder\.0\.layers\.0\.fc1$", r"aifi.fc1"),
    (r"^model\.encoder\.encoder\.0\.layers\.0\.fc2$", r"aifi.fc2"),
    (r"^model\.encoder\.lateral_convs\.(\d)$", r"lat\1"),
    (r"^model\.encoder\.fpn_blocks\.(\d)\.bottlenecks\.(\d)\.conv([12])$", r"fpn\1.bn\2.c\3"),
    (r"^model\.encoder\.fpn_blocks\.(\d)\.conv([123])$", r"fpn\1.c\2"),
    (r"^model\.encoder\.downsample_convs\.(\d)$", r"pand\1"),
    (r"^model\.encoder\.pan_blocks\.(\d)\.bottlenecks\.(\d)\.conv([12])$", r"pan\1.bn\2.c\3"),
    (r"^model\.encoder\.pan_blocks\.(\d)\.conv([123])$", r"pan\1.c\2"),
    (r"^model\.encoder\.mask_feature_head\.scale_heads\.(\d)\.layers\.(\d)$", r"mh\1.\2"),
    (r"^model\.encoder\.mask_feature_head\.output_conv$", r"mh.out"),
    (r"^model\.encoder\.encoder_mask_lateral$", r"mlat"),
    (r"^model\.encoder\.encoder_mask_output\.base_conv$", r"mout.base"),
    (r"^model\.encoder\.encoder_mask_output\.conv$", r"mout.conv"),
    (r"^model\.enc_output\.0$", r"enc_out.l"),
    (r"^model\.enc_output\.1$", r"enc_out.ln"),
    (r"^model\.enc_score_head$", r"score_head"),
    (r"^model\.enc_bbox_head\.layers\.(\d)$", r"bbox_head.\1"),
    (r"^model\.decoder\.layers\.(\d)\.self_attn\.q_proj$", r"dec\1.sa.q"),
    (r"^model\.decoder\.layers\.(\d)\.self_attn\.k_proj$", r"dec\1.sa.k"),
    (r"^model\.decoder\.layers\.(\d)\.self_attn\.v_proj$", r"dec\1.sa.v"),
    (r"^model\.decoder\.layers\.(\d)\.self_attn\.out_proj$", r"dec\1.sa.o"),
    (r"^model\.decoder\.layers\.(\d)\.self_attn_layer_norm$", r"dec\1.ln1"),
    (r"^model\.decoder\.layers\.(\d)\.encoder_attn\.sampling_offsets$", r"dec\1.ca.so"),
    (r"^model\.decoder\.layers\.(\d)\.encoder_attn\.attention_weights$", r"dec\1.ca.aw"),
    (r"^model\.decoder\.layers\.(\d)\.encoder_attn\.value_proj$", r"dec\1.ca.vp"),
    (r"^model\.decoder\.layers\.(\d)\.encoder_attn\.output_proj$", r"dec\1.ca.op"),
    (r"^model\.decoder\.layers\.(\d)\.encoder_attn_layer_norm$", r"dec\1.ln2"),
    (r"^model\.decoder\.layers\.(\d)\.final_layer_norm$", r"dec\1.ln3"),
    (r"^model\.decoder\.layers\.(\d)\.fc1$", r"dec\1.fc1"),
    (r"^model\.decoder\.layers\.(\d)\.fc2$", r"dec\1.fc2"),
    (r"^model\.decoder\.query_pos_head\.layers\.(\d)$", r"qpos.\1"),
    (r"^model\.decoder_norm$", r"dec_norm"),
    (r"^model\.decoder_order_head\.(\d)$", r"order\1"),
    (r"^model\.decoder_global_pointer\.dense$", r"gp"),
    (r"^model\.mask_query_head\.layers\.(\d)$", r"mq.\1"),
]


def rename(prefix: str) -> str:
    for pat, rep in RENAMES:
        if re.match(pat, prefix):
            return re.sub(pat, rep, prefix)
    raise KeyError(f"no rename rule for: {prefix}")


def main():
    f = safe_open(SRC, "np")
    keys = set(f.keys())

    # group by module prefix (strip last component: weight/bias/running_*)
    modules = {}
    for k in keys:
        prefix, leaf = k.rsplit(".", 1)
        modules.setdefault(prefix, {})[leaf] = k

    out = {}  # new_name -> np.array
    consumed = set()

    # 1) conv + BN folding. BN module has running_mean.
    for prefix, leaves in sorted(modules.items()):
        if "running_mean" not in leaves:
            continue
        # locate sibling conv module
        for conv_suffix, bn_suffix in ((".convolution", ".normalization"),
                                       (".conv", ".norm"),
                                       (".0", ".1")):
            if prefix.endswith(bn_suffix):
                conv_prefix = prefix[: -len(bn_suffix)] + conv_suffix
                if conv_prefix in modules:
                    break
        else:
            raise KeyError(f"BN without conv: {prefix}")
        unit = prefix[: -len(bn_suffix)]
        w = f.get_tensor(modules[conv_prefix]["weight"]).astype(np.float64)
        g = f.get_tensor(leaves["weight"]).astype(np.float64)
        b = f.get_tensor(leaves["bias"]).astype(np.float64)
        m = f.get_tensor(leaves["running_mean"]).astype(np.float64)
        v = f.get_tensor(leaves["running_var"]).astype(np.float64)
        scale = g / np.sqrt(v + BN_EPS)
        w_f = (w * scale[:, None, None, None]).astype(np.float32)
        b_f = (b - m * scale).astype(np.float32)
        name = rename(unit)
        out[name + ".weight"] = w_f
        out[name + ".bias"] = b_f
        for kk in leaves.values():
            consumed.add(kk)
        for kk in modules[conv_prefix].values():
            consumed.add(kk)

    # 2) everything else verbatim (skip denoising embed, training-only)
    for k in sorted(keys - consumed):
        if k.startswith("model.denoising_class_embed"):
            continue
        prefix, leaf = k.rsplit(".", 1)
        name = rename(prefix) + "." + leaf
        out[name] = f.get_tensor(k).astype(np.float32)

    for n in out:
        assert len(n) < 64, f"name too long: {n}"

    w = gguf.GGUFWriter(DST, "pp-doclayout-v3")
    w.add_description("PP-DocLayoutV3 F32, BN folded")
    w.add_uint32("ppdlv3.d_model", 256)
    w.add_uint32("ppdlv3.decoder_layers", 6)
    w.add_uint32("ppdlv3.num_queries", 300)
    w.add_uint32("ppdlv3.num_classes", 25)
    for name in sorted(out):
        w.add_tensor(name, np.ascontiguousarray(out[name]))
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_tensors_to_file()
    w.close()
    print(f"wrote {DST}: {len(out)} tensors")


if __name__ == "__main__":
    main()
