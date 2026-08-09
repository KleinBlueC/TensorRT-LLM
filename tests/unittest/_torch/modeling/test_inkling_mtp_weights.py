# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""The MTP draft chain's checkpoint keys, checked against the real weights.

A key set derived by reading code is a guess until it is compared with a
checkpoint. These tests take the actual ``*.index.json`` of the two shipped
releases and require an EXACT match -- no missing keys (a silently
unloaded draft tensor is wrong numbers, not a crash) and no unaccounted ones
(a tensor nobody claims means the derivation is incomplete).

Skipped, not failed, when the weights are not mounted: the derivation is still
covered by the shape assertions below.
"""

import glob
import json
import os

import pytest

from tensorrt_llm._torch.configs.inkling import InklingConfig
from tensorrt_llm._torch.models.checkpoints.hf.inkling_weight_mapper import (
    inkling_expected_mtp_keys,
)

_HF_ROOT = os.environ.get(
    "INKLING_HF_ROOT",
    "/lustre/fs1/portfolios/coreai/projects/coreai_comparch_trtllm/users/kleinc/hf_data",
)
_CHECKPOINTS = ["Inkling-NVFP4-full", "Inkling-small-NVFP4"]


def _load(ckpt):
    path = os.path.join(_HF_ROOT, ckpt)
    index = glob.glob(os.path.join(path, "*.index.json"))
    cfg_path = os.path.join(path, "config.json")
    if not index or not os.path.exists(cfg_path):
        pytest.skip(f"{ckpt} not mounted")
    with open(index[0]) as f:
        keys = set(json.load(f)["weight_map"])
    with open(cfg_path) as f:
        raw = json.load(f)
    cfg = InklingConfig(text_config=raw.get("text_config") or {}, mtp_config=raw.get("mtp_config"))
    return keys, cfg, raw


@pytest.mark.parametrize("ckpt", _CHECKPOINTS)
def test_mtp_keys_match_the_checkpoint_exactly(ckpt):
    """Every ``model.mtp.*`` key is claimed, and every claimed key exists."""
    keys, cfg, raw = _load(ckpt)
    depths = raw["mtp_config"]["num_nextn_predict_layers"]
    expected = inkling_expected_mtp_keys(cfg.text_config, depths)
    actual = {k for k in keys if k.startswith("model.mtp.")}

    assert not (expected - actual), (
        f"{ckpt}: derived keys the checkpoint does not have "
        f"(sample: {sorted(expected - actual)[:5]})"
    )
    assert not (actual - expected), (
        f"{ckpt}: checkpoint keys nobody claims -- the derivation is "
        f"incomplete (sample: {sorted(actual - expected)[:5]})"
    )


@pytest.mark.parametrize("ckpt", _CHECKPOINTS)
def test_chain_depth_and_geometry_come_from_the_checkpoint(ckpt):
    """The declared depth matches the weights, and the bandedness is read."""
    keys, cfg, raw = _load(ckpt)
    depths = raw["mtp_config"]["num_nextn_predict_layers"]
    present = {
        int(k.split("model.mtp.layers.")[1].split(".")[0])
        for k in keys
        if k.startswith("model.mtp.layers.")
    }
    assert present == set(range(depths)), (
        f"{ckpt}: declared {depths} depths, checkpoint has {sorted(present)}"
    )
    assert cfg.text_config.mtp_local_layer_ids == raw["mtp_config"]["local_layer_ids"]


@pytest.mark.parametrize("ckpt", _CHECKPOINTS)
def test_every_depth_is_dense_not_moe(ckpt):
    """MTP blocks use the dense MLP, so no expert tensors may appear.

    SGLang forces the dense MLP for every depth. If a checkpoint ever ships
    expert tensors here, the derivation above would silently drop them, so
    assert the assumption rather than rely on it.
    """
    keys, _, _ = _load(ckpt)
    experts = {k for k in keys if k.startswith("model.mtp.") and (".experts." in k or "gate." in k)}
    assert not experts, f"{ckpt}: unexpected MoE tensors in the draft chain: {sorted(experts)[:5]}"


def test_derivation_scales_with_depth_without_a_checkpoint():
    """The key count is linear in depth and includes the three fold tensors."""
    cfg = InklingConfig(
        text_config={}, mtp_config={"num_nextn_predict_layers": 8, "local_layer_ids": [0, 2]}
    )
    one = inkling_expected_mtp_keys(cfg.text_config, 1)
    two = inkling_expected_mtp_keys(cfg.text_config, 2)
    assert len(two) == 2 * len(one)
    for name in ("embed_norm.weight", "hidden_norm.weight", "input_proj.weight"):
        assert f"model.mtp.layers.0.{name}" in one
