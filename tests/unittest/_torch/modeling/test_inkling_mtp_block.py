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
"""The derived config that makes a decoder layer behave as an MTP block.

The draft block reuses ``InklingDecoderLayer`` unchanged: the layer asks its
config which layers are dense and which are banded, and ``mtp_block_config``
answers with the CHAIN's geometry instead of the trunk's.

That indirection is the whole risk. If a banded depth were built with the
trunk's window, the checkpoint's ``rel_logits_proj`` for that depth -- trained
at the head's window -- would be applied at the wrong extent: wrong numbers,
no crash, nothing at runtime to notice it. So the derived config is asserted
field by field rather than trusted.
"""

import pytest

from tensorrt_llm._torch.configs.inkling import InklingConfig

# As shipped in both Inkling-NVFP4-full and Inkling-small-NVFP4.
_CKPT_MTP = {
    "num_nextn_predict_layers": 8,
    "chain_hidden_post_norm": False,
    "local_layer_ids": [0, 2, 4, 5, 6, 7],
}

_TRUNK = {
    "num_hidden_layers": 66,
    "dense_mlp_idx": 2,
    "local_layer_ids": [1, 3, 5],  # deliberately different from the chain's
    "sliding_window_size": 512,
    "num_attention_heads": 48,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "swa_num_attention_heads": 64,
    "swa_num_key_value_heads": 16,
    "swa_head_dim": 64,
}


def _text(**overrides):
    cfg = {**_TRUNK, **overrides}
    return InklingConfig(text_config=cfg, mtp_config=dict(_CKPT_MTP)).text_config


@pytest.mark.parametrize("depth", range(8))
def test_every_depth_is_dense(depth):
    """MTP blocks always use the dense MLP, at any depth index.

    Depth 5 sits well past the trunk's ``dense_mlp_idx`` of 2, so without the
    override it would be built as MoE and then fail to find expert weights the
    checkpoint does not have.
    """
    block = _text().mtp_block_config(depth)
    assert block.is_dense_layer(depth) is True


@pytest.mark.parametrize(
    "depth,banded",
    [(0, True), (1, False), (2, True), (3, False), (4, True), (5, True), (6, True), (7, True)],
)
def test_bandedness_follows_the_chain_not_the_trunk(depth, banded):
    """The trunk's banded layers are [1,3,5]; the chain's are [0,2,4,5,6,7].

    Depths 1 and 3 are banded in the trunk and global in the chain, so a config
    that leaked the trunk's list would get exactly these two backwards.
    """
    block = _text().mtp_block_config(depth)
    assert block.is_local_layer(depth) is banded


def test_banded_depth_uses_the_chain_window_and_heads():
    text = _text()
    block = text.mtp_block_config(0)  # banded
    assert block.is_local_layer(0)
    assert block.layer_window(0) == 512  # defaults to the trunk window
    assert block.layer_num_heads(0) == 64
    assert block.layer_num_kv_heads(0) == 16
    assert block.layer_head_dim(0) == 64


def test_explicit_chain_extent_reaches_the_block():
    """An overridden ``local_extent`` must land on the layer's window."""
    cfg = InklingConfig(text_config=dict(_TRUNK), mtp_config={**_CKPT_MTP, "local_extent": 2048})
    block = cfg.text_config.mtp_block_config(0)
    assert block.layer_window(0) == 2048


def test_global_depth_keeps_the_full_attention_geometry():
    """A global depth must not be given the SWA head counts."""
    block = _text().mtp_block_config(1)  # global in the chain
    assert block.is_local_layer(1) is False
    assert block.layer_window(1) is None
    assert block.layer_num_heads(1) == 48
    assert block.layer_num_kv_heads(1) == 8
    assert block.layer_head_dim(1) == 128


def test_deriving_does_not_mutate_the_trunk_config():
    """Building a block must not disturb the trunk the model is still using."""
    text = _text()
    before = (
        text.dense_mlp_idx,
        list(text.local_layer_ids),
        text.sliding_window_size,
        text.swa_num_attention_heads,
    )
    for depth in range(8):
        text.mtp_block_config(depth)
    after = (
        text.dense_mlp_idx,
        list(text.local_layer_ids),
        text.sliding_window_size,
        text.swa_num_attention_heads,
    )
    assert before == after


# --- the draft chain's own KV cache geometry -------------------------------
# The draft chain gets a SEPARATE cache manager, built with num_layers = the
# number of built depths. KVCacheManagerV2 asserts len(num_kv_heads) equals
# that count, so handing it the trunk's per-layer list is an outright failure
# -- which is how this was found, several minutes into a 4-GPU job.


@pytest.mark.parametrize("depths", [1, 3, 8])
def test_draft_kv_head_list_has_one_entry_per_built_depth(depths):
    """Length must follow the chain the runtime built, not the checkpoint's 8.

    A server asking for 3 draft tokens builds 3 blocks; the manager is created
    with num_layers=3 and asserts the list matches.
    """
    assert len(_text().mtp_num_kv_heads_per_layer(depths)) == depths


def test_draft_kv_heads_follow_the_chain_banded_pattern():
    """Chain banded depths are [0,2,4,5,6,7]; the trunk's are [1,3,5].

    On the full checkpoint banded layers carry 16 KV heads and global ones 8,
    so a slice of the trunk's list would size depths 1 and 3 for 16 heads and
    depth 1 for 8 -- pages allocated against the wrong head count, with nothing
    at runtime to report it.
    """
    assert _text().mtp_num_kv_heads_per_layer(8) == [16, 8, 16, 8, 16, 16, 16, 16]


def test_uniform_checkpoint_gives_a_uniform_draft_list():
    """Inkling-small has swa_num_key_value_heads == num_key_value_heads == 8.

    Banded and global depths then agree, and the list must simply be uniform
    rather than accidentally picking up the trunk's 16 from a stale default.
    """
    text = _text(num_key_value_heads=8, swa_num_key_value_heads=8)
    assert text.mtp_num_kv_heads_per_layer(4) == [8, 8, 8, 8]


def test_chain_swa_geometry_can_differ_from_the_trunk():
    text = _text()
    text.mtp_swa_num_attention_heads = 32
    text.mtp_swa_num_key_value_heads = 4
    text.mtp_swa_head_dim = 256
    block = text.mtp_block_config(0)
    assert block.layer_num_heads(0) == 32
    assert block.layer_num_kv_heads(0) == 4
    assert block.layer_head_dim(0) == 256


# --- the two indices a draft block lives under -----------------------------
# Geometry is indexed by CHAIN depth (0..7); the KV cache is keyed by GLOBAL
# layer index (trunk layers + depth), because the draft manager's layer offsets
# are global. Folding one into the other is a KeyError in the first draft
# forward, several minutes into a multi-GPU run.


def test_block_config_answers_for_the_global_index():
    """Built with the global index, the config must still say dense and banded."""
    text = _text()
    trunk = text.num_hidden_layers
    for depth in range(8):
        cfg = text.mtp_block_config(depth, trunk + depth)
        assert cfg.is_dense_layer(trunk + depth) is True
        assert cfg.is_local_layer(trunk + depth) is text.is_mtp_local_depth(depth)


def test_global_index_geometry_matches_the_chain_depth_geometry():
    """Same block, two indices, identical answers.

    The window and head counts come from the chain depth; only the index the
    layer is addressed by changes. If these ever diverge, a banded depth would
    be built with the wrong window and its rel_logits_proj -- trained at the
    head's window -- would be applied at the wrong extent, with no crash.
    """
    text = _text()
    trunk = text.num_hidden_layers
    for depth in range(8):
        by_depth = text.mtp_block_config(depth)
        by_global = text.mtp_block_config(depth, trunk + depth)
        assert by_global.layer_window(trunk + depth) == by_depth.layer_window(depth)
        assert by_global.layer_num_heads(trunk + depth) == by_depth.layer_num_heads(depth)
        assert by_global.layer_num_kv_heads(trunk + depth) == by_depth.layer_num_kv_heads(depth)
