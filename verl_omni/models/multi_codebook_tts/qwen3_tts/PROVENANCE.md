# Qwen3-TTS Source Provenance

The two model files in this directory (`configuration_qwen3_tts.py` and
`modeling_qwen3_tts.py`) are vendored from the upstream Qwen3-TTS torch
source tree.

## Source

- **Upstream repo**: https://github.com/QwenLM/Qwen3-TTS
- **Local clone path**: `/lustre/fsw/portfolios/coreai/users/yuekaiz/tts/Qwen3-TTS`
- **Source subpath**: `qwen_tts/core/models/{configuration_qwen3_tts,modeling_qwen3_tts}.py`
- **Commit at vendoring**: `022e286b98fbec7e1e916cb940cdf532cd9f488e`
- **Vendored on**: 2026-05-22

## Local edits

The vendored files are not byte-identical to the upstream source. The
following targeted edits are applied to make them work as a self-contained
training-time dependency inside `verl-omni`:

### `configuration_qwen3_tts.py`

- `Qwen3TTSConfig.__init__` ends with a loop that promotes the standard
  transformer hyperparameters (`hidden_size`, `num_attention_heads`,
  `num_key_value_heads`, `num_hidden_layers`, `vocab_size`,
  `intermediate_size`, `rms_norm_eps`, `rope_theta`,
  `max_position_embeddings`, `head_dim`, `attention_bias`,
  `attention_dropout`) from `talker_config` onto the top-level config
  when they are missing. verl's FSDP / monkey-patch paths read these
  fields off the top-level config directly.
- Added `text_config` `@property` that aliases to `talker_config`. verl's
  VLM-style fallback inspects `config.text_config.hidden_size`; the alias
  makes that path resolve.

Previously these patches lived in the top-level
`qwen3_tts_autoregister.py` (now deleted in M5) and were applied via
`Qwen3TTSConfig.__init__ = _patched_init`. Folding them into the vendored
class means there is no monkey-patcher in the import chain and no
implicit Ray-worker-setup-hook ordering requirement.

### `modeling_qwen3_tts.py`

- Removed the relative import `from ...inference.qwen3_tts_tokenizer
  import Qwen3TTSTokenizer` (line 46 upstream). The `Qwen3TTSTokenizer`
  is an audio-to-codec encoder used only at inference time to turn
  `ref_audio` waveforms into codec tokens. The training side in
  verl-omni consumes codec tokens that already come out of the vLLM-Omni
  rollout, so the tokenizer is never needed.
- Stripped the speech_tokenizer + generate_config side-effects from
  `Qwen3TTSForConditionalGeneration.from_pretrained`. The vendored
  method now just calls `super().from_pretrained(...)` and returns the
  model. If you need inference-side speech tokenization, instantiate the
  upstream `qwen_tts` package's `Qwen3TTSTokenizer` separately and
  assign it via `model.load_speech_tokenizer(...)`.
- (Planned, follow-up task) Add `forward_training(input_ids, codec_ids,
  attention_mask, response_mask, prompt_lens)` on
  `Qwen3TTSForConditionalGeneration` that drives `self.talker.model` +
  `self.talker.codec_head` for `talker_logits` and
  `self.talker.forward_sub_talker_finetune` for `cb_rest_logits`.

## Re-sync procedure

To re-sync to a newer upstream commit:

```bash
cd /lustre/fsw/portfolios/coreai/users/yuekaiz/tts/Qwen3-TTS && git fetch && git checkout <new-sha>
cp /lustre/fsw/portfolios/coreai/users/yuekaiz/tts/Qwen3-TTS/qwen_tts/core/models/configuration_qwen3_tts.py <this-dir>/configuration_qwen3_tts.py
cp /lustre/fsw/portfolios/coreai/users/yuekaiz/tts/Qwen3-TTS/qwen_tts/core/models/modeling_qwen3_tts.py <this-dir>/modeling_qwen3_tts.py
# Re-apply the local edits above (they all live in well-isolated regions).
```

Update the commit hash at the top of this file after re-syncing.
