#!/usr/bin/env bash
set -euo pipefail
export OPENAI_API_KEY=""
# Usage:
#   bash movie_agent/script/run_freegraftor_hunyuan_i2v.sh \
#     --script_path dataset/FrozenII/script_synopsis.json \
#     --character_photo_path dataset/FrozenII/character_list \
#     --llm deepseek-v3 \
#     --cuda 0
#
# Notes:
# - LLM uses OPENAI compatible API. Please export OPENAI_API_KEY before running, e.g.:
#     export OPENAI_API_KEY="..."
# - To pin GPU:
#     export CUDA_VISIBLE_DEVICES=0
#   or pass: --cuda 0
# - HunyuanVideo_I2V config should be available in configs/HunyuanVideo_I2V.json
# - FreeGraftor models should exist under FreeGraftor/FreeGraftor/models

SCRIPT_PATH=""
CHARACTER_PHOTO_PATH=""
LLM="deepseek-v3"
CUDA_DEVICES=""

# keep compatibility flags (optional; FreeGraftor pipeline currently skips audio by default)
AUDIO_MODEL=""
TALK_MODEL=""
SKIP_AUDIO=1

# resume controls
RESUME=1
START_FROM=""

# parse args
while [[ $# -gt 0 ]]; do
  case "$1" in
    --script_path)
      SCRIPT_PATH="$2"; shift 2;;
    --character_photo_path)
      CHARACTER_PHOTO_PATH="$2"; shift 2;;
    --llm|--LLM)
      LLM="$2"; shift 2;;
    --cuda)
      CUDA_DEVICES="$2"; shift 2;;
    --audio_model)
      AUDIO_MODEL="$2"; shift 2;;
    --talk_model)
      TALK_MODEL="$2"; shift 2;;
    --no-skip-audio)
      SKIP_AUDIO=0; shift 1;;
    --no-resume)
      RESUME=0; shift 1;;
    --start_from)
      START_FROM="$2"; shift 2;;
    -h|--help)
      echo "See header comment for usage."; exit 0;;
    *)
      echo "Unknown argument: $1"; exit 1;;
  esac
done

if [[ -z "$SCRIPT_PATH" ]]; then
  echo "ERROR: --script_path is required"; exit 1
fi
if [[ -z "$CHARACTER_PHOTO_PATH" ]]; then
  echo "ERROR: --character_photo_path is required"; exit 1
fi

# align with legacy scripts
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-1}

if [[ -n "$CUDA_DEVICES" ]]; then
  export CUDA_VISIBLE_DEVICES="$CUDA_DEVICES"
fi

if [[ -z "${OPENAI_API_KEY:-}" ]]; then
  echo "WARN: OPENAI_API_KEY is empty. LLM calls will fail unless you export it." >&2
fi

EXTRA_ARGS=()
if [[ -n "$AUDIO_MODEL" ]]; then
  EXTRA_ARGS+=(--audio_model "$AUDIO_MODEL")
fi
if [[ -n "$TALK_MODEL" ]]; then
  EXTRA_ARGS+=(--talk_model "$TALK_MODEL")
fi
if [[ "$SKIP_AUDIO" -eq 1 ]]; then
  EXTRA_ARGS+=(--skip_audio)
fi
if [[ "$RESUME" -eq 1 ]]; then
  EXTRA_ARGS+=(--resume)
fi
if [[ -n "$START_FROM" ]]; then
  EXTRA_ARGS+=(--start_from "$START_FROM")
fi
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
python movie_agent/run.py \
  --script_path "$SCRIPT_PATH" \
  --character_photo_path "$CHARACTER_PHOTO_PATH" \
  --LLM "$LLM" \
  --audio_model VALL-E \
  --gen_model FreeGraftor \
  --Image2Video HunyuanVideo_I2V \
  "${EXTRA_ARGS[@]}"
