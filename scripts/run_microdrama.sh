#!/usr/bin/env bash
set -euo pipefail

# Micro-Drama Agent runner
# Usage examples:
#   bash scripts/run_microdrama.sh --user_story "..." --character_photo_path "./characters" \
#     --only "Sub-Script 2,Scene 1,Shot 2" --start_from video --images_only
#
# Notes:
# - You can override any environment variable below by exporting it BEFORE running this script.
# - This script defaults to offline mode and local model paths; adjust as needed.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"

# If this script was edited on Windows, CRLF may break argument parsing.
# Strip trailing carriage returns from all args.
_args=()
for a in "$@"; do
  _args+=("${a%$'\r'}")
done

# --------- Default environment (override by pre-exporting) ---------
: "${OMP_NUM_THREADS:=1}"

# Validate OMP_NUM_THREADS to avoid libgomp warning when it's empty/invalid
if ! [[ "${OMP_NUM_THREADS}" =~ ^[0-9]+$ ]] || [[ "${OMP_NUM_THREADS}" -lt 1 ]]; then
  OMP_NUM_THREADS=1
fi

# Force regenerate keyframe image(s) even if cached results exist (if respected by backend)
: "${MOVIEAGENT_FORCE_REGEN_IMAGE:=1}"

# InstantCharacter tuning knobs
: "${INSTANTCHAR_SUBJECT_SCALE:=1.7}"
: "${INSTANTCHAR_SUBJECT_SCALE_MULTI_DECAY:=0.92}"
: "${INSTANTCHAR_CFG:=2.8}"
: "${INSTANTCHAR_SEED:=123456}"
: "${INSTANTCHAR_SEED_PER_SHOT:=1}"

# CharacterBank / backend selection flags
: "${MOVIEAGENT_BANK_BUILDER:=characonsist}"
: "${MOVIEAGENT_BANK_MULTI_VIEW:=1}"

# Offline HF (recommended for autodl environments)
: "${HF_HOME:=$ROOT_DIR/microdrama_agent/models/hf_home}"
: "${HF_HUB_OFFLINE:=1}"
: "${TRANSFORMERS_OFFLINE:=1}"

# Global style knobs

: "${MOVIEAGENT_GLOBAL_QUALITY:=high quality, ultra-detailed}"

# --------- Model paths ---------
# CharaConsist (FLUX.1-dev)
: "${CHARACONSIST_MODEL_PATH:=$ROOT_DIR/microdrama_agent/models/FLUX.1-dev}"

# InstantCharacter
: "${INSTANTCHAR_BASE_MODEL:=$ROOT_DIR/microdrama_agent/models/FLUX.1-dev}"
: "${INSTANTCHAR_IP_ADAPTER:=$ROOT_DIR/microdrama_agent/models/InstantCharacter/checkpoints/instantcharacter_ip-adapter.bin}"
: "${INSTANTCHAR_IMAGE_ENCODER_2:=$ROOT_DIR/microdrama_agent/models/dinov2}"
# If you have a local SigLIP snapshot, point INSTANTCHAR_IMAGE_ENCODER to it; otherwise leave empty.
: "${INSTANTCHAR_IMAGE_ENCODER:=}"

# InstantCharacter runtime knobs (set to 1 for CPU/offload; 0 for full GPU)
: "${INSTANTCHAR_PROJECTOR_CPU:=0}"
: "${INSTANTCHAR_ENCODER_CPU:=0}"
: "${INSTANTCHAR_OFFLOAD:=0}"

# HunyuanVideo_I2V: sequential CPU offload lowers VRAM (slower). Helpful if I2V still OOMs after T2I release.
# : "${HUNYUAN_CPU_OFFLOAD:=1}"

# HunyuanVideo_I2V mmap/loader knobs
: "${HUNYUAN_TORCH_LOAD_MAP:=cuda}"
: "${HUNYUAN_TORCH_LOAD_MMAP:=1}"
: "${HUNYUAN_TORCH_LOAD_WEIGHTS_ONLY:=0}"
: "${HUNYUAN_TEXT_ENCODER_LOW_CPU_MEM:=1}"
: "${HUNYUAN_TEXT_ENCODER_DEVICE_MAP:=cpu}"

export OMP_NUM_THREADS
export MOVIEAGENT_FORCE_REGEN_IMAGE
export INSTANTCHAR_SUBJECT_SCALE INSTANTCHAR_SUBJECT_SCALE_MULTI_DECAY
export INSTANTCHAR_CFG 
export INSTANTCHAR_SEED=1
export INSTANTCHAR_SEED_PER_SHOT=1
export MOVIEAGENT_BANK_BUILDER MOVIEAGENT_BANK_MULTI_VIEW
export HF_HOME HF_HUB_OFFLINE TRANSFORMERS_OFFLINE
export MOVIEAGENT_GLOBAL_STYLE_NEG MOVIEAGENT_GLOBAL_QUALITY
export CHARACONSIST_MODEL_PATH
export INSTANTCHAR_BASE_MODEL INSTANTCHAR_IP_ADAPTER INSTANTCHAR_IMAGE_ENCODER_2
export INSTANTCHAR_PROJECTOR_CPU INSTANTCHAR_ENCODER_CPU INSTANTCHAR_OFFLOAD
export HUNYUAN_TORCH_LOAD_MAP HUNYUAN_TORCH_LOAD_MMAP HUNYUAN_TORCH_LOAD_WEIGHTS_ONLY
export HUNYUAN_TEXT_ENCODER_LOW_CPU_MEM HUNYUAN_TEXT_ENCODER_DEVICE_MAP
export INSTANTCHAR_OFFLOAD=1
if [[ -n "$INSTANTCHAR_IMAGE_ENCODER" ]]; then
  export INSTANTCHAR_IMAGE_ENCODER
fi

# --------- Logging ---------
mkdir -p Results
TS="$(date +%Y%m%d_%H%M%S)"
LOG_FILE="Results/run_${TS}.log"

# --------- Args passthrough ---------
# Everything provided to this script is forwarded to microdrama_agent/run.py
# You can pass --user_story / --script_path / --only / --images_only etc.

echo "[run_microdrama.sh] Running: python microdrama_agent/run.py ${_args[*]}" | tee -a "$LOG_FILE"

set +e
python microdrama_agent/run.py "${_args[@]}" 2>&1 | tee -a "$LOG_FILE"
EXIT_CODE=${PIPESTATUS[0]}
set -e

echo "" | tee -a "$LOG_FILE"
echo "[run_microdrama.sh] Exit code: ${EXIT_CODE}" | tee -a "$LOG_FILE"
echo "[run_microdrama.sh] Log saved to: ${LOG_FILE}"
exit "$EXIT_CODE"
