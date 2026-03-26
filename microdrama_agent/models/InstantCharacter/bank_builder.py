"""Build MovieAgent CharacterBank images using Flux (no FreeGraftor).

We generate ONE stable asset per role:
- identity.png: full-body, front view, neutral standing pose (single image)

NOTE: We intentionally do NOT generate pose.png to save compute and improve identity stability.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch


def _align(v: int, stride: int = 64) -> int:
    v = int(v)
    return ((v + stride - 1) // stride) * stride


def _to_list(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, list):
        return [str(i) for i in x if str(i).strip()]
    if isinstance(x, str):
        return [x] if x.strip() else []
    return [str(x)]


def _dedup(xs: List[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for s in xs:
        s = str(s).strip()
        if not s:
            continue
        if s in seen:
            continue
        seen.add(s)
        out.append(s)
    return out


def _join(xs: List[str]) -> str:
    return ", ".join(_dedup(xs))


def _default_base_model() -> str:
    base_model_env = (os.getenv("INSTANTCHAR_BASE_MODEL") or "").strip()
    if base_model_env:
        return base_model_env

    # Prefer local FLUX.1-dev weights bundled in this repo.
    local_flux = Path(__file__).resolve().parents[1] / "FLUX.1-dev"
    if local_flux.exists():
        return str(local_flux)
    return "black-forest-labs/FLUX.1-dev"


_PIPE = None
_PIPE_CFG = None


def _ensure_pipe(base_model: str):
    global _PIPE, _PIPE_CFG

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    cfg = {"base_model": str(base_model), "device": device, "dtype": str(dtype)}
    if _PIPE is not None and _PIPE_CFG == cfg:
        return _PIPE

    # --- sentencepiece hotfix ---
    # sentencepiece binding requires a str path. Some tokenizer code paths may pass a Path-like object,
    # causing: TypeError: not a string
    try:
        import sentencepiece as _spm  # type: ignore

        _SP = _spm.SentencePieceProcessor
        if not getattr(_SP, "_movieagent_str_path_patch", False):
            _orig_load = _SP.Load
            _orig_load_from_file = getattr(_SP, "LoadFromFile", None)

            def _load_str(self, model_file, *args, **kwargs):
                return _orig_load(self, str(model_file), *args, **kwargs)

            _SP.Load = _load_str  # type: ignore[assignment]

            if callable(_orig_load_from_file):
                def _load_from_file_str(self, model_file, *args, **kwargs):
                    return _orig_load_from_file(self, str(model_file), *args, **kwargs)

                _SP.LoadFromFile = _load_from_file_str  # type: ignore[assignment]

            _SP._movieagent_str_path_patch = True  # type: ignore[attr-defined]
    except Exception:
        # If sentencepiece isn't installed or patching fails, continue and let the real error surface.
        pass

    # --- transformers T5Tokenizer vocab_file=None hotfix ---
    # Some FLUX tokenizer_2 folders store sentencepiece model as `tokenizer.model`.
    # When transformers falls back to building slow tokenizer, it may pass vocab_file=None,
    # producing: OSError: Not found: "None".
    # Patch T5Tokenizer.__init__ to infer vocab_file from local folder.
    try:
        from transformers.models.t5.tokenization_t5 import T5Tokenizer as _T5Tok  # type: ignore

        if not getattr(_T5Tok, "_movieagent_vocabfile_patch", False):
            _orig_init = _T5Tok.__init__

            def _init_with_vocab_fallback(self, vocab_file=None, *args, **kwargs):
                if vocab_file is None:
                    # Try common local names used by HF exports
                    name_or_path = kwargs.get("name_or_path") or getattr(self, "name_or_path", None)
                    if name_or_path:
                        base = Path(str(name_or_path))
                        # If passed a file (e.g. tokenizer_config.json), use its parent
                        if base.is_file():
                            base = base.parent
                        for cand in [base / "tokenizer.model", base / "spiece.model"]:
                            if cand.exists():
                                vocab_file = str(cand)
                                break
                return _orig_init(self, vocab_file=vocab_file, *args, **kwargs)

            _T5Tok.__init__ = _init_with_vocab_fallback  # type: ignore[assignment]
            _T5Tok._movieagent_vocabfile_patch = True  # type: ignore[attr-defined]
    except Exception:
        pass

    # Prefer the bundled InstantCharacter Flux pipeline:
    # - Makes CharacterBank generation consistent with InstantCharacter keyframes
    # - Avoids some FluxPipeline tokenizer/sentencepiece edge cases
    from .pipeline import InstantCharacterFluxPipeline

    # Reduce noisy warnings & avoid tokenizer parallelism issues
    os.environ.setdefault("TRANSFORMERS_NO_ADVISORY_WARNINGS", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    pipe = InstantCharacterFluxPipeline.from_pretrained(
        base_model,
        torch_dtype=dtype,
    )
    pipe.to(device)

    _PIPE = pipe
    _PIPE_CFG = cfg
    return pipe


@dataclass
class BankBuildResult:
    identity_path: str
    pose_path: str | None = None


def _build_prompts(*, role: str, profile: Dict, style: Dict, view: str = "front") -> Tuple[str, str]:
    """Return (identity_prompt, negative_prompt).

    view: 'front' | 'side' | 'back'

    Identity spec:
    - single subject
    - full-body
    - plain studio background
    - specified view

    NOTE:
    - This is meant for CharacterBank multi-view references (not shot generation).
    """

    gs = _to_list(style.get("global_style_keywords"))
    if not gs:
        gs = [
            "Pixar-like 3D animated feature film style",
            "stylized characters",
            "appealing proportions",
            "clean shapes",
            "soft cinematic lighting",
            "coherent color palette",
            "consistent character design across shots",
        ]

    global_negative = (style.get("global_negative") or "").strip()

    ident = profile.get("identity") or {}
    outfit = profile.get("default_outfit") or {}

    outfit_txt = _join(
        [
            f"{outfit.get('top','')}".strip(),
            f"{outfit.get('bottom','')}".strip(),
            f"{outfit.get('shoes','')}".strip(),
        ]
    )

    face_features = ident.get("face_features") or []
    face_txt = _join(face_features) if isinstance(face_features, list) else str(face_features)

    base_keywords = profile.get("prompt_keywords") or []
    kw_txt = _join(base_keywords) if isinstance(base_keywords, list) else str(base_keywords)


    view = str(view or "front").strip().lower()
    if view not in {"front", "side", "back"}:
        view = "front"

    view_phrase = {
        "front": "front view",
        "side": "side profile view",
        "back": "back view",
    }[view]

    # Keep prompts compact and explicit to avoid multi-view reference sheets.
    style_prefix = _join(
        [
            "single subject",
            "full body",
            view_phrase,
            "standing",
            "neutral pose",
            "arms relaxed",
            "plain studio background",
            "soft even lighting",
            "sharp focus",
            "centered composition",
            "high consistency",
        ]
        + gs
    )

    anti_realism = _join(
        [
            "photorealistic",
            "realistic skin texture",
            "pores",
            "live-action",
            "film still",
            "HDR",
            "gritty realism",
        ]
    )

    # Strongly block multi-view / turnaround / pose sheet behaviors.
    anti_multiview = _join(
        [
            "multiple views",
            "turnaround",
            "character sheet",
            "reference sheet",
            "collage",
            "split screen",
            "front and side view",
            "back view and front view",
            "rotating",
            "text",
            "watermark",
            "logo",
        ]
    )

    # Keep bank negative prompt compact: do NOT append per-role profile negative from Step_0.
    neg_parts = [t for t in [global_negative, anti_realism, anti_multiview] if str(t).strip()]
    neg_txt = "; ".join([str(t).strip() for t in neg_parts if str(t).strip()])

    identity_prompt = _join(
        [
            role,
            style_prefix,
            f"age {ident.get('age_range','')}",
            f"{ident.get('gender_presentation','')}",
            f"skin tone {ident.get('skin_tone','')}",
            f"hair {ident.get('hair','')}",
            f"face {face_txt}",
            f"body {ident.get('body','')}",
            f"outfit {outfit_txt}" if outfit_txt else "",
            f"keywords {kw_txt}" if kw_txt else "",
        ]
    ).strip(", ")

    return identity_prompt, neg_txt


def build_character_bank_multi_view_for_role(
    *,
    role: str,
    profile: Dict,
    style: Dict,
    out_dir: str,
    size=(768, 768),
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
    views: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Generate multi-view reference images for a role.

    Returns a dict view->path.

    Files written under out_dir:
    - identity_front.png
    - identity_side.png
    - identity_back.png

    Also writes/updates identity.png as identity_front.png (main default).
    """

    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    if views is None:
        views = ["front", "side", "back"]

    # Generation params share InstantCharacter defaults when present.
    if seed is None:
        seed = int(os.getenv("INSTANTCHAR_SEED", "123456"))
    if steps is None:
        steps = int(os.getenv("INSTANTCHAR_STEPS", "28"))
    if guidance is None:
        guidance = float(os.getenv("INSTANTCHAR_CFG", "3.5"))

    base_model = _default_base_model()
    pipe = _ensure_pipe(base_model)
    device = getattr(pipe, "_execution_device", None) or ("cuda" if torch.cuda.is_available() else "cpu")

    w, h = _align(int(size[0]), 64), _align(int(size[1]), 64)

    results: Dict[str, str] = {}

    # Ensure deterministic but distinct per view.
    view_seed_offset = {"front": 0, "side": 100, "back": 200}

    for v in views:
        v = str(v).strip().lower()
        if v not in {"front", "side", "back"}:
            continue

        out_path = str(out_dir_p / f"identity_{v}.png")
        if os.path.exists(out_path):
            results[v] = out_path
            continue

        prompt, negative_prompt = _build_prompts(role=role, profile=profile, style=style, view=v)

        gen = torch.Generator(device=str(device)).manual_seed(int(seed) + int(view_seed_offset.get(v, 0)))
        img = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt or None,
            num_inference_steps=int(steps),
            guidance_scale=float(guidance),
            width=w,
            height=h,
            generator=gen,
        ).images[0]
        img.save(out_path)
        results[v] = out_path

    # Keep backward-compatible default identity.png
    if "front" in results:
        identity_path = str(out_dir_p / "identity.png")
        try:
            from shutil import copyfile

            copyfile(results["front"], identity_path)
        except Exception:
            pass

    return results


def build_character_bank_for_role(
    *,
    role: str,
    profile: Dict,
    style: Dict,
    out_dir: str,
    size=(768, 768),
    seed: Optional[int] = None,
    steps: Optional[int] = None,
    guidance: Optional[float] = None,
) -> BankBuildResult:
    out_dir_p = Path(out_dir)
    out_dir_p.mkdir(parents=True, exist_ok=True)

    identity_path = str(out_dir_p / "identity.png")

    # NEW: optionally generate multi-view bank (front/side/back)
    multi_view = os.getenv("MOVIEAGENT_BANK_MULTI_VIEW", "0").strip() not in ["0", "false", "False"]
    if multi_view:
        build_character_bank_multi_view_for_role(
            role=role,
            profile=profile,
            style=style,
            out_dir=str(out_dir_p),
            size=size,
            seed=seed,
            steps=steps,
            guidance=guidance,
        )
        return BankBuildResult(identity_path=identity_path, pose_path=None)

    identity_prompt, negative_prompt = _build_prompts(role=role, profile=profile, style=style, view="front")

    # Generation params share InstantCharacter defaults when present.
    if seed is None:
        seed = int(os.getenv("INSTANTCHAR_SEED", "123456"))
    if steps is None:
        steps = int(os.getenv("INSTANTCHAR_STEPS", "28"))
    if guidance is None:
        guidance = float(os.getenv("INSTANTCHAR_CFG", "3.5"))

    base_model = _default_base_model()

    pipe = _ensure_pipe(base_model)
    device = getattr(pipe, "_execution_device", None) or ("cuda" if torch.cuda.is_available() else "cpu")

    w, h = _align(int(size[0]), 64), _align(int(size[1]), 64)
    gen = torch.Generator(device=str(device)).manual_seed(int(seed))

    img = pipe(
        prompt=identity_prompt,
        negative_prompt=negative_prompt or None,
        num_inference_steps=int(steps),
        guidance_scale=float(guidance),
        width=w,
        height=h,
        generator=gen,
    ).images[0]
    img.save(identity_path)

    # Do NOT generate pose.png
    return BankBuildResult(identity_path=identity_path, pose_path=None)
