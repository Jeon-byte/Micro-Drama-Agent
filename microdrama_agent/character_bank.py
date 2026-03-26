"""CharacterBank for MovieAgent.

Goal: stop depending on dataset-provided `character_list/<role>/best.png` and instead
build a reusable global character asset bank from the script JSON (Character section).

This module is intentionally lightweight:
- It defines where assets live on disk.
- It can "ensure" that bank assets exist (currently via placeholders).

Flux integration hook:
- Later you can implement `FluxCharacterBankBuilder` to generate identity/outfit/pose
  images using local FLUX weights.

File layout (default):
Results/<movie>/<LLM>_<gen>_<i2v>/character_bank/<role>/{identity.png,outfit.png,pose.png}
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional


DEFAULT_ASSETS = {
    "identity": "identity.png",
    "outfit": "outfit.png",
    "pose": "pose.png",
}


@dataclass
class CharacterAssets:
    role: str
    identity_path: str
    outfit_path: str
    pose_path: str

    def identity_path_for_view(self, view: str | None) -> str:
        """Return view-specific identity image path if exists, else fallback to identity.png.

        Expected filenames under same role dir:
        - identity_front.png
        - identity_side.png
        - identity_back.png
        """
        view = (str(view or "").strip().lower() or "front")
        if view in ["front", "f"]:
            cand = str(Path(self.identity_path).with_name("identity_front.png"))
        elif view in ["side", "profile", "s"]:
            cand = str(Path(self.identity_path).with_name("identity_side.png"))
        elif view in ["back", "rear", "b"]:
            cand = str(Path(self.identity_path).with_name("identity_back.png"))
        else:
            cand = ""

        if cand and os.path.exists(cand):
            return cand
        return self.identity_path

    def as_list(self, prefer: str = "identity") -> List[str]:
        # prefer can be: identity|outfit|pose|identity_front|identity_side|identity_back
        order = [prefer, "identity", "outfit", "pose"]
        seen = set()
        out: List[str] = []
        for k in order:
            if k in seen:
                continue
            seen.add(k)

            if k in ["identity_front", "identity_side", "identity_back"]:
                view = k.split("_", 1)[-1]
                p = self.identity_path_for_view(view)
            else:
                p = getattr(self, f"{k}_path", None)

            if p and os.path.exists(p):
                out.append(p)
        # dedupe while preserving order
        dedup: List[str] = []
        for p in out:
            if p not in dedup:
                dedup.append(p)
        return dedup


def _safe_role(role: str) -> str:
    return role.replace(" ", "_")


class CharacterBank:
    def __init__(self, root_dir: str):
        self.root_dir = Path(root_dir)

    def role_dir(self, role: str) -> Path:
        return self.root_dir / _safe_role(role)

    def assets(self, role: str) -> CharacterAssets:
        d = self.role_dir(role)
        return CharacterAssets(
            role=role,
            identity_path=str(d / DEFAULT_ASSETS["identity"]),
            outfit_path=str(d / DEFAULT_ASSETS["outfit"]),
            pose_path=str(d / DEFAULT_ASSETS["pose"]),
        )

    def ensure(
        self,
        roles: List[str],
        placeholders_from: Optional[str] = None,
        *,
        profiles_json_path: Optional[str] = None,
        use_flux: bool = True,
        flux_conda_env: str = "freegraftor",
    ) -> Dict[str, CharacterAssets]:
        """Ensure bank assets exist.

        Priority:
        1) If `use_flux` and `profiles_json_path` exists: generate assets with Flux.
           Builder is selected by env MOVIEAGENT_BANK_BUILDER:
             - 'instantcharacter' (default): uses InstantCharacter Flux pipeline
             - 'characonsist': uses CharaConsist pipeline (training-free masks/point matching)

           When env MOVIEAGENT_BANK_MULTI_VIEW=1, builder may generate:
             - identity_front.png / identity_side.png / identity_back.png
           and keep identity.png as the front view for backward compatibility.

        2) Else: If placeholders_from/<role>/best.png exists, copy to identity/outfit/pose if missing.
        3) Else: write tiny placeholders.
        """
        out: Dict[str, CharacterAssets] = {}
        self.root_dir.mkdir(parents=True, exist_ok=True)

        profiles = None
        style = {}
        if use_flux and profiles_json_path and os.path.exists(profiles_json_path):
            data = json.loads(Path(profiles_json_path).read_text(encoding="utf-8"))
            style = data.get("style") or {}
            profiles = data.get("characters") or {}

        # Merge MovieAgent global style env into the bank generation style.
        # This keeps CharacterBank (identity/pose) aligned with keyframe prompts.
        global_style = (os.getenv("MOVIEAGENT_GLOBAL_STYLE") or "").strip()
        global_quality = (os.getenv("MOVIEAGENT_GLOBAL_QUALITY") or "").strip()
        global_neg = (os.getenv("MOVIEAGENT_GLOBAL_STYLE_NEG") or "").strip()

        if global_style or global_quality or global_neg:
            gsk = style.get("global_style_keywords")
            if not isinstance(gsk, list):
                gsk = []
            if global_style:
                gsk.append(global_style)
            if global_quality:
                gsk.append(global_quality)
            # de-dupe
            dedup = []
            seen = set()
            for s in [str(x).strip() for x in gsk if str(x).strip()]:
                if s in seen:
                    continue
                seen.add(s)
                dedup.append(s)
            style["global_style_keywords"] = dedup

            if global_neg:
                prev = (style.get("global_negative") or "").strip()
                style["global_negative"] = (prev + "; " + global_neg).strip("; ") if prev else global_neg

        builder = (os.getenv("MOVIEAGENT_BANK_BUILDER") or "instantcharacter").strip().lower()
        bank_multi_view = os.getenv("MOVIEAGENT_BANK_MULTI_VIEW", "0").strip() not in ["0", "false", "False"]

        # CharaConsist builder is intended for front/side/back ID-constrained bank generation.
        if builder == "characonsist":
            bank_multi_view = True

        chara_pred = None
        chara_pred_cls = None

        for role in roles:
            a = self.assets(role)
            d = self.role_dir(role)
            d.mkdir(parents=True, exist_ok=True)

            # 1) Flux generate if available and missing
            if profiles is not None and role in profiles:
                required_view_files = [d / "identity_front.png", d / "identity_side.png", d / "identity_back.png"] if bank_multi_view else []
                need_generate = (not os.path.exists(a.identity_path)) or any((not p.exists()) for p in required_view_files)

                if need_generate:
                    if builder == "characonsist":
                        if chara_pred is None:
                            try:
                                from movie_agent.models.CharaConsist.characonsist_predict import CharaConsistPredictor
                            except Exception:
                                from models.CharaConsist.characonsist_predict import CharaConsistPredictor  # type: ignore

                            chara_pred_cls = CharaConsistPredictor
                            chara_pred = CharaConsistPredictor()

                        chara_pred.generate_bank_multi_view(
                            role=role,
                            profile=profiles[role],
                            style=style,
                            out_dir=str(d),
                        )
                    else:
                        try:
                            from movie_agent.models.InstantCharacter.bank_builder import build_character_bank_for_role
                        except Exception:
                            from models.InstantCharacter.bank_builder import build_character_bank_for_role  # type: ignore

                        build_character_bank_for_role(
                            role=role,
                            profile=profiles[role],
                            style=style,
                            out_dir=str(d),
                        )

                # outfit is stable: copy from identity if not exists
                if os.path.exists(a.identity_path) and (not os.path.exists(a.outfit_path)):
                    from shutil import copyfile

                    copyfile(a.identity_path, a.outfit_path)

                # pose is optional now: DO NOT force-create it here.
                out[role] = a
                continue

            # 2) best.png fallback
            src_best = None
            if placeholders_from:
                cand = Path(placeholders_from) / _safe_role(role) / "best.png"
                if cand.exists():
                    src_best = str(cand)

            for p in [a.identity_path, a.outfit_path, a.pose_path]:
                if os.path.exists(p):
                    continue
                if src_best is not None:
                    from shutil import copyfile

                    copyfile(src_best, p)
                else:
                    _write_tiny_png(p)

            out[role] = a

        # Important for InstantCharacter workflow: free CharaConsist model after bank build
        # to avoid overlapping large pipelines in memory.
        if builder == "characonsist" and chara_pred_cls is not None:
            try:
                chara_pred_cls.release_pipe()
            except Exception as e:
                print(f"[CharacterBank][WARN] failed to release CharaConsist pipeline: {e}")

        return out


def _write_tiny_png(path: str):
    # Minimal valid PNG (1x1 transparent). Avoid pillow dependency.
    png_bytes = (
        b"\x89PNG\r\n\x1a\n"  # signature
        b"\x00\x00\x00\rIHDR"  # IHDR
        b"\x00\x00\x00\x01\x00\x00\x00\x01"  # 1x1
        b"\x08\x06\x00\x00\x00"  # RGBA
        b"\x1f\x15\xc4\x89"  # crc
        b"\x00\x00\x00\x0aIDAT"  # IDAT
        b"x\x9cc\x00\x01\x00\x00\x05\x00\x01"  # zlib
        b"\x0d\n\x2d\xb4"  # crc
        b"\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    Path(path).write_bytes(png_bytes)


def roles_from_script_json(script_path: str, n: int = 40) -> List[str]:
    data = json.loads(Path(script_path).read_text(encoding="utf-8"))
    chars = data.get("Character") or []
    return [c for c in chars[:n] if isinstance(c, str) and c.strip()]
