import os
import hashlib
from pathlib import Path
from typing import List, Optional, Union

import torch
from PIL import Image


def _align(v: int, stride: int = 64) -> int:
    v = int(v)
    return ((v + stride - 1) // stride) * stride


def _default_base_model() -> str:
    base_model_env = (os.getenv("INSTANTCHAR_BASE_MODEL") or "").strip()
    if base_model_env:
        return base_model_env

    local_flux = Path(__file__).resolve().parents[1] / "FLUX.1-dev"
    if local_flux.exists():
        return str(local_flux)
    return "black-forest-labs/FLUX.1-dev"


def _default_ip_adapter_path() -> str:
    ip_adapter_env = (os.getenv("INSTANTCHAR_IP_ADAPTER") or "").strip()
    if ip_adapter_env:
        return ip_adapter_env

    local_ip_adapter = Path(__file__).resolve().parent / "checkpoints" / "instantcharacter_ip-adapter.bin"
    if local_ip_adapter.exists():
        return str(local_ip_adapter)
    return ""


def _default_image_encoder_2_path() -> str:
    image_encoder_2_env = (os.getenv("INSTANTCHAR_IMAGE_ENCODER_2") or "").strip()
    if image_encoder_2_env:
        return image_encoder_2_env

    local_dinov2 = Path(__file__).resolve().parents[1] / "dinov2"
    if local_dinov2.exists():
        return str(local_dinov2)
    return "facebook/dinov2-giant"


class InstantCharacterPredictor:
    """InstantCharacter (Flux.1-dev + subject IP-Adapter) predictor.

    This predictor is used ONLY for scene keyframe T2I in MovieAgent.

    Inputs:
    - prompt: full prompt (MovieAgent already injects GLOBAL STYLE + MOTION prefix)
    - refer_image: list[str] or str. We map it to a single subject_image (best effort).
    - character_box: optional dict[str, box]; currently not used by InstantCharacter (no inpaint yet).

    Env overrides:
    - INSTANTCHAR_BASE_MODEL: default local FLUX.1-dev under microdrama_agent/models, else 'black-forest-labs/FLUX.1-dev'
    - INSTANTCHAR_IP_ADAPTER: default local checkpoint under microdrama_agent/models/InstantCharacter/checkpoints, else manual path
    - INSTANTCHAR_IMAGE_ENCODER: default 'google/siglip-so400m-patch14-384'
    - INSTANTCHAR_IMAGE_ENCODER_2: default local dinov2 under microdrama_agent/models, else 'facebook/dinov2-giant'
    - INSTANTCHAR_SUBJECT_SCALE: default '1.1'
    - INSTANTCHAR_STEPS: default '28'
    - INSTANTCHAR_CFG: default '3.5'
    - INSTANTCHAR_SEED: default '123456'
    - INSTANTCHAR_OFFLOAD: '1' to enable sequential cpu offload (if supported)

    Note: for true multi-character consistency, implement bbox-based inpainting in a later step.
    """

    _pipe = None
    _pipe_cfg = None
    _subject_emb_cache = {}  # key: abs_image_path -> torch.Tensor (ip_hidden_states)

    def __init__(self):
        pass

    @classmethod
    def release_pipe(cls):
        """Release cached InstantCharacter pipeline and embedding cache."""
        pipe = cls._pipe
        cls._pipe = None
        cls._pipe_cfg = None
        cls._subject_emb_cache = {}

        if pipe is not None:
            try:
                pipe.to("cpu")
            except Exception:
                pass
            try:
                del pipe
            except Exception:
                pass

        try:
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

        print("[InstantCharacter][INFO] released cached pipeline")

    def _pick_subject_image(self, refer_image: Union[List[str], str, None], character_box) -> Optional[str]:
        # Heuristic:
        # - If refer_image is a list, pick the first item (MovieAgent already orders by bank_prefer).
        # - If character_box is dict, we could later map role -> ref image, but current pipeline passes a flat list.
        if refer_image is None:
            return None
        if isinstance(refer_image, str):
            return refer_image if refer_image else None
        for p in list(refer_image):
            if p:
                return p
        return None

    def _ensure_pipe(self):
        base_model = _default_base_model()
        ip_adapter_path = _default_ip_adapter_path()
        image_encoder_path = (os.getenv("INSTANTCHAR_IMAGE_ENCODER") or "google/siglip-so400m-patch14-384").strip()
        image_encoder_2_path = _default_image_encoder_2_path()

        if not ip_adapter_path:
            raise RuntimeError(
                "INSTANTCHAR_IP_ADAPTER is not set and no local checkpoint was found. "
                "Please set it to the path of instantcharacter_ip-adapter.bin"
            )

        cfg = {
            "base_model": base_model,
            "ip_adapter_path": ip_adapter_path,
            "image_encoder_path": image_encoder_path,
            "image_encoder_2_path": image_encoder_2_path,
        }

        if InstantCharacterPredictor._pipe is not None and InstantCharacterPredictor._pipe_cfg == cfg:
            return

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
        try:
            from pathlib import Path
            from transformers.models.t5.tokenization_t5 import T5Tokenizer as _T5Tok  # type: ignore

            if not getattr(_T5Tok, "_movieagent_vocabfile_patch", False):
                _orig_init = _T5Tok.__init__

                def _init_with_vocab_fallback(self, vocab_file=None, *args, **kwargs):
                    if vocab_file is None:
                        name_or_path = kwargs.get("name_or_path") or getattr(self, "name_or_path", None)
                        if name_or_path:
                            base = Path(str(name_or_path))
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

        # Lazy import to avoid importing extra deps when not used
        # Prefer bundled pipeline in this package.
        from .pipeline import InstantCharacterFluxPipeline

        pipe = InstantCharacterFluxPipeline.from_pretrained(base_model, torch_dtype=torch.bfloat16)

        # Force low-peak init options (safe defaults)
        # - Keep projector on CPU during init (huge VRAM saver at init time)
        os.environ.setdefault("INSTANTCHAR_PROJECTOR_CPU", "1")

        # - Enable sequential CPU offload by default (if accelerate is available)
        #   You can disable with INSTANTCHAR_OFFLOAD=0
        offload_enabled = os.getenv("INSTANTCHAR_OFFLOAD", "1") not in ("0", "false", "False")
        print(f"[InstantCharacter][INFO] offload_enabled={offload_enabled}")
        if offload_enabled:
            m = getattr(pipe, "enable_sequential_cpu_offload", None)
            if callable(m):
                try:
                    pipe.enable_sequential_cpu_offload()
                except Exception:
                    pass
        else:
            # Only move full pipeline to GPU when offload is explicitly disabled.
            pipe.to("cuda")

        # Init adapter (encoders + attn proc + projector)
        # In offload mode, initialize adapter on CPU first to avoid load-time GPU memory spikes.
        adapter_device = torch.device("cpu") if offload_enabled else torch.device("cuda")
        pipe.init_adapter(
            image_encoder_path=image_encoder_path,
            image_encoder_2_path=image_encoder_2_path,
            subject_ipadapter_cfg=dict(subject_ip_adapter_path=ip_adapter_path, nb_token=1024),
            device=adapter_device,
        )

        InstantCharacterPredictor._pipe = pipe
        InstantCharacterPredictor._pipe_cfg = cfg

    def _make_composite_subject_image(self, paths: List[str], cell_size: int = 512) -> Image.Image:
        """把多张参考图拼成一个方格拼图，返回 PIL.Image。

        - paths: 有效的图片路径列表
        - cell_size: 每个单元格的目标最小边长（会按图片长宽比缩放并居中填充）
        """
        images = [Image.open(p).convert("RGB") for p in paths]
        # 计算每个单元格大小：使用所有图片最大边长，但不小于 384，也不大于 cell_size
        max_side = max(max(im.size) for im in images)
        cell = min(max(cell_size, 384), max_side if max_side > 0 else cell_size)

        # 网格行列数
        import math
        n = len(images)
        cols = math.ceil(math.sqrt(n))
        rows = math.ceil(n / cols)

        canvas_w = cols * cell
        canvas_h = rows * cell
        canvas = Image.new("RGB", (canvas_w, canvas_h), (255, 255, 255))

        for idx, im in enumerate(images):
            r = idx // cols
            c = idx % cols
            # 缩放并保持长宽比
            w, h = im.size
            scale = min(cell / w, cell / h)
            new_w, new_h = int(w * scale), int(h * scale)
            resized = im.resize((new_w, new_h), Image.LANCZOS)
            # paste 到单元格中心
            off_x = c * cell + (cell - new_w) // 2
            off_y = r * cell + (cell - new_h) // 2
            canvas.paste(resized, (off_x, off_y))

        # 最终返回尽量方形的图（pipeline 内会进一步按需要裁切/分块）
        return canvas

    def _ordered_roles_for_composite(self, character_box) -> List[str]:
        """确定多角色拼图中的角色顺序。

        目标：让拼图的格子顺序尽量与画面中角色的重要性一致（面积更大/更靠前更重要）。
        """
        if not isinstance(character_box, dict):
            return []

        def _area(b):
            try:
                if isinstance(b, (list, tuple)) and len(b) >= 4:
                    x0, y0, x1, y1 = b[:4]
                    return max(0.0, float(x1) - float(x0)) * max(0.0, float(y1) - float(y0))
            except Exception:
                pass
            return 0.0

        items = []
        for role, box in character_box.items():
            items.append((role, _area(box)))
        # 面积从大到小；同面积按名字稳定排序
        items.sort(key=lambda t: (-t[1], str(t[0])))
        return [r for r, _ in items]

    def _cache_key_for_ref(self, ref_path: str) -> str:
        """Cache key for a reference image.

        Include path + file stat so if the same path is overwritten (e.g. bank rebuilt),
        we won't accidentally reuse stale subject embeddings.
        """
        try:
            p = Path(ref_path).expanduser().resolve()
            st = p.stat()
            return f"{p}|{int(st.st_mtime_ns)}|{int(st.st_size)}"
        except Exception:
            try:
                return str(Path(ref_path).expanduser().resolve())
            except Exception:
                return str(ref_path)

    @torch.inference_mode()
    def _get_or_compute_subject_ip_hidden_states(self, ref_path: str, timestep: torch.Tensor) -> Optional[torch.Tensor]:
        """Compute (and cache) per-role ip_hidden_states from a CharacterBank reference image.

        Returns a tensor shaped like subject_image_prompt_embeds produced in pipeline (`subject_image_proj_model(...)[0]`).
        """
        if not ref_path:
            return None
        k = self._cache_key_for_ref(ref_path)
        disable_cache = os.getenv("INSTANTCHAR_DISABLE_EMB_CACHE", "0") not in ("0", "false", "False", "")
        if not disable_cache:
            cached = InstantCharacterPredictor._subject_emb_cache.get(k)
            if cached is not None:
                return cached

        p = Path(ref_path)
        if not p.exists():
            return None

        pipe = InstantCharacterPredictor._pipe
        if pipe is None:
            return None

        device = pipe._execution_device
        if isinstance(device, str):
            device = torch.device(device)
        dtype = pipe.transformer.dtype

        img = Image.open(str(p)).convert("RGB")
        img = img.resize((max(img.size), max(img.size)))
        img_embeds = pipe.encode_image_emb(img, device, dtype)

        # Projector may be partially on CPU when using sequential_cpu_offload + CPU init:
        # checking only next(proj.parameters()) misses LayerNorm weights still on CPU.
        # Always move the full module tree to the execution device for this forward.
        proj = pipe.subject_image_proj_model
        keep_proj_cpu = os.getenv("INSTANTCHAR_PROJECTOR_CPU", "0") not in ("0", "false", "False")

        proj.to(device)
        try:
            ip = proj(
                low_res_shallow=img_embeds['image_embeds_low_res_shallow'],
                low_res_deep=img_embeds['image_embeds_low_res_deep'],
                high_res_deep=img_embeds['image_embeds_high_res_deep'],
                timesteps=timestep.to(device=device, dtype=dtype),
                need_temb=True,
            )[0]
        finally:
            if keep_proj_cpu:
                try:
                    proj.to(torch.device("cpu"))
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        # keep on GPU for speed. If OOM, user can disable caching or implement CPU caching later.
        if not disable_cache:
            InstantCharacterPredictor._subject_emb_cache[k] = ip
        return ip

    def predict(self, prompt: str, refer_image, character_box, save_path: str, size=(1024, 512)):
        self._ensure_pipe()

        width_px, height_px = int(size[0]), int(size[1])
        width_px, height_px = _align(width_px, 64), _align(height_px, 64)

        # Steps/cfg/seed (consistency-oriented defaults)
        steps = int(os.getenv("INSTANTCHAR_STEPS", "30"))
        guidance_scale = float(os.getenv("INSTANTCHAR_CFG", "2.8"))

        # Base subject scale: lower than 2.0 to avoid oily/over-injected faces,
        # while keeping identity consistency strong enough.
        base_subject_scale = float(os.getenv("INSTANTCHAR_SUBJECT_SCALE", "1.35"))
        multi_decay = float(os.getenv("INSTANTCHAR_SUBJECT_SCALE_MULTI_DECAY", "0.92"))
        subject_scale_floor = float(os.getenv("INSTANTCHAR_SUBJECT_SCALE_FLOOR", "0.95"))

        n_roles = len(character_box) if isinstance(character_box, dict) else 1
        subject_scale = max(subject_scale_floor, base_subject_scale * (multi_decay ** max(0, n_roles - 1)))

        # Seed strategy:
        # - INSTANTCHAR_SEED=-1 or random -> new random seed each call
        # - otherwise stable seed, optionally diversified per shot path.
        seed_raw = str(os.getenv("INSTANTCHAR_SEED", "123456")).strip().lower()
        if seed_raw in ("-1", "random", "rand"):
            seed = int.from_bytes(os.urandom(4), byteorder="little") & 0x7FFFFFFF
        else:
            try:
                base_seed = int(seed_raw)
            except Exception:
                base_seed = 123456

            per_shot_seed = os.getenv("INSTANTCHAR_SEED_PER_SHOT", "1") not in ("0", "false", "False", "")
            if per_shot_seed:
                seed_hash = hashlib.sha256(f"{base_seed}|{save_path}".encode("utf-8")).hexdigest()
                seed = (base_seed ^ int(seed_hash[:8], 16)) & 0x7FFFFFFF
            else:
                seed = base_seed

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        gen = torch.Generator(device="cuda").manual_seed(seed)
        print(
            f"[InstantCharacter][DEBUG][predict] params steps={steps} cfg={guidance_scale} "
            f"subject_scale={subject_scale:.3f} n_roles={n_roles} seed={seed}"
        )

        # If we have role->ref and role->bbox, prefer true bbox-routed multi-subject injection.
        # refer_image: dict[role->ref_path]
        # character_box: dict[role->bbox]
        if isinstance(refer_image, dict) and isinstance(character_box, dict) and len(refer_image) > 0 and len(character_box) > 0:
            pipe = InstantCharacterPredictor._pipe
            device = pipe._execution_device
            timestep = torch.zeros([1], device=device, dtype=torch.float32)

            # one-time debug: print role -> ref image mapping and cache hit
            try:
                mapping = []
                for role, bbox in character_box.items():
                    ref_path = refer_image.get(role)
                    if not ref_path:
                        mapping.append((str(role), None, "missing_ref"))
                        continue
                    k = self._cache_key_for_ref(ref_path)
                    hit = k in InstantCharacterPredictor._subject_emb_cache
                    mapping.append((str(role), str(ref_path), f"cache_hit={hit}"))
                print(f"[InstantCharacter][DEBUG][predict] role_ref_mapping={mapping}")
            except Exception as e:
                print(f"[InstantCharacter][DEBUG][predict] role_ref_mapping_print_failed: {e}")

            subj_items = []
            for role, bbox in character_box.items():
                ref_path = refer_image.get(role)
                if not ref_path:
                    continue
                ip_hid = self._get_or_compute_subject_ip_hidden_states(ref_path, timestep=timestep)
                if ip_hid is None:
                    continue
                subj_items.append(
                    {
                        "role": role,
                        "ip_hidden_states": ip_hid,
                        "scale": subject_scale,
                        "bbox": bbox,
                    }
                )

            if subj_items:
                out = pipe(
                    prompt=prompt,
                    num_inference_steps=steps,
                    guidance_scale=guidance_scale,
                    subject_emb_dicts=subj_items,
                    width=width_px,
                    height=height_px,
                    generator=gen,
                ).images[0]
                out.save(save_path)
                return save_path

        # Fallback to previous single-subject/collage behavior
        subject_image = None

        if isinstance(refer_image, dict) and isinstance(character_box, dict):
            ordered_roles = self._ordered_roles_for_composite(character_box)
            paths = []
            for role in ordered_roles:
                p = refer_image.get(role)
                if p and Path(p).exists():
                    paths.append(p)
            if not paths:
                for role, p in refer_image.items():
                    if p and Path(p).exists():
                        paths.append(p)

            if len(paths) == 1:
                subject_image = Image.open(paths[0]).convert("RGB")
            elif len(paths) > 1:
                subject_image = self._make_composite_subject_image(paths)

        elif isinstance(refer_image, (list, tuple)):
            existing_paths = [p for p in list(refer_image) if p and Path(p).exists()]
            if len(existing_paths) == 1:
                subject_image = Image.open(existing_paths[0]).convert("RGB")
            elif len(existing_paths) > 1:
                subject_image = self._make_composite_subject_image(existing_paths)

        else:
            subject_img_path = self._pick_subject_image(refer_image, character_box)
            if subject_img_path and Path(subject_img_path).exists():
                subject_image = Image.open(subject_img_path).convert("RGB")

        pipe = InstantCharacterPredictor._pipe
        if subject_image is None:
            out = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                width=width_px,
                height=height_px,
                generator=gen,
            ).images[0]
        else:
            out = pipe(
                prompt=prompt,
                num_inference_steps=steps,
                guidance_scale=guidance_scale,
                subject_image=subject_image,
                subject_scale=subject_scale,
                width=width_px,
                height=height_px,
                generator=gen,
            ).images[0]

        out.save(save_path)
        return save_path
