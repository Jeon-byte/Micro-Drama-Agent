from PIL import Image, ImageDraw, ImageFont
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video
import os
import cv2
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from .hyvideo.utils.file_utils import save_videos_grid
from .hyvideo.config import parse_args
from .hyvideo.inference import HunyuanVideoSampler


class HunyuanVideo_I2V_pipe:
    def __init__(self,args):

        # print(args)
        # args = parse_args(args)

        models_root_path = Path(args.model_base)
        if not models_root_path.exists():
            raise ValueError(f"`models_root` not exists: {models_root_path}")

        # Ensure hyvideo.constants uses the same model root.
        # hyvideo/constants.py reads MODEL_BASE from env at import time; set it here
        # so all submodules (VAE/TEXT/TOKENIZER paths) resolve correctly.
        os.environ["MODEL_BASE"] = str(models_root_path)

        # Load models
        self.hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)

        # Get the updated args
        args = self.hunyuan_video_sampler.args
        self.args = args

        # ------------------------------
        # Optional memory optimizations
        # ------------------------------
        # 不改 infer_steps/cfg/分辨率的前提下，主要靠 offload/切片来降峰值显存。
        # 用环境变量控制，默认不改变现有行为：
        #   HUNYUAN_CPU_OFFLOAD=1    -> 启用 sequential CPU offload（最省显存，速度会慢）
        #   HUNYUAN_ATTENTION_SLICING=1 -> 启用 attention slicing（中等省显存）
        #   HUNYUAN_VAE_SLICING=1    -> VAE slicing
        #   HUNYUAN_VAE_TILING=1     -> VAE tiling（通常更省显存，但可能更慢）
        pipe = getattr(self.hunyuan_video_sampler, "pipeline", None)
        if pipe is not None:
            if os.environ.get("HUNYUAN_CPU_OFFLOAD", "0") not in ("0", "false", "False", ""):
                try:
                    pipe.enable_sequential_cpu_offload()
                    logger.info("[Hunyuan I2V] enable_sequential_cpu_offload() enabled")
                except Exception as e:
                    logger.warning(f"[Hunyuan I2V] enable_sequential_cpu_offload() failed: {e}")

            if os.environ.get("HUNYUAN_ATTENTION_SLICING", "0") not in ("0", "false", "False", ""):
                try:
                    pipe.enable_attention_slicing()
                    logger.info("[Hunyuan I2V] enable_attention_slicing() enabled")
                except Exception as e:
                    logger.warning(f"[Hunyuan I2V] enable_attention_slicing() failed: {e}")

            if os.environ.get("HUNYUAN_VAE_SLICING", "0") not in ("0", "false", "False", ""):
                try:
                    pipe.enable_vae_slicing()
                    logger.info("[Hunyuan I2V] enable_vae_slicing() enabled")
                except Exception as e:
                    logger.warning(f"[Hunyuan I2V] enable_vae_slicing() failed: {e}")

            if os.environ.get("HUNYUAN_VAE_TILING", "0") not in ("0", "false", "False", ""):
                try:
                    pipe.enable_vae_tiling()
                    logger.info("[Hunyuan I2V] enable_vae_tiling() enabled")
                except Exception as e:
                    logger.warning(f"[Hunyuan I2V] enable_vae_tiling() failed: {e}")

    def release_pipe(self):
        """Drop HunyuanVideo sampler + pipeline so T2I (e.g. InstantCharacter/Flux) can use VRAM alone.

        MovieAgent may run many shots: without this, the I2V model stays loaded while the next keyframe
        is generated, doubling GPU memory and often causing OOM (or Linux OOM-killer, exit 137).
        """
        import gc

        sampler = getattr(self, "hunyuan_video_sampler", None)
        self.hunyuan_video_sampler = None
        if sampler is not None:
            try:
                pipe = getattr(sampler, "pipeline", None)
                if pipe is not None:
                    try:
                        pipe.to("cpu")
                    except Exception:
                        pass
            except Exception:
                pass
            try:
                del sampler
            except Exception:
                pass

        try:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                try:
                    torch.cuda.ipc_collect()
                except Exception:
                    pass
        except Exception:
            pass

        logger.info("[Hunyuan I2V] released pipeline (freed VRAM for image generation)")

    def predict(self, prompt, image_path, video_save_path, size = (569, 320)):
        
        self.args.prompt = prompt
        self.args.video_size[0] = size[1]
        self.args.video_size[1] = size[0]
        self.args.video_length = int(getattr(self.args, "video_length", 129) or 129)
        self.args.i2v_image_path = image_path

        # Auto-match i2v resolution to requested output size (can be disabled).
        auto_match_resolution = os.environ.get("HUNYUAN_AUTO_MATCH_I2V_RESOLUTION", "1") not in ("0", "false", "False", "")
        if auto_match_resolution:
            req_h, req_w = int(size[1]), int(size[0])
            if max(req_h, req_w) >= 1200 or min(req_h, req_w) >= 700:
                matched_resolution = "720p"
            elif min(req_h, req_w) >= 500:
                matched_resolution = "540p"
            else:
                matched_resolution = "360p"
            if getattr(self.args, "i2v_resolution", None) != matched_resolution:
                logger.info(
                    f"[Hunyuan I2V] auto-match i2v_resolution: {getattr(self.args, 'i2v_resolution', None)} -> {matched_resolution}"
                )
            self.args.i2v_resolution = matched_resolution

        # Keep inference controls configurable and avoid forcing prompt-dominant cfg.
        self.args.infer_steps = int(getattr(self.args, "infer_steps", 50) or 50)
        self.args.i2v_stability = bool(getattr(self.args, "i2v_stability", True))

        # Optional explicit overrides via env (for quick A/B tests).
        cond_override = os.environ.get("HUNYUAN_I2V_CONDITION_TYPE", "").strip()
        if cond_override in ("token_replace", "latent_concat"):
            self.args.i2v_condition_type = cond_override

        # i2v default cfg is intentionally low to preserve reference frame identity.
        cfg_default = 1.0
        try:
            cfg_scale = float(getattr(self.args, "cfg_scale", cfg_default) or cfg_default)
        except Exception:
            cfg_scale = cfg_default
        if cfg_scale <= 0:
            cfg_scale = cfg_default
        cfg_override = os.environ.get("HUNYUAN_CFG_SCALE", "").strip()
        if cfg_override:
            try:
                cfg_scale = float(cfg_override)
            except Exception:
                pass
        self.args.cfg_scale = cfg_scale

        if getattr(self.args, "neg_prompt", None) in (None, "", [""], [" "]):
            self.args.neg_prompt = None

        if getattr(self.args, "flow_shift", None) is None:
            self.args.flow_shift = 5.0

        embedded_guidance_scale = getattr(self.args, "embedded_cfg_scale", None)
        if embedded_guidance_scale is True:
            embedded_guidance_scale = 6.0
        elif embedded_guidance_scale in (False, 0, 0.0):
            embedded_guidance_scale = None

        # Optional env override for quick tuning.
        egs_override = os.environ.get("HUNYUAN_EMBEDDED_CFG_SCALE", "").strip()
        if egs_override:
            try:
                embedded_guidance_scale = float(egs_override)
            except Exception:
                pass

        outputs = self.hunyuan_video_sampler.predict(
            prompt=self.args.prompt,
            height=self.args.video_size[0],
            width=self.args.video_size[1],
            video_length=self.args.video_length,
            seed=self.args.seed,
            negative_prompt=self.args.neg_prompt,
            infer_steps=self.args.infer_steps,
            guidance_scale=self.args.cfg_scale,
            num_videos_per_prompt=self.args.num_videos,
            flow_shift=self.args.flow_shift,
            batch_size=self.args.batch_size,
            embedded_guidance_scale=embedded_guidance_scale,
            i2v_mode=self.args.i2v_mode,
            i2v_resolution=self.args.i2v_resolution,
            i2v_image_path=self.args.i2v_image_path,
            i2v_condition_type=self.args.i2v_condition_type,
            i2v_stability=self.args.i2v_stability,
        )
        samples = outputs['samples']

        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            save_videos_grid(sample, video_save_path, fps=24)



