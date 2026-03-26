import sys
import os
import json

from tqdm import tqdm




class GenModel:
    def __init__(self,args, model_name, save_mode="video") -> None:
        self.save_mode = save_mode
        if model_name == "vc2":
            from models.VC2.vc2_predict import VideoCrafter
            self.predictor = VideoCrafter("vc2")
        elif model_name == "vc09":
            from models.VC09.vc09_predict import VideoCrafter09
            self.predictor = VideoCrafter09()
        elif model_name == "modelscope":
            from models.modelscope.modelscope_predict import ModelScope
            self.predictor = ModelScope()
        elif model_name == "latte1":
            from models.latte.latte_1_predict import Latte1
            self.predictor = Latte1()
            
        elif model_name == "SDXL-1":
            from models.SD.sd_predict import SDXL
            self.predictor = SDXL()
        elif model_name == "SD-21":
            from models.SD.sd_predict import SD21
            self.predictor = SD21()
        elif model_name == "SD-14":
            from models.SD.sd_predict import SD14
            self.predictor = SD14()
        elif model_name == "SD-3":
            from models.SD.sd_predict import SD3
            self.predictor = SD3() 
        elif model_name == "ConsisID":
            from models.ConsisID.consisid_predict import ConsisID
            self.predictor = ConsisID(model_path="/storage/wuweijia/MovieGen/MovieDirector/MovieDirector/movie_agent/ckpts") 
        elif model_name == "StoryDiffusion":
            from models.StoryDiffusion.storydiffusion import StoryDiffusion
            # self.predictor = StoryDiffusion()
        elif model_name == "OmniGen":
            from models.OmniGen.OmniGen import OminiGen_pipe
            self.predictor = OminiGen_pipe()
        elif model_name == "ROICtrl":
            from models.ROICtrl.ROICtrl import ROICtrl_pipe
            self.predictor = ROICtrl_pipe(args.pretrained_roictrl , args.roictrl_path)
        elif model_name == "FreeGraftor":
            from models.FreeGraftor.freegraftor_predict import FreeGraftorPredictor
            self.predictor = FreeGraftorPredictor()
        elif model_name == "InstantCharacter":
            # Flux.1-dev + subject IP-Adapter (tuning-free) for character-consistent keyframes
            from models.InstantCharacter.instantcharacter_predict import InstantCharacterPredictor
            self.predictor = InstantCharacterPredictor()
        else:
            raise ValueError(f"This {model_name} has not been implemented yet")
    
    
    def predict(self, prompt, refer_image, character_box, save_path, size):
        # os.makedirs(save_path, exist_ok=True)
        # name = prompt.strip().replace(" ", "_")
        # if self.save_mode == "video":
        #     save_name = os.path.join(save_path, f"{name}.mp4")
        # elif self.save_mode == "img":
        #     save_name = os.path.join(save_path, f"{name}.png")
        # else:
        #     raise NotImplementedError(f"Wrong mode -- {self.save_mode}")
        
        self.predictor.predict(prompt, refer_image, character_box, save_path, size)
        return prompt, save_path





class ToolBox:
    def __init__(self) -> None:
        pass
    

    def call(self, tool_name, video_pairs):
        method = getattr(self, tool_name, None)
        
        
        if callable(method):
            return method(video_pairs)
        else:
            raise AttributeError(f"'{self.__class__.__name__}' object has no attribute '{tool_name}'")
    
    def color_binding(self, image_pairs):
        sys.path.insert(0, "eval_tools/t2i_comp/BLIPvqa_eval")
        from eval_tools.t2i_comp.BLIPvqa_eval.BLIP_vqa_eval_agent import calculate_attribute_binding
        results = calculate_attribute_binding(image_pairs)
        return results
    
    def shape_binding(self, image_pairs):
        sys.path.insert(0, "eval_tools/t2i_comp/BLIPvqa_eval")
        from eval_tools.t2i_comp.BLIPvqa_eval.BLIP_vqa_eval_agent import calculate_attribute_binding
        results = calculate_attribute_binding(image_pairs)
        return results

    def texture_binding(self, image_pairs):
        sys.path.insert(0, "eval_tools/t2i_comp/BLIPvqa_eval")
        from eval_tools.t2i_comp.BLIPvqa_eval.BLIP_vqa_eval_agent import calculate_attribute_binding
        results = calculate_attribute_binding(image_pairs)
        return results
    

    def non_spatial(self, image_pairs):
        sys.path.insert(0, "eval_tools/t2i_comp/CLIPScore_eval")
        from eval_tools.t2i_comp.CLIPScore_eval.CLIP_similarity_eval_agent import calculate_clip_score
        results = calculate_clip_score(image_pairs)
        return results
    
    
    def overall_consistency(self, video_pairs):
        from eval_tools.vbench.overall_consistency import compute_overall_consistency
        results = compute_overall_consistency(video_pairs)
        return results
    
    
    def aesthetic_quality(self, video_pairs):
        from eval_tools.vbench.aesthetic_quality import compute_aesthetic_quality
        results = compute_aesthetic_quality(video_pairs)
        return results

    def appearance_style(self, video_pairs):
        from eval_tools.vbench.appearance_style import compute_appearance_style
        results = compute_appearance_style(video_pairs)
        return results
    
    
    def background_consistency(self, video_pairs):
        from eval_tools.vbench.background_consistency import compute_background_consistency
        results = compute_background_consistency(video_pairs)
        return results

    def color(self, video_pairs):
        from eval_tools.vbench.color import compute_color
        results = compute_color(video_pairs)
        return results
    
    def dynamic_degree(self, video_pairs):
        from eval_tools.vbench.dynamic_degree import compute_dynamic_degree
        results = compute_dynamic_degree(video_pairs)
        return results

    def human_action(self, video_pairs):
        from eval_tools.vbench.human_action import compute_human_action
        results = compute_human_action(video_pairs)
        return results

    def imaging_quality(self, video_pairs):
        from eval_tools.vbench.imaging_quality import compute_imaging_quality
        results = compute_imaging_quality(video_pairs)
        return results

    def motion_smoothness(self, video_pairs):
        from eval_tools.vbench.motion_smoothness import compute_motion_smoothness
        results = compute_motion_smoothness (video_pairs)
        return results

    def multiple_objects(self, video_pairs):
        from eval_tools.vbench.multiple_objects import compute_multiple_objects
        results = compute_multiple_objects(video_pairs)
        return results

    def object_class(self, video_pairs):
        from eval_tools.vbench.object_class import compute_object_class
        results = compute_object_class(video_pairs)
        return results
    
    def scene(self, video_pairs):
        from eval_tools.vbench.scene import compute_scene
        results = compute_scene(video_pairs)
        return results
    
    def spatial_relationship(self, video_pairs):
        from eval_tools.vbench.spatial_relationship import compute_spatial_relationship
        results = compute_spatial_relationship(video_pairs)
        return results

    def subject_consistency(self, video_pairs):
        from eval_tools.vbench.subject_consistency import compute_subject_consistency
        results = compute_subject_consistency(video_pairs)
        return results

    def temporal_style(self, video_pairs):
        from eval_tools.vbench.temporal_style import compute_temporal_style
        results = compute_temporal_style(video_pairs)
        return results



class AudioGenModel:
    def __init__(self, model_name,photo_audio_path,characters_list) -> None:
        if model_name == "VALL-E":
            from models.VALLE.VALL_E import VALLE_pipe
            self.predictor = VALLE_pipe(photo_audio_path,characters_list)
        else:
            raise ValueError(f"This {model_name} has not been implemented yet")
    
    
    def predict(self, subtitle, save_path):
        
        self.predictor.predict(subtitle, save_path)
        return subtitle, save_path
    

class TalkingModel:
    def __init__(self, model_name) -> None:
        pass
        # if model_name == "Hallo2":
        #     from models.VALLE.VALL_E import VALLE_pipe
        #     self.predictor = VALLE_pipe()
        # else:
        #     raise ValueError(f"This {model_name} has not been implemented yet")
    
    
    def predict(self, subtitle, save_path):
        pass
        
        # self.predictor.predict(subtitle, save_path)
        # return subtitle, save_path

class Image2VideoModel:
    def __init__(self, args,model_name) -> None:
        # pass
        if model_name == "CogVideoX":
            from models.CogVideoX.CogVideoX import CogVideoX_pipe
            self.predictor = CogVideoX_pipe()
        elif model_name == "SVD":
            from models.SVD.svd import SVD_pipe
            self.predictor = SVD_pipe()
        elif model_name == "I2Vgen":
            from models.I2Vgen.I2Vgen import I2Vgen_pipe
            self.predictor = I2Vgen_pipe()
        elif model_name == "HunyuanVideo_I2V":
            from models.HunyuanVideo_I2V.HunyuanVideo_I2V import HunyuanVideo_I2V_pipe
            self.predictor = HunyuanVideo_I2V_pipe(args)
        else:
            raise ValueError(f"This {model_name} has not been implemented yet")
    
    
    def predict(self, prompt, image_path,video_save_path, size):

        self.predictor.predict(prompt, image_path,video_save_path, size)
        return image_path

class ToolCalling:
    def __init__(self, args, sample_model, audio_model, talk_model, Image2Video, photo_audio_path, characters_list, save_mode):
        self.args = args
        self.gen = GenModel(args, sample_model, save_mode)
        self._image2video_name = Image2Video
        self.image2video = None

        # Audio is optional; when --skip_audio is set OR no audio model is provided,
        # we should not even import/init VALLE.
        if getattr(args, "skip_audio", False) or (not audio_model) or (not talk_model):
            self.audio_gen = None
            self.talk_gen = None
        else:
            self.audio_gen = AudioGenModel(audio_model, photo_audio_path, characters_list)
            self.talk_gen = TalkingModel(talk_model)

        self.eval_tools = ToolBox()

    def _ensure_image2video(self):
        """Lazy-init Image2Video model to avoid memory overlap with bank builder."""
        if self.image2video is None:
            self.image2video = Image2VideoModel(self.args, self._image2video_name)

    def _crop_bbox_pil(self, img, bbox, *, pad=0.08):
        """Crop bbox region from PIL image.

        bbox can be normalized ([0,1]) or pixel; format [x0,y0,x1,y1].
        """
        from PIL import Image

        if img is None or bbox is None:
            return None
        if not isinstance(img, Image.Image):
            return None

        try:
            w, h = img.size
            x0, y0, x1, y1 = bbox[:4]
            mx = max(float(x0), float(y0), float(x1), float(y1))
            if mx <= 1.5:  # normalized
                x0, x1 = float(x0) * w, float(x1) * w
                y0, y1 = float(y0) * h, float(y1) * h
            else:
                x0, x1 = float(x0), float(x1)
                y0, y1 = float(y0), float(y1)

            bw, bh = max(1.0, x1 - x0), max(1.0, y1 - y0)
            px = bw * float(pad)
            py = bh * float(pad)

            xx0 = int(max(0, min(w - 1, x0 - px)))
            yy0 = int(max(0, min(h - 1, y0 - py)))
            xx1 = int(max(xx0 + 1, min(w, x1 + px)))
            yy1 = int(max(yy0 + 1, min(h, y1 + py)))
            return img.crop((xx0, yy0, xx1, yy1))
        except Exception:
            return None

    def _hist_cosine_similarity(self, a_pil, b_pil, bins: int = 64) -> float:
        """Fallback similarity metric (no extra deps): HSV histogram cosine similarity in [0,1]."""
        import numpy as np

        if a_pil is None or b_pil is None:
            return 0.0
        try:
            a = np.asarray(a_pil.convert("RGB"))
            b = np.asarray(b_pil.convert("RGB"))
            # downsample for speed
            a = a[::2, ::2]
            b = b[::2, ::2]

            def _hsv_hist(x):
                import cv2

                hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
                hist = cv2.calcHist([hsv], [0, 1, 2], None, [bins, bins, bins], [0, 180, 0, 256, 0, 256])
                hist = hist.flatten().astype(np.float32)
                n = np.linalg.norm(hist) + 1e-8
                return hist / n

            ha = _hsv_hist(a)
            hb = _hsv_hist(b)
            sim = float(np.dot(ha, hb))
            # clamp
            if sim < 0:
                sim = 0.0
            if sim > 1:
                sim = 1.0
            return sim
        except Exception:
            return 0.0

    def _imgutils_similarity(self, crop_pil, ref_pil) -> float:
        """Compute similarity in [0,1] using imgutils CCIP feature.

        We use:
        - imgutils.metrics.ccip_extract_feature
        - imgutils.metrics.ccip_difference (lower is more similar)

        Returned value is a monotonic similarity: sim = 1 / (1 + diff)
        """
        try:
            from imgutils.metrics import ccip_extract_feature, ccip_difference
        except Exception:
            # hard fallback
            return self._hist_cosine_similarity(crop_pil, ref_pil)

        try:
            fa = ccip_extract_feature(crop_pil)
            fb = ccip_extract_feature(ref_pil)
            diff = float(ccip_difference(fa, fb))
            if diff < 0:
                diff = 0.0
            return float(1.0 / (1.0 + diff))
        except Exception:
            return self._hist_cosine_similarity(crop_pil, ref_pil)

    def _check_character_consistency(self, img_path: str, refer_path, character_box, *, threshold: float, pad: float) -> tuple[bool, dict]:
        """Return (ok, details).

        Two modes:
        1) role-aware (strict): if refer_path and character_box are dict and roles overlap, check per-role.
        2) set-level (fast, no matching): compare all bbox crops with all reference images; for each crop take best match.

        details contains:
        - 'mode': 'role' or 'set'
        - 'sims': per role or per crop index
        - 'min_sim': minimum over checked pairs
        """
        from PIL import Image

        if not getattr(self.args, "consistency_check", False):
            return True, {}

        # need bbox dict at least
        if not isinstance(character_box, dict) or len(character_box) == 0:
            return True, {}

        try:
            img = Image.open(img_path).convert("RGB")
        except Exception:
            return True, {}

        # Build reference list (prefer dict[role->path] else list)
        ref_items = []
        if isinstance(refer_path, dict):
            for r, p in refer_path.items():
                if p:
                    ref_items.append((str(r), p))
        elif isinstance(refer_path, (list, tuple)):
            for i, p in enumerate(list(refer_path)):
                if p:
                    ref_items.append((str(i), p))

        if not ref_items:
            return True, {}

        # Load refs
        refs = []
        for rid, p in ref_items:
            try:
                ref = Image.open(p).convert("RGB")
                refs.append((rid, ref))
            except Exception:
                continue

        if not refs:
            return True, {}

        # --- Mode 1: role-aware if possible ---
        if isinstance(refer_path, dict):
            overlap = [r for r in character_box.keys() if r in refer_path]
            if overlap:
                sims = {}
                ok_all = True
                min_sim = 1.0
                for role in overlap:
                    bbox = character_box.get(role)
                    ref_img_path = refer_path.get(role)
                    if not ref_img_path:
                        continue
                    try:
                        ref = Image.open(ref_img_path).convert("RGB")
                    except Exception:
                        continue

                    crop = self._crop_bbox_pil(img, bbox, pad=pad)
                    if crop is None:
                        continue
                    sim = float(self._imgutils_similarity(crop, ref))
                    sims[str(role)] = sim
                    min_sim = min(min_sim, sim)
                    if sim < float(threshold):
                        ok_all = False

                return ok_all, {"mode": "role", "sims": sims, "min_sim": min_sim}

        # --- Mode 2: set-level (no role matching) ---
        # For each bbox crop, compute best similarity among all references.
        sims = {}
        ok_all = True
        min_sim = 1.0

        # keep stable order by bbox area desc
        crops = []
        for idx, (role, bbox) in enumerate(character_box.items()):
            crop = self._crop_bbox_pil(img, bbox, pad=pad)
            if crop is not None:
                crops.append((str(role), crop))

        if not crops:
            return True, {"mode": "set", "sims": {}, "min_sim": 1.0}

        for role, crop in crops:
            best = 0.0
            for _rid, ref in refs:
                s = float(self._imgutils_similarity(crop, ref))
                if s > best:
                    best = s
            sims[str(role)] = best
            min_sim = min(min_sim, best)
            if best < float(threshold):
                ok_all = False

        return ok_all, {"mode": "set", "sims": sims, "min_sim": min_sim}

    def sample(self, prompt, refer_path, character_box, subtitle, save_path, size = (1024, 512)):
        # Reuse existing keyframe when doing full pipeline (non-images_only):
        # if image already exists, skip T2I and go straight to I2V.
        images_only = bool(getattr(self.args, "images_only", False))
        keyframe_exists = (
            os.path.isfile(save_path)
            and os.path.getsize(save_path) > 0
        )
        force_regen_image = os.getenv("MOVIEAGENT_FORCE_REGEN_IMAGE", "0") not in ("0", "false", "False", "")
        reuse_existing_keyframe = (not images_only) and keyframe_exists and (not force_regen_image)

        if reuse_existing_keyframe:
            print(f"[MovieAgent] keyframe exists, skip image generation and continue to I2V: {save_path}")
        else:
            if (not images_only) and keyframe_exists and force_regen_image:
                print(f"[MovieAgent] MOVIEAGENT_FORCE_REGEN_IMAGE=1 -> regenerate keyframe: {save_path}")
            # Generate keyframe image (with optional consistency retry loop)
            max_retry = int(getattr(self.args, "consistency_max_retry", 2) or 0)
            threshold = float(getattr(self.args, "consistency_threshold", 0.55) or 0.0)
            pad = float(getattr(self.args, "consistency_bbox_pad", 0.08) or 0.0)

            attempt = 0
            while True:
                prompt, _content = self.gen.predict(prompt, refer_path, character_box, save_path, size)

                ok, details = self._check_character_consistency(
                    save_path,
                    refer_path=refer_path,
                    character_box=character_box,
                    threshold=threshold,
                    pad=pad,
                )
                if ok:
                    if getattr(self.args, "consistency_check", False) and details:
                        print(f"[MovieAgent][ConsistCheck] PASS sims={details}")
                    break

                attempt += 1
                print(f"[MovieAgent][ConsistCheck] FAIL sims={details} threshold={threshold} retry={attempt}/{max_retry}")
                if attempt > max_retry:
                    print("[MovieAgent][ConsistCheck][WARN] max retry reached, keep current keyframe.")
                    break

            # If images_only mode, stop after keyframe generation.
            if images_only:
                return

        # Release heavy T2I pipeline (e.g., InstantCharacter/Flux) before loading I2V model
        # to avoid overlapping memory peaks.
        try:
            pred = getattr(self.gen, "predictor", None)
            release_fn = getattr(pred, "release_pipe", None)
            if callable(release_fn):
                release_fn()
        except Exception as e:
            print(f"[MovieAgent][WARN] failed to release T2I pipeline before I2V: {e}")

        self._ensure_image2video()
        base, _ext = os.path.splitext(save_path)
        video_save_path = base + ".mp4"
        save_path = self.image2video.predict(prompt, save_path, video_save_path, size)

    def eval(self, tool_name, video_pairs):
        results = self.eval_tools.call(tool_name, video_pairs)
        return results





def save_json(content, file_path):
    with open(file_path, 'w') as json_file:
        json.dump(content, json_file, indent=4)

