import torch
torch.cuda.init()
torch.cuda.set_device(0)

import os
from datetime import datetime
import argparse

# Allow running as a script from different working directories by ensuring
# the folder containing this file is on sys.path.
import sys
from pathlib import Path
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from base_agent import BaseAgent
from system_prompts import sys_prompts
from tools import ToolCalling, save_json
import json
import yaml

# CharacterBank (new global assets; replaces legacy best.png dependency)
from character_bank import CharacterBank, roles_from_script_json


def parse_args():
    
    parser = argparse.ArgumentParser(description="MovieAgent")

    parser.add_argument(
        "--user_story",
        type=str,
        default=None,
        help="(optional) user input story in natural language. If provided, MovieAgent will first generate a script_synopsis.json and then run the full pipeline.",
    )

    parser.add_argument(
        "--script_path",
        type=str,
        required=False,
        default=None,
        help="script synopsis json path (optional when --user_story is used)",
    )
    parser.add_argument(
        "--character_photo_path",
        type=str,
        required=True,
        help="legacy reference root folder (optional as placeholder source for bank).",
    )
    parser.add_argument(
        "--character_bank_root",
        type=str,
        default=None,
        help="character bank root dir. Default: <Results>/<movie>/<config>/character_bank",
    )
    parser.add_argument(
        "--bank_prefer",
        type=str,
        default="identity",
        choices=[
            "identity",
            "outfit",
            "pose",
            "identity_front",
            "identity_side",
            "identity_back",
            "identity_auto",
        ],
        help="which bank asset to prioritize as reference image",
    )
    parser.add_argument(
        "--bank_builder",
        type=str,
        default=None,
        choices=["instantcharacter", "characonsist"],
        help="character bank builder backend (default auto: InstantCharacter keyframe flow prefers characonsist)",
    )
    parser.add_argument(
        "--bank_multi_view",
        type=int,
        default=None,
        choices=[0, 1],
        help="force character bank multi-view generation: 1=front/side/back, 0=single identity",
    )
    parser.add_argument(
        "--LLM",
        type=str,
        required=False,
        help="model: gpt4-o | deepseek-r1 | deepseek-v3",
    )
    parser.add_argument(
        "--gen_model",
        type=str,
        required=False,
        help="model: ROICtrl | StoryDiffusion | FreeGraftor | InstantCharacter | CharaConsist",
    )
    parser.add_argument(
        "--audio_model",
        type=str,
        required=False,
        help="model (optional)",
    ) 
    parser.add_argument(
        "--talk_model",
        type=str,
        required=False,
        help="model (optional)",
    )
    parser.add_argument(
        "--Image2Video",
        type=str,
        required=False,
        help="model: SVD | I2Vgen | CogVideoX | HunyuanVideo_I2V",
    )

    parser.add_argument(
        "--skip_audio",
        action="store_true",
        help="skip audio generation (currently audio is disabled in code; this flag is reserved for a complete pipeline)",
    )

    # Resume / skip LLM stages
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse existing Step_1/Step_2/Step_3 json outputs if present and skip those LLM stages",
    )
    parser.add_argument(
        "--start_from",
        type=str,
        default=None,
        choices=["script", "scene", "shot", "video", "final"],
        help="start pipeline from a specific stage: script|scene|shot|video|final",
    )
    parser.add_argument(
        "--only",
        type=str,
        default=None,
        help='(optional) only generate a specific shot for quick testing. Format: "Scene 2,Shot 1". Works in video stage.',
    )

    # NEW: image-only mode (skip I2V)
    parser.add_argument(
        "--images_only",
        action="store_true",
        help="only generate the first-frame key image for each shot and skip image-to-video generation",
    )

    # NEW: enable CharaConsist as an alternative to InstantCharacter
    parser.add_argument(
        "--CharaConsist",
        action="store_true",
        help="use CharaConsist as the character-consistent keyframe generator (overrides --gen_model)",
    )

    # --- Character consistency check (bbox vs reference) ---
    parser.add_argument('--consistency_check', action='store_true', help='Check bbox character similarity vs reference image and retry keyframe generation if too low')
    parser.add_argument('--consistency_threshold', type=float, default=0.55, help='Similarity threshold in [0,1]; lower triggers retry')
    parser.add_argument('--consistency_max_retry', type=int, default=2, help='Max retry times for keyframe regeneration when consistency check fails')
    parser.add_argument('--consistency_bbox_pad', type=float, default=0.08, help='Padding ratio for bbox crop before similarity check')

    args = parser.parse_args()

    # If user_story is provided, script_path can be generated later.
    if not args.user_story and not args.script_path:
        raise SystemExit("Either --script_path or --user_story must be provided")

    # If user explicitly enables CharaConsist, override gen_model
    if getattr(args, "CharaConsist", False):
        args.gen_model = "CharaConsist"

    if args.gen_model:
        config = load_config(args.gen_model)
        # print(config)
        for key, value in config.items():
            if not getattr(args, key, None):  
                setattr(args, key, value)
    
    if args.Image2Video:
        config = load_config(args.Image2Video)
        # print(config)
        for key, value in config.items():
            if not getattr(args, key, None):  
                setattr(args, key, value)

    # --- normalize common model paths to absolute (relative to movie_agent/) ---
    base_dir = Path(__file__).resolve().parent
    for path_key in ["model_base", "dit_weight", "i2v_dit_weight", "pretrained_roictrl", "roictrl_path"]:
        if hasattr(args, path_key):
            v = getattr(args, path_key)
            if isinstance(v, str) and v:
                p = Path(v)
                if not p.is_absolute():
                    # config uses ./weight/... which is under movie_agent/
                    p2 = (base_dir / v).resolve()
                    setattr(args, path_key, str(p2))

    # Ensure hyvideo uses the correct model root (used by hyvideo/constants.py via env var).
    if getattr(args, "model_base", None):
        os.environ["MODEL_BASE"] = str(Path(args.model_base).resolve())

    # Expose CharacterBank controls as environment for downstream builders.
    if getattr(args, "bank_builder", None):
        os.environ["MOVIEAGENT_BANK_BUILDER"] = str(args.bank_builder).strip().lower()
    if getattr(args, "bank_multi_view", None) is not None:
        os.environ["MOVIEAGENT_BANK_MULTI_VIEW"] = str(int(args.bank_multi_view))

    return args


def load_config(model_name):
    """ with model_name, read config """
    # configs 位于 movie_agent/configs 下（而不是仓库根目录 configs）
    base_dir = Path(__file__).resolve().parent
    config_path = base_dir / "configs" / f"{model_name}.json"

    if config_path.exists():
        with open(config_path, "r", encoding="utf-8") as f:
            if config_path.suffix == ".json":
                return json.load(f)
            elif config_path.suffix in [".yaml", ".yml"]:
                return yaml.safe_load(f)
    return {}  

class ScriptBreakAgent:
    def __init__(self, args, sample_model="sdxl-1", audio_model="VALL-E", talk_model = "Hallo2", Image2Video = "CogVideoX",
                 script_path = "", character_photo_path="", save_mode="img"):
        self.args = args
        self.sample_model = sample_model
        self.audio_model = audio_model
        self.talk_model = talk_model
        self.Image2Video = Image2Video

        # If args.user_story is set, script_path will be generated later.
        self.script_path = script_path
        self.character_photo_path = character_photo_path
        self.characters_list = []

        if script_path:
            self.movie_name = script_path.split("/")[-1].replace(".json","")
        else:
            self.movie_name = "user_story"

        self.save_mode = save_mode

        # 1) init paths first (NO LLM calls here)
        self.update_info(paths_only=True)
        # 2) init agents (LLM available)
        self.init_agent()
        # 3) if user_story mode: generate synopsis json now
        self.update_info(paths_only=False)
        # 4) init downstream tools using final script_path
        self.init_videogen()
    
    def init_videogen(self):
        movie_script, characters_list = self.extract_characters_from_json(self.script_path, 40)

        self.tools = ToolCalling(self.args, sample_model=self.sample_model, audio_model = self.audio_model, \
                                 talk_model = self.talk_model, Image2Video = self.Image2Video, \
                                    photo_audio_path = self.character_photo_path, \
                                    characters_list=characters_list, save_mode=self.save_mode)
    
    def init_agent(self):
        # initialize agent
        self.screenwriter_agent = BaseAgent(self.args.LLM, system_prompt=sys_prompts["screenwriterCoT-sys"], use_history=False, temp=0.7)
        self.sceneplanning_agent = BaseAgent(self.args.LLM, system_prompt=sys_prompts["ScenePlanningCoT-sys"], use_history=False, temp=0.7)
        self.shotplotcreate_agent = BaseAgent(self.args.LLM, system_prompt=sys_prompts["ShotPlotCreateCoT-sys"], use_history=False, temp=0.7)

        self.character_profile_agent = BaseAgent(self.args.LLM, system_prompt=sys_prompts["CharacterProfileJSON-sys"], use_history=False, temp=0.2)

        # new: from user story -> script synopsis json generator
        self.user_story_agent = BaseAgent(self.args.LLM, system_prompt=sys_prompts["UserStory2Synopsis-sys"], use_history=False, temp=0.7)

    def format_results(self, results):
        formatted_text = "Observation:\n\n"
        for item in results:
            formatted_text += f"Prompt: {item['Prompt']}\n"
            for question, answer in zip(item["Questions"], item["Answers"]):
                formatted_text += f"Question: {question} -- Answer: {answer}\n"
            formatted_text += "\n"
        return formatted_text

    
    def update_info(self, paths_only: bool = False):
        # If running from free-form user story, create a dedicated results folder name.
        if getattr(self.args, "user_story", None):
            folder_name = "UserStory"
        else:
            folder_name = self.script_path.split("/")[-2]

        self.save_path = f"./Results/{folder_name}"

        model_config = self.args.LLM + "_" + self.sample_model + "_" + self.args.Image2Video 
        self.video_save_path = os.path.join(self.save_path, model_config, "video")

        self.user_story_synopsis_path = os.path.join(self.save_path, model_config, "Step_-1_script_synopsis.json")

        self.sub_script_path = os.path.join(self.save_path, model_config, f"Step_1_script_results.json")
        self.scene_path = os.path.join(self.save_path, model_config, f"Step_2_scene_results.json")
        self.shot_path = os.path.join(self.save_path, model_config, f"Step_3_shot_results.json")
        self.character_profiles_path = os.path.join(self.save_path, model_config, f"Step_0_character_profiles.json")

        os.makedirs(self.save_path, exist_ok=True)
        os.makedirs(self.video_save_path, exist_ok=True)

        # If user_story is provided, generate synopsis json and set script_path
        if (not paths_only) and getattr(self.args, "user_story", None):
            self._ensure_user_story_synopsis()
            self.script_path = self.user_story_synopsis_path

    def _ensure_user_story_synopsis(self):
        if os.path.exists(self.user_story_synopsis_path):
            return
        query = f"""
User Story:
{self.args.user_story}

Return JSON only.
"""
        resp = self.user_story_agent(query, parse=True)
        save_json(resp, self.user_story_synopsis_path)

    def read_json(self, file_path):
        if not os.path.exists(file_path):
            raise FileNotFoundError(
                f"[MovieAgent] Required JSON not found: {file_path}\n"
                "This usually means you started from a later stage (e.g. --start_from video) "
                "but the earlier stage outputs were never generated for this run/config.\n"
                "Fix: run once with --start_from script (or scene/shot), or use --resume without "
                "explicit --start_from so it can auto-pick the right stage."
            )
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data

    def extract_characters_from_json(self,file_path, n):
        data = self.read_json(file_path)
        movie_script = data['MovieScript']
        characters = data['Character']
        selected_characters = characters[:n]
        self.characters_list = selected_characters
        return movie_script,selected_characters

    def ScriptBreak(self, all_chat=[]):
        

        movie_script, characters_list = self.extract_characters_from_json(self.script_path, 40)
        first_sentence = movie_script.split(".")[0]
        # all_chat.append(query)
        previous_sub_script = None
        n = 0
        index = 1
        result = {}
        characters_list = str(characters_list)
        while True:
            if previous_sub_script: 
                query = f"""
                    Script Synopsis: {movie_script}
                    Character: {characters_list}
                    Previous Sub-Script: {previous_sub_script}
                    """
            else:
                query = f"""
                    Script Synopsis: {movie_script}
                    Character: {characters_list}
                    There is no Previous Sub-Script. 
                    The current sub-script is the first one. Please start summarizing the first sub-script based on the following content: {first_sentence}.
                    """
                
            query = f"""
                    Script Synopsis: {movie_script}
                    Character: {characters_list}
                    """
            
            task_response = self.screenwriter_agent(query, parse=True)
            # task_response = task_response.replace("'",'"')
            result = task_response

            break
        
        # all_chat.append(self.task_agent.messages)
        save_json(result, self.sub_script_path)
        # return 

    def ScenePlanning(self):
        data = self.read_json(self.sub_script_path)
        data_scene = data
        
        character_relationships = data['Relationships']
        sub_script_list = data['Sub-Script']

        for sub_script_name in sub_script_list:
            sub_script = sub_script_list[sub_script_name]["Plot"]
            query = f"""
                        Given the following inputs:
                        - Script Synopsis: "{sub_script}"
                        - Character Relationships: {character_relationships}
                        """
            task_response = self.sceneplanning_agent(query, parse=True)
            # if "Scene Annotation" not in data_scene[sub_script_name]:
            #     data_scene[sub_script_name]["Scene Annotation"] = []
            
            data_scene['Sub-Script'][sub_script_name]["Scene Annotation"]=task_response

            save_json(data_scene, self.scene_path)
            # break
    
    def ShotPlotCreate(self):
        data = self.read_json(self.scene_path)
        data_scene = data
        
        character_relationships = data['Relationships']
        sub_script_list = data['Sub-Script']

        for sub_script_name in sub_script_list:
            scene_list = sub_script_list[sub_script_name]["Scene Annotation"]["Scene"]
            for scene_name in scene_list:
                scene_details = scene_list[scene_name]
                query = f"""
                            Given the following Scene Details:
                            - Involving Characters: "{scene_details['Involving Characters']}" 
                            - Plot: "{scene_details['Plot']}"
                            - Scene Description: "{scene_details['Scene Description']}"
                            - Key Props: {scene_details['Key Props']}
                            - Cinematography Notes: "{scene_details['Cinematography Notes']}"
                            """
                #- Emotional Tone: "{scene_details['Emotional Tone']}"先去掉            
                task_response = self.shotplotcreate_agent(query, parse=True)
                # if "Shot Annotation" not in data_scene[sub_script_name]:
                #     data_scene[sub_script_name]["Shot Annotation"] = []
                
                data_scene['Sub-Script'][sub_script_name]["Scene Annotation"]["Scene"][scene_name]["Shot Annotation"] = task_response

                save_json(data_scene, self.shot_path)
            #     break
        
    def CharacterProfiles(self):
        """Generate global character profile JSON for Flux CharacterBank."""
        if os.path.exists(self.character_profiles_path):
            return

        data = self.read_json(self.script_path)
        movie_script = data.get('MovieScript', '')
        characters = data.get('Character', [])

        query = sys_prompts["CharacterProfileJSON-sys"].format(SCRIPT=movie_script, CHARACTERS=str(characters))
        task_response = self.character_profile_agent(query, parse=True)
        save_json(task_response, self.character_profiles_path)

    def _build_motion_prefix(self, shot_info: dict, scene_details: dict | None = None) -> str:
        """Build a short, explicit cinematography/motion instruction prefix for I2V.

        NOTE: Keep this compact. CLIP-like text encoders may truncate from the end in some setups,
        so the *most important visual content* should be placed later in the final prompt.

        We intentionally:
        - DO NOT include Emotional tone (too verbose / low ROI)
        - DO NOT include Emotional emphasis (often very long)
        - Keep only: shot type, camera movement, action intent
        """

        def _get(d, k):
            if not isinstance(d, dict):
                return None
            v = d.get(k)
            if v is None:
                return None
            if isinstance(v, str):
                v = v.strip()
                return v if v else None
            return v

        shot_type = _get(shot_info, "Shot Type")
        cam_move = _get(shot_info, "Camera Movement")
        coarse_plot = _get(shot_info, "Coarse Plot")

        # Build a single natural-language sentence (no field labels).
        # Keep order: cinematography first, intent last.
        parts: list[str] = []
        if shot_type:
            parts.append(f"{shot_type}")
        if cam_move:
            parts.append(f"{cam_move}")

        directive = None
        if parts and coarse_plot:
            directive = ", ".join(parts) + f"; {coarse_plot}."
        elif parts:
            directive = ", ".join(parts) + "."
        elif coarse_plot:
            directive = f"{coarse_plot}."

        if not directive:
            return ""

        return "[CINEMATIC DIRECTIVE] " + directive + "\n"

    def _global_style_prefix(self) -> str:
        """Global style prefix applied to ALL shots.

        Default is intentionally minimal to save tokens.

        Control by env:
        - MOVIEAGENT_GLOBAL_STYLE: e.g. "Pixar-like 3D animation"
        - MOVIEAGENT_GLOBAL_QUALITY: e.g. "ultra-detailed"
        - MOVIEAGENT_GLOBAL_STYLE_NEG: negative style constraints

        IMPORTANT: We do NOT add the legacy marker "[GLOBAL STYLE BIBLE]" to reduce clutter.
        """

        style = (os.getenv("MOVIEAGENT_GLOBAL_STYLE") or "").strip()
        quality = (os.getenv("MOVIEAGENT_GLOBAL_QUALITY") or "").strip()
        neg = (os.getenv("MOVIEAGENT_GLOBAL_STYLE_NEG") or "").strip()

        # Defaults requested: short + stable.
        if not style:
            style = "Pixar-like 3D animation"
        if not quality:
            quality = "ultra-detailed"

        parts: list[str] = [f"{style}", f"{quality}"]

        # Keep negative style lightweight.
        if neg:
            parts.append(f"avoid: {neg}")
        if not neg:
            neg = "deformation, a poor composition and deformed video, bad teeth, bad eyes, bad limbs,low quality,bad anatomy,deformed,duplicate face,blurry"
            parts.append(f"avoid: {neg}")
        return ", ".join([p for p in parts if p]) + ".\n"

    def _get_global_style_keywords_from_profiles(self) -> str | None:
        """Try to read global style keywords from Step_0_character_profiles.json.

        Returns a compact comma-separated string or None.
        """
        try:
            if not os.path.exists(self.character_profiles_path):
                return None
            with open(self.character_profiles_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            style = (data or {}).get("style") or {}
            kws = style.get("global_style_keywords")
            if isinstance(kws, list):
                kws = [str(x).strip() for x in kws if str(x).strip()]
                if kws:
                    return ", ".join(kws)
            if isinstance(kws, str) and kws.strip():
                return kws.strip()
        except Exception:
            return None
        return None

    def _assemble_shot_prompt(self, shot_info: dict, scene_details: dict | None = None, global_style_prefix: str = "") -> str:
        """Assemble final positive prompt using compact structured fields.

        Preferred order (short -> important at end):
        style_keywords + cinematic_directive + visual_description (+ background/props)

        If fields are missing (old cache), fall back to legacy keys.
        """

        def _s(v):
            if v is None:
                return ""
            if isinstance(v, str):
                return v.strip()
            return str(v).strip()

        cinematic = _s(shot_info.get("cinematic_directive"))
        visual = _s(shot_info.get("visual_description"))
        background = _s(shot_info.get("background"))
        negative = _s(shot_info.get("negative"))
        props = shot_info.get("props")
        props_str = ""
        if isinstance(props, list):
            props_clean = [str(p).strip() for p in props if str(p).strip()]
            if props_clean:
                props_str = ", ".join(props_clean)
        elif isinstance(props, str) and props.strip():
            props_str = props.strip()

        # Fallback for old schema (existing Step_3_shot_results.json)
        if not visual:
            visual = _s(shot_info.get("Plot/Visual Description"))
        if not cinematic:
            cinematic = self._build_motion_prefix(shot_info, scene_details).replace("[CINEMATIC DIRECTIVE]", "").strip()

        # Natural language join (no extra labels)
        parts: list[str] = []
        if global_style_prefix:
            parts.append(global_style_prefix.strip())
        if cinematic:
            parts.append(cinematic.rstrip(".") + ".")
        if background:
            parts.append(background.rstrip(".") + ".")
        if props_str:
            parts.append(("props: " + props_str).rstrip(".") + ".")
        if visual:
            parts.append(visual)

        prompt = "\n".join([p for p in parts if p])

        # Simple length control: if too long, keep tail (user observed tail is important).
        # Char-based heuristic: avoids dependency on tokenizer.
        max_chars = int(os.getenv("MOVIEAGENT_MAX_PROMPT_CHARS", "500"))
        if max_chars > 0 and len(prompt) > max_chars:
            prompt = prompt[-max_chars:]
            # avoid starting mid-word
            if " " in prompt:
                prompt = prompt.split(" ", 1)[-1]

        # Attach negative as a lightweight suffix (some models read it; harmless otherwise)
        if negative:
            prompt = prompt + "\n" + ("avoid: " + negative)

        return prompt

    def _infer_identity_view_for_shot(self, shot_info: dict, plot_prompt: str) -> str:
        """Infer desired character identity view: front|side|back.

        We keep this lightweight and heuristic-based (no left/right distinction).
        Priority:
        - If shot JSON provides per-role view mapping: use it (handled by _get_role_view_from_shot()).
        - explicit "back view/rear view" -> back
        - explicit "side view/profile" -> side
        - otherwise -> front
        """
        txt = ""
        try:
            txt = " ".join(
                [
                    str(shot_info.get("Camera Angle") or ""),
                    str(shot_info.get("Shot Type") or ""),
                    str(shot_info.get("Plot/Visual Description") or ""),
                    str(shot_info.get("visual_description") or ""),
                    str(plot_prompt or ""),
                ]
            )
        except Exception:
            txt = str(plot_prompt or "")

        t = txt.lower()
        if any(k in t for k in ["back view", "rear view", "from behind", "seen from behind", "背面", "背影", "从背后"]):
            return "back"
        if any(k in t for k in ["side view", "side profile", "profile view", "in profile", "侧面", "侧身"]):
            return "side"
        return "front"

    def _get_role_view_from_shot(self, shot_info: dict, role: str, plot_prompt: str) -> str:
        """Get per-role desired view from Step_3 shot design if present.

        Expected new field from ShotPlotCreateCoT-sys:
        - shot_info["character_views"] = { "<RoleName>": "front|side|back", ... }

        Backward-compatible aliases accepted:
        - "Character Views", "character_view", "Character View", etc.
        """
        mapping = None
        if isinstance(shot_info, dict):
            for k in [
                "character_views",
                "Character Views",
                "Character views",
                "character_view",
                "Character View",
                "characterView",
            ]:
                v = shot_info.get(k)
                if isinstance(v, dict):
                    mapping = v
                    break

        if isinstance(mapping, dict):
            vv = mapping.get(role)
            if isinstance(vv, str) and vv.strip():
                s = vv.strip().lower()
                if s in ["front", "f"]:
                    return "front"
                if s in ["side", "profile", "s"]:
                    return "side"
                if s in ["back", "rear", "b"]:
                    return "back"

        # fallback: global heuristic (applies to all roles)
        return self._infer_identity_view_for_shot(shot_info, plot_prompt=plot_prompt)

    def VideoAudioGen(self):
        # Step_3_shot_results.json is mandatory for video generation.
        data = self.read_json(self.shot_path)
        sub_script_list = data['Sub-Script']

        # Ensure Character profiles exist (Flux input)
        self.CharacterProfiles()

        # Build/load global CharacterBank assets once.
        # Default strategy for InstantCharacter keyframes: use CharaConsist to build
        # front/side/back identity references, then inject per-shot view in scene generation.
        if self.sample_model == "InstantCharacter":
            os.environ.setdefault("MOVIEAGENT_BANK_BUILDER", "characonsist")
            os.environ.setdefault("MOVIEAGENT_BANK_MULTI_VIEW", "1")

        roles = roles_from_script_json(self.script_path, 40)
        bank_root = self.args.character_bank_root or os.path.join(self.save_path, self.args.LLM + "_" + self.sample_model + "_" + self.args.Image2Video, "character_bank")
        bank = CharacterBank(bank_root)
        bank_assets = bank.ensure(
            roles=roles,
            placeholders_from=self.character_photo_path,
            profiles_json_path=self.character_profiles_path,
            use_flux=True,
        )

        # Global style prefix (same for all shots)
        # Priority:
        # 1) env MOVIEAGENT_GLOBAL_STYLE / QUALITY / NEG
        # 2) exported style keywords from character profiles (Step_0_character_profiles.json)
        exported_style = self._get_global_style_keywords_from_profiles()
        if exported_style and not os.getenv("MOVIEAGENT_GLOBAL_STYLE"):
            os.environ["MOVIEAGENT_GLOBAL_STYLE"] = exported_style
        global_style_prefix = self._global_style_prefix()

        # --- UPDATED: support flexible --only (order-independent) ---
        only_sub_script = None
        only_scene = None
        only_shot = None
        if getattr(self.args, "only", None):
            try:
                parts = [p.strip() for p in str(self.args.only).split(",") if p.strip()]

                def _norm(s: str) -> str:
                    return str(s).strip().lower().replace("_", " ")

                for p in parts:
                    pl = _norm(p)
                    if pl.startswith("sub-script"):
                        only_sub_script = p.strip()
                    elif pl.startswith("scene"):
                        only_scene = p.strip()
                    elif pl.startswith("shot"):
                        only_shot = p.strip()

                # Backward compatible fallback for old positional format
                if (only_scene is None and only_shot is None and only_sub_script is None) and parts:
                    if len(parts) == 1:
                        only_scene = parts[0]
                    elif len(parts) == 2:
                        only_scene, only_shot = parts[0], parts[1]
                    else:
                        only_sub_script, only_scene, only_shot = parts[0], parts[1], parts[2]
            except Exception:
                only_sub_script = None
                only_scene = None
                only_shot = None

        matched_any = False

        # If images_only: we still call tools.sample to produce the keyframe image,
        # but we avoid any downstream video/final stage assumptions.
        images_only = bool(getattr(self.args, "images_only", False))

        for idx_1, sub_script_name in enumerate(sub_script_list):
            if only_sub_script and sub_script_name != only_sub_script:
                continue

            scene_list = sub_script_list[sub_script_name]["Scene Annotation"]["Scene"]

            for scene_name in scene_list:
                if only_scene and scene_name != only_scene:
                    continue

                scene_details = scene_list[scene_name]

                # --- Robust shot list extraction (avoid KeyError on variant Step_3 schema) ---
                shot_anno = (
                    scene_details.get("Shot Annotation")
                    or scene_details.get("Shot")
                    or scene_details.get("Shots")
                    or scene_details.get("Shot_Annotation")
                )
                if isinstance(shot_anno, dict):
                    shot_lists = shot_anno.get("Shot")
                else:
                    shot_lists = None

                if not isinstance(shot_lists, dict):
                    print(
                        f"[MovieAgent][WARN] Missing shot list in {sub_script_name}/{scene_name}. "
                        f"Available keys: {list(scene_details.keys())} (shot_path={self.shot_path})"
                    )
                    continue

                for shot_name in shot_lists:
                    if only_shot and shot_name != only_shot:
                        continue

                    matched_any = True

                    shot_info = shot_lists[shot_name]

                    # --- NEW: assemble prompt from structured fields (or fallback) ---
                    plot = self._assemble_shot_prompt(
                        shot_info=shot_info,
                        scene_details=scene_details,
                        global_style_prefix=global_style_prefix,
                    )

                    involving = shot_info["Involving Characters"]
                    subtitle = shot_info["Subtitles"]

                    if isinstance(involving, dict):
                        character_roles = list(involving.keys())
                        character_boxes = involving
                    else:
                        character_roles = list(involving)
                        character_boxes = involving

                    # Build reference images.
                    # - For InstantCharacter multi-person: pass role->single_ref_image to match character_boxes keys.
                    # - For other models: keep legacy flat list.
                    prefer = self.args.bank_prefer
                    bank_multi_view_on = os.getenv("MOVIEAGENT_BANK_MULTI_VIEW", "0").strip() not in ["0", "false", "False"]
                    if self.sample_model == "InstantCharacter" and prefer == "identity" and bank_multi_view_on:
                        # In multi-view mode, auto route to per-shot view unless user explicitly requested another asset type.
                        prefer = "identity_auto"

                    if self.sample_model == "InstantCharacter" and isinstance(character_boxes, dict):
                        character_phot_list = {}
                        for role in character_roles:
                            a = bank_assets.get(role) or bank.assets(role)
                            if prefer == "identity_auto":
                                v = self._get_role_view_from_shot(shot_info, role=role, plot_prompt=plot)
                                # Choose view-specific identity if available, else fallback to identity.png
                                p = a.identity_path_for_view(v)
                                if p and os.path.exists(p):
                                    character_phot_list[role] = p
                                else:
                                    refs = a.as_list(prefer="identity")
                                    if refs:
                                        character_phot_list[role] = refs[0]
                            else:
                                refs = a.as_list(prefer=prefer)
                                if refs:
                                    character_phot_list[role] = refs[0]
                    else:
                        character_phot_list = []
                        for role in character_roles:
                            a = bank_assets.get(role) or bank.assets(role)
                            if prefer == "identity_auto":
                                v = self._get_role_view_from_shot(shot_info, role=role, plot_prompt=plot)
                                p = a.identity_path_for_view(v)
                                if p and os.path.exists(p):
                                    character_phot_list.append(p)
                                else:
                                    character_phot_list.extend(a.as_list(prefer="identity"))
                            else:
                                character_phot_list.extend(a.as_list(prefer=prefer))

                    # Use PNG to avoid JPEG artifacts (helps I2V sharpness)
                    save_path = os.path.join(self.video_save_path, sub_script_name + "|" + scene_name + "|" + shot_name + ".png")
                    save_path = save_path.replace(" ", "_")

                    if images_only:
                        print("[MovieAgent] images_only=1 -> generating keyframe image only:", save_path)
                    else:
                        print("Save the video to path:", save_path)

                    # Higher resolution keyframe for I2V (must be aligned to 64; 1280x720 is OK)
                    # tools.sample will generate the keyframe image; in images_only mode we rely on it
                    # to stop after image generation.
                    self.tools.sample(plot, character_phot_list, character_boxes, subtitle, save_path, (1280, 720))

                    if images_only:
                        # Skip any subsequent per-subtitle audio/video logic (reserved for full pipeline)
                        continue

                    for i, name in enumerate(subtitle):
                        wave_path = save_path.replace(".png", "") + "_" + str(i) + "_" + name + ".wav"
                        image_path = save_path
                        text_prompt = subtitle[name]

        if getattr(self.args, "only", None) and (not matched_any):
            print(
                f"[MovieAgent][WARN] --only='{self.args.only}' matched nothing. "
                "Expected format like 'Scene 2,Shot 1' or 'Sub-Script 1,Scene 2,Shot 1'."
            )
    def Final(self):
        # Lazy import: Final() is skipped when --only is used.
        from moviepy.editor import VideoFileClip, concatenate_videoclips

        directory = self.video_save_path
        mp4_files = [f for f in os.listdir(directory) if f.endswith('.mp4')]

        mp4_files.sort()

        clips = []
        for file in mp4_files:
            file_path = os.path.join(directory, file)
            clip = VideoFileClip(file_path)
            clips.append(clip)

        final_video = concatenate_videoclips(clips)

        final_video_path = os.path.join(directory, "final_video.mp4")
        final_video.write_videofile(final_video_path, codec="libx264")
            
def main():
    args = parse_args()
    script_path = args.script_path
    character_photo_path = args.character_photo_path
    movie_director = ScriptBreakAgent(args,sample_model=args.gen_model, audio_model=args.audio_model, \
                                      talk_model=args.talk_model, Image2Video=args.Image2Video, script_path = script_path, \
                                    character_photo_path=character_photo_path, \
                                    save_mode="video")

    # stage control
    # default behavior: run everything
    start_from = args.start_from

    # If resume is enabled, skip stages whose result json already exists.
    if args.resume and start_from is None:
        if os.path.exists(movie_director.shot_path):
            start_from = "video"
        elif os.path.exists(movie_director.scene_path):
            start_from = "shot"
        elif os.path.exists(movie_director.sub_script_path):
            start_from = "scene"
        else:
            start_from = "script"

    if start_from is None:
        start_from = "script"

    # When generating a single shot via --only, do NOT auto-concatenate all existing mp4 files.
    # (Otherwise old mp4s in the folder will be merged into final_video.mp4.)
    skip_final_when_only = bool(getattr(args, "only", None))

    if start_from == "script":
        movie_director.ScriptBreak()
        movie_director.ScenePlanning()
        movie_director.ShotPlotCreate()
        movie_director.VideoAudioGen()
        if not skip_final_when_only:
            movie_director.Final()
    elif start_from == "scene":
        # assume Step_1 exists
        movie_director.ScenePlanning()
        movie_director.ShotPlotCreate()
        movie_director.VideoAudioGen()
        if not skip_final_when_only:
            movie_director.Final()
    elif start_from == "shot":
        # assume Step_2 exists
        movie_director.ShotPlotCreate()
        movie_director.VideoAudioGen()
        if not skip_final_when_only:
            movie_director.Final()
    elif start_from == "video":
        # assume Step_3 exists
        movie_director.VideoAudioGen()
        if not skip_final_when_only:
            movie_director.Final()
    elif start_from == "final":
        movie_director.Final()


if __name__ == "__main__":
    main()











