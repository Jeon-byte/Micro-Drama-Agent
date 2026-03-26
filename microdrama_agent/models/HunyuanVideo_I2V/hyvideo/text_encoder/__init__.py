from dataclasses import dataclass
from typing import Optional, Tuple
from copy import deepcopy

import os
import time

import torch
import torch.nn as nn
from transformers import (
    CLIPTextModel,
    CLIPTokenizer,
    AutoTokenizer,
    AutoProcessor,
    AutoModel,
    LlavaForConditionalGeneration,
    CLIPImageProcessor,
)
from transformers.utils import ModelOutput

from ..constants import TEXT_ENCODER_PATH, TOKENIZER_PATH
from ..constants import PRECISION_TO_TYPE


def use_default(value, default):
    return value if value is not None else default


def _sanitize_omp_threads(logger=None):
    """Avoid libgomp hangs due to invalid OMP_NUM_THREADS.

    We saw cases where env var exists but is not an int ("", "None", etc), causing:
    `libgomp: Invalid value for environment variable OMP_NUM_THREADS`.

    In that case, force it to "1".
    """
    v = os.environ.get("OMP_NUM_THREADS")
    if v is None:
        return
    s = str(v).strip()
    if not s.isdigit() or int(s) <= 0:
        os.environ["OMP_NUM_THREADS"] = "1"
        if logger is not None:
            logger.warning(
                f"[hyvideo.text_encoder] Invalid OMP_NUM_THREADS={v!r}; fallback to 1 to avoid libgomp issues"
            )


def load_text_encoder(
    text_encoder_type,
    text_encoder_precision=None,
    text_encoder_path=None,
    logger=None,
    device=None,
):
    if text_encoder_path is None:
        text_encoder_path = TEXT_ENCODER_PATH[text_encoder_type]
    if logger is not None:
        logger.info(
            f"Loading text encoder model ({text_encoder_type}) from: {text_encoder_path}"
        )

    target_dtype = PRECISION_TO_TYPE[text_encoder_precision] if text_encoder_precision is not None else None

    # Lower load-time memory peak: try loading directly in target dtype.
    # Extra knobs (optional):
    # - HUNYUAN_TEXT_ENCODER_LOW_CPU_MEM=1 (default)
    # - HUNYUAN_TEXT_ENCODER_DEVICE_MAP=cpu|auto|... (optional)
    load_kwargs = {}
    if target_dtype is not None:
        load_kwargs["torch_dtype"] = target_dtype

    if os.getenv("HUNYUAN_TEXT_ENCODER_LOW_CPU_MEM", "1") not in ("0", "false", "False", ""):
        load_kwargs["low_cpu_mem_usage"] = True

    dm = os.getenv("HUNYUAN_TEXT_ENCODER_DEVICE_MAP", "").strip()
    if dm:
        load_kwargs["device_map"] = dm

    def _safe_load(cls, path, kwargs):
        try:
            return cls.from_pretrained(path, **kwargs)
        except TypeError:
            # Compatibility fallback for older transformers versions
            k = dict(kwargs)
            k.pop("device_map", None)
            try:
                return cls.from_pretrained(path, **k)
            except TypeError:
                k.pop("low_cpu_mem_usage", None)
                try:
                    return cls.from_pretrained(path, **k)
                except TypeError:
                    k.pop("torch_dtype", None)
                    return cls.from_pretrained(path, **k)

    if logger is not None:
        logger.info(f"text_encoder from_pretrained kwargs: {load_kwargs}")

    if text_encoder_type == "clipL":
        text_encoder = _safe_load(CLIPTextModel, text_encoder_path, load_kwargs)
        text_encoder.final_layer_norm = text_encoder.text_model.final_layer_norm
    elif text_encoder_type == "llm":
        text_encoder = _safe_load(AutoModel, text_encoder_path, load_kwargs)
        text_encoder.final_layer_norm = text_encoder.norm
    elif text_encoder_type == "llm-i2v":
        text_encoder = _safe_load(LlavaForConditionalGeneration, text_encoder_path, load_kwargs)
    else:
        raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")
    # from_pretrained will ensure that the model is in eval mode.

    # If direct-dtype loading didn't take effect, cast once here.
    if target_dtype is not None and text_encoder.dtype != target_dtype:
        text_encoder = text_encoder.to(dtype=target_dtype)

    text_encoder.requires_grad_(False)

    if logger is not None:
        logger.info(f"Text encoder to dtype: {text_encoder.dtype}")

    if device is not None:
        text_encoder = text_encoder.to(device)

    return text_encoder, text_encoder_path


def load_tokenizer(
    tokenizer_type, tokenizer_path=None, padding_side="right", logger=None
):
    # sanitize OMP threads before tokenizer init (fast tokenizer can touch OpenMP)
    _sanitize_omp_threads(logger=logger)

    if tokenizer_path is None:
        tokenizer_path = TOKENIZER_PATH[tokenizer_type]
    if logger is not None:
        logger.info(f"Loading tokenizer ({tokenizer_type}) from: {tokenizer_path}")

    t0 = time.time()

    processor = None
    if tokenizer_type == "clipL":
        tokenizer = CLIPTokenizer.from_pretrained(tokenizer_path, max_length=77)
    elif tokenizer_type == "llm":
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_path, padding_side=padding_side
        )
    elif tokenizer_type == "llm-i2v":
        processor = AutoProcessor.from_pretrained(tokenizer_path)
        tokenizer = getattr(processor, "tokenizer", None)
        if tokenizer is None:
            # Fallback for unusual processor packages.
            tokenizer = AutoTokenizer.from_pretrained(
                tokenizer_path, padding_side=padding_side
            )
        else:
            try:
                tokenizer.padding_side = padding_side
            except Exception:
                pass
    else:
        raise ValueError(f"Unsupported tokenizer type: {tokenizer_type}")

    if logger is not None:
        logger.info(
            f"Tokenizer ({tokenizer_type}) ready in {time.time() - t0:.2f}s (OMP_NUM_THREADS={os.environ.get('OMP_NUM_THREADS')})"
        )

    return tokenizer, tokenizer_path, processor


@dataclass
class TextEncoderModelOutput(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.

    Args:
        hidden_state (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        attention_mask (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in ``[0, 1]``:
        hidden_states_list (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        text_outputs (`list`, *optional*, returned when `return_texts=True` is passed):
            List of decoded texts.
    """

    hidden_state: torch.FloatTensor = None
    attention_mask: Optional[torch.LongTensor] = None
    hidden_states_list: Optional[Tuple[torch.FloatTensor, ...]] = None
    text_outputs: Optional[list] = None


class TextEncoder(nn.Module):
    def __init__(
        self,
        text_encoder_type: str,
        max_length: int,
        text_encoder_precision: Optional[str] = None,
        text_encoder_path: Optional[str] = None,
        tokenizer_type: Optional[str] = None,
        tokenizer_path: Optional[str] = None,
        output_key: Optional[str] = None,
        use_attention_mask: bool = True,
        i2v_mode: bool = False,
        input_max_length: Optional[int] = None,
        prompt_template: Optional[dict] = None,
        prompt_template_video: Optional[dict] = None,
        hidden_state_skip_layer: Optional[int] = None,
        apply_final_norm: bool = False,
        reproduce: bool = False,
        logger=None,
        device=None,
        image_embed_interleave=None,
    ):
        super().__init__()
        self.text_encoder_type = text_encoder_type
        self.max_length = max_length
        self.precision = text_encoder_precision
        self.model_path = text_encoder_path
        self.tokenizer_type = (
            tokenizer_type if tokenizer_type is not None else text_encoder_type
        )
        self.tokenizer_path = (
            tokenizer_path if tokenizer_path is not None else text_encoder_path
        )
        self.use_attention_mask = use_attention_mask
        if prompt_template_video is not None:
            assert (
                use_attention_mask is True
            ), "Attention mask is True required when training videos."
        self.input_max_length = (
            input_max_length if input_max_length is not None else max_length
        )
        self.prompt_template = prompt_template
        self.prompt_template_video = prompt_template_video
        self.hidden_state_skip_layer = hidden_state_skip_layer
        self.apply_final_norm = apply_final_norm
        self.i2v_mode = i2v_mode
        self.reproduce = reproduce
        self.logger = logger
        self.image_embed_interleave = image_embed_interleave

        self.use_template = self.prompt_template is not None
        if self.use_template:
            assert (
                isinstance(self.prompt_template, dict)
                and "template" in self.prompt_template
            ), f"`prompt_template` must be a dictionary with a key 'template', got {self.prompt_template}"
            assert "{}" in str(self.prompt_template["template"]), (
                "`prompt_template['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template['template']}"
            )

        self.use_video_template = self.prompt_template_video is not None
        if self.use_video_template:
            if self.prompt_template_video is not None:
                assert (
                    isinstance(self.prompt_template_video, dict)
                    and "template" in self.prompt_template_video
                ), f"`prompt_template_video` must be a dictionary with a key 'template', got {self.prompt_template_video}"
            assert "{}" in str(self.prompt_template_video["template"]), (
                "`prompt_template_video['template']` must contain a placeholder `{}` for the input text, "
                f"got {self.prompt_template_video['template']}"
            )

        if "t5" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        elif "clip" in text_encoder_type:
            self.output_key = output_key or "pooler_output"
        elif "llm" in text_encoder_type or "glm" in text_encoder_type:
            self.output_key = output_key or "last_hidden_state"
        else:
            raise ValueError(f"Unsupported text encoder type: {text_encoder_type}")

        self.model, self.model_path = load_text_encoder(
            text_encoder_type=self.text_encoder_type,
            text_encoder_precision=self.precision,
            text_encoder_path=self.model_path,
            logger=self.logger,
            device=device,
        )
        self.dtype = self.model.dtype
        self.device = self.model.device

        self.tokenizer, self.tokenizer_path, self.processor = load_tokenizer(
            tokenizer_type=self.tokenizer_type,
            tokenizer_path=self.tokenizer_path,
            padding_side="right",
            logger=self.logger,
        )

        # For llava i2v checkpoints, preprocessor_config may miss these fields.
        # Align processor settings with model config to avoid token/feature mismatch.
        if self.i2v_mode and (self.processor is not None):
            try:
                model_cfg = getattr(self.model, "config", None)
                vision_cfg = getattr(model_cfg, "vision_config", None)

                patch_size = getattr(vision_cfg, "patch_size", None)
                if patch_size is not None:
                    self.processor.patch_size = int(patch_size)

                strategy = getattr(model_cfg, "vision_feature_select_strategy", None)
                if strategy is not None:
                    self.processor.vision_feature_select_strategy = str(strategy)

                image_token = getattr(self.tokenizer, "image_token", None)
                if image_token is None:
                    image_token_index = getattr(model_cfg, "image_token_index", None)
                    if image_token_index is not None and hasattr(self.tokenizer, "convert_ids_to_tokens"):
                        try:
                            image_token = self.tokenizer.convert_ids_to_tokens(image_token_index)
                        except Exception:
                            image_token = None
                if image_token is not None:
                    self.processor.image_token = image_token

                # CLIP vision backbones include a CLS token; set +1 so
                # (h/p * w/p + 1 - 1) keeps the expected patch-token length.
                num_additional = getattr(model_cfg, "num_additional_image_tokens", None)
                if num_additional is None:
                    is_clip_vision = str(getattr(vision_cfg, "model_type", "")).startswith("clip")
                    num_additional = 1 if is_clip_vision else 0
                self.processor.num_additional_image_tokens = int(num_additional)

                if self.logger is not None:
                    self.logger.info(
                        "[hyvideo.text_encoder] llava processor aligned: "
                        f"patch_size={getattr(self.processor, 'patch_size', None)}, "
                        f"vision_feature_select_strategy={getattr(self.processor, 'vision_feature_select_strategy', None)}, "
                        f"num_additional_image_tokens={getattr(self.processor, 'num_additional_image_tokens', None)}, "
                        f"image_token={getattr(self.processor, 'image_token', None)}"
                    )
            except Exception as e:
                if self.logger is not None:
                    self.logger.warning(f"[hyvideo.text_encoder] failed to align llava processor config: {e}")

    def __repr__(self):
        return f"{self.text_encoder_type} ({self.precision} - {self.model_path})"

    @staticmethod
    def apply_text_to_template(text, template, prevent_empty_text=True):
        """
        Apply text to template.

        Args:
            text (str): Input text.
            template (str or list): Template string or list of chat conversation.
            prevent_empty_text (bool): If Ture, we will prevent the user text from being empty
                by adding a space. Defaults to True.
        """
        if isinstance(template, str):
            # Will send string to tokenizer. Used for llm
            return template.format(text)
        else:
            raise TypeError(f"Unsupported template type: {type(template)}")

    # NOTE: i2v token/image alignment is now handled by AutoProcessor in text2tokens.

    def text2tokens(self, text, data_type="image", semantic_images=None):
        """
        Tokenize the input text.

        Args:
            text (str or list): Input text.
        """
        t0 = time.time()
        tokenize_input_type = "str"
        if self.use_template:
            if data_type == "image":
                prompt_template = self.prompt_template["template"]
            elif data_type == "video":
                prompt_template = self.prompt_template_video["template"]
            else:
                raise ValueError(f"Unsupported data type: {data_type}")
            if isinstance(text, (list, tuple)):
                text = [
                    self.apply_text_to_template(one_text, prompt_template)
                    for one_text in text
                ]
                if isinstance(text[0], list):
                    tokenize_input_type = "list"
            elif isinstance(text, str):
                text = self.apply_text_to_template(text, prompt_template)
                if isinstance(text, list):
                    tokenize_input_type = "list"
            else:
                raise TypeError(f"Unsupported text type: {type(text)}")

        max_length = self.max_length
        if self.i2v_mode and data_type in ["image", "video"]:
            tpl = self.prompt_template_video if data_type == "video" else self.prompt_template
            image_emb_len = int((tpl or {}).get("image_emb_len", 576) or 576)
            # Keep enough room after image-token expansion.
            max_length = max(int(self.max_length), int(self.max_length) + image_emb_len)

        kwargs = dict(
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt",
        )
        use_auto_processor = bool(
            self.i2v_mode
            and data_type in ["image", "video"]
            and (self.processor is not None)
            and (semantic_images is not None)
        )

        if use_auto_processor:
            # Canonical llm-i2v path: build aligned input_ids + pixel_values together.
            proc_kwargs = dict(
                text=text,
                images=semantic_images,
                return_tensors="pt",
                truncation=True,
                max_length=max_length,
                padding="max_length",
            )
            out = self.processor(**proc_kwargs)
            if "attention_mask" not in out:
                out["attention_mask"] = (out["input_ids"] != self.tokenizer.pad_token_id).long()
            if self.logger is not None:
                try:
                    image_token_id = getattr(getattr(self.model, "config", None), "image_token_index", None)
                    n_img = int((out["input_ids"] == image_token_id).sum().item()) if image_token_id is not None else -1
                    self.logger.info(
                        f"[hyvideo.text_encoder] AutoProcessor i2v tokens: n_image_tokens={n_img}, max_length={max_length}, input_ids_shape={tuple(out['input_ids'].shape)}"
                    )
                except Exception:
                    pass
        else:
            if tokenize_input_type == "str":
                out = self.tokenizer(
                    text,
                    return_length=False,
                    return_overflowing_tokens=False,
                    return_attention_mask=True,
                    **kwargs,
                )
            elif tokenize_input_type == "list":
                out = self.tokenizer.apply_chat_template(
                    text,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    **kwargs,
                )
            else:
                raise ValueError(f"Unsupported tokenize_input_type: {tokenize_input_type}")

        if self.logger is not None:
            try:
                in_len = len(text) if isinstance(text, str) else len(text[0])
            except Exception:
                in_len = -1
            self.logger.info(
                f"[hyvideo.text_encoder] text2tokens done in {time.time() - t0:.2f}s; data_type={data_type}, tokenize_input_type={tokenize_input_type}, text_len={in_len}"
            )
        return out

    def encode(
        self,
        batch_encoding,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=None,
        hidden_state_skip_layer=None,
        return_texts=False,
        data_type="image",
        semantic_images=None,
        device=None,
    ):
        t0 = time.time()
        device = self.model.device if device is None else device
        use_attention_mask = use_default(use_attention_mask, self.use_attention_mask)
        hidden_state_skip_layer = use_default(
            hidden_state_skip_layer, self.hidden_state_skip_layer
        )
        do_sample = use_default(do_sample, not self.reproduce)
        if not self.i2v_mode:
            attention_mask = (
                batch_encoding["attention_mask"].to(device)
                if use_attention_mask
                else None
            )
            outputs = self.model(
                input_ids=batch_encoding["input_ids"].to(device),
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states
                or hidden_state_skip_layer is not None,
            )
            if hidden_state_skip_layer is not None:
                last_hidden_state = outputs.hidden_states[
                    -(hidden_state_skip_layer + 1)
                ]
                # Real last hidden state already has layer norm applied. So here we only apply it
                # for intermediate layers.
                if hidden_state_skip_layer > 0 and self.apply_final_norm:
                    last_hidden_state = self.model.final_layer_norm(last_hidden_state)
            else:
                last_hidden_state = outputs[self.output_key]

            # Remove hidden states of instruction tokens, only keep prompt tokens.
            if self.use_template:
                if data_type == "image":
                    crop_start = self.prompt_template.get("crop_start", -1)
                elif data_type == "video":
                    crop_start = self.prompt_template_video.get("crop_start", -1)
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")
                if crop_start > 0:
                    last_hidden_state = last_hidden_state[:, crop_start:]
                    attention_mask = (
                        attention_mask[:, crop_start:] if use_attention_mask else None
                    )

            if output_hidden_states:
                return TextEncoderModelOutput(
                    last_hidden_state, attention_mask, outputs.hidden_states
                )
            return TextEncoderModelOutput(last_hidden_state, attention_mask)
        else:
            # Prefer pixel_values already produced by AutoProcessor(text+images, ...)
            # to keep token/image alignment from a single preprocessing path.
            pixel_values = None
            try:
                pixel_values = batch_encoding.get("pixel_values", None)
            except Exception:
                if "pixel_values" in batch_encoding:
                    pixel_values = batch_encoding["pixel_values"]

            if pixel_values is not None:
                image_outputs = pixel_values.to(device)
            else:
                if semantic_images is None:
                    raise ValueError("i2v_mode requires semantic_images when pixel_values are missing")

                # Compatibility fallback: some transformers versions require text in
                # LlavaProcessor.__call__; use image_processor directly for image-only path.
                if self.processor is not None and hasattr(self.processor, "image_processor"):
                    image_outputs = self.processor.image_processor(
                        semantic_images, return_tensors="pt"
                    )["pixel_values"].to(device)
                else:
                    raise ValueError("Unable to build pixel_values: processor.image_processor is unavailable")

            attention_mask = (
                batch_encoding["attention_mask"].to(device)
                if use_attention_mask
                else None
            )
            outputs = self.model(
                input_ids=batch_encoding["input_ids"].to(device),
                attention_mask=attention_mask,
                output_hidden_states=output_hidden_states
                or hidden_state_skip_layer is not None,
                pixel_values=image_outputs,
            )
            if hidden_state_skip_layer is not None:
                last_hidden_state = outputs.hidden_states[
                    -(hidden_state_skip_layer + 1)
                ]
                # Real last hidden state already has layer norm applied. So here we only apply it
                # for intermediate layers.
                if hidden_state_skip_layer > 0 and self.apply_final_norm:
                    last_hidden_state = self.model.final_layer_norm(last_hidden_state)
            else:
                last_hidden_state = outputs[self.output_key]
            if self.use_template:
                if data_type == "video":
                    crop_start = self.prompt_template_video.get("crop_start", -1)
                    text_crop_start = (
                        crop_start
                        - 1
                        + self.prompt_template_video.get("image_emb_len", 576)
                    )
                    image_crop_start = self.prompt_template_video.get(
                        "image_emb_start", 5
                    )
                    image_crop_end = self.prompt_template_video.get(
                        "image_emb_end", 581
                    )
                    batch_indices, last_double_return_token_indices = torch.where(
                        batch_encoding["input_ids"]
                        == self.prompt_template_video.get("double_return_token_id", 271)
                    )
                    if last_double_return_token_indices.shape[0] == 3:
                        # in case the prompt is too long
                        last_double_return_token_indices = torch.cat(
                            (
                                last_double_return_token_indices,
                                torch.tensor([batch_encoding["input_ids"].shape[-1]]),
                            )
                        )
                        batch_indices = torch.cat((batch_indices, torch.tensor([0])))
                    last_double_return_token_indices = (
                        last_double_return_token_indices.reshape(
                            batch_encoding["input_ids"].shape[0], -1
                        )[:, -1]
                    )
                    batch_indices = batch_indices.reshape(
                        batch_encoding["input_ids"].shape[0], -1
                    )[:, -1]
                    assistant_crop_start = (
                        last_double_return_token_indices
                        - 1
                        + self.prompt_template_video.get("image_emb_len", 576)
                        - 4
                    )
                    assistant_crop_end = (
                        last_double_return_token_indices
                        - 1
                        + self.prompt_template_video.get("image_emb_len", 576)
                    )
                    attention_mask_assistant_crop_start = (
                        last_double_return_token_indices - 4
                    )
                    attention_mask_assistant_crop_end = last_double_return_token_indices
                else:
                    raise ValueError(f"Unsupported data type: {data_type}")

                text_last_hidden_state = []
                text_attention_mask = []
                image_last_hidden_state = []
                image_attention_mask = []
                for i in range(batch_encoding["input_ids"].shape[0]):
                    text_last_hidden_state.append(
                        torch.cat(
                            [
                                last_hidden_state[
                                    i, text_crop_start : assistant_crop_start[i].item()
                                ],
                                last_hidden_state[i, assistant_crop_end[i].item() :],
                            ]
                        )
                    )
                    text_attention_mask.append(
                        torch.cat(
                            [
                                attention_mask[
                                    i, text_crop_start : assistant_crop_start[i].item()
                                ],
                                attention_mask[
                                    i, assistant_crop_end[i].item() :
                                ],
                            ]
                        )
                        if use_attention_mask
                        else None
                    )
                    image_last_hidden_state.append(
                        last_hidden_state[i, image_crop_start:image_crop_end]
                    )
                    image_attention_mask.append(
                        torch.ones(image_last_hidden_state[-1].shape[0])
                        .to(last_hidden_state.device)
                        .to(attention_mask.dtype)
                        if use_attention_mask
                        else None
                    )

                text_last_hidden_state = torch.stack(text_last_hidden_state)
                text_attention_mask = torch.stack(text_attention_mask)
                image_last_hidden_state = torch.stack(image_last_hidden_state)
                image_attention_mask = torch.stack(image_attention_mask)

                if semantic_images is not None and 0 < self.image_embed_interleave < 6:
                    image_last_hidden_state = image_last_hidden_state[
                        :, ::self.image_embed_interleave, :
                    ]
                    image_attention_mask = image_attention_mask[
                        :, ::self.image_embed_interleave
                    ]

                assert (
                    text_last_hidden_state.shape[0] == text_attention_mask.shape[0]
                    and image_last_hidden_state.shape[0]
                    == image_attention_mask.shape[0]
                )

                last_hidden_state = torch.cat(
                    [image_last_hidden_state, text_last_hidden_state], dim=1
                )
                attention_mask = torch.cat(
                    [image_attention_mask, text_attention_mask], dim=1
                )

                # Guard against token/mask drift when processor/model templates differ.
                if use_attention_mask and attention_mask.shape[1] != last_hidden_state.shape[1]:
                    seq_hidden = int(last_hidden_state.shape[1])
                    seq_mask = int(attention_mask.shape[1])
                    if self.logger is not None:
                        self.logger.warning(
                            f"[hyvideo.text_encoder] align i2v attention_mask length: hidden={seq_hidden}, mask={seq_mask}"
                        )
                    if seq_mask > seq_hidden:
                        attention_mask = attention_mask[:, :seq_hidden]
                    else:
                        pad = torch.ones(
                            (attention_mask.shape[0], seq_hidden - seq_mask),
                            dtype=attention_mask.dtype,
                            device=attention_mask.device,
                        )
                        attention_mask = torch.cat([attention_mask, pad], dim=1)
            if output_hidden_states:
                return TextEncoderModelOutput(
                    last_hidden_state,
                    attention_mask,
                    hidden_states_list=outputs.hidden_states,
                )
            return TextEncoderModelOutput(last_hidden_state, attention_mask)

    def forward(
        self,
        text,
        use_attention_mask=None,
        output_hidden_states=False,
        do_sample=False,
        hidden_state_skip_layer=None,
        return_texts=False,
    ):
        batch_encoding = self.text2tokens(text)
        return self.encode(
            batch_encoding,
            use_attention_mask=use_attention_mask,
            output_hidden_states=output_hidden_states,
            do_sample=do_sample,
            hidden_state_skip_layer=hidden_state_skip_layer,
            return_texts=return_texts,
        )
