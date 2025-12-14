import asyncio
import logging
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING

import gradio as gr
from gradio.components import Component
from gradio.themes import Base, GoogleFont
from gradio.themes.utils import colors

from ai.settings import GradioConfig
from disco_snake.bot import DiscoSnake

if TYPE_CHECKING:
    from ai import Ai

logger = logging.getLogger(__name__)
gr_logger = logging.getLogger("gradio")
gr_logger.setLevel(logging.INFO)


class GradioUi:
    def __init__(self, cog: "Ai", config: GradioConfig) -> None:
        logger.info("Initializing Gradio UI")
        self.cog = cog
        self.config = config

        if self.config.theme is None or self.config.theme == "default":
            self.theme = Base(
                primary_hue=colors.violet,
                secondary_hue=colors.indigo,
                neutral_hue=colors.slate,
                font=[GoogleFont("Fira Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
                font_mono=[GoogleFont("Fira Code"), "ui-monospace", "Consolas", "monospace"],
            ).set(
                slider_color_dark="*primary_500",
            )
        else:
            self.theme = self.config.theme

        self.lm_gensettings = cog.provider_config.gensettings
        self.dynamic_elements = {}

        try:
            self.css = Path(__file__).with_suffix(".css").read_text()
        except Exception:
            logger.exception("Failed to load CSS file")
            self.css = ""

        self.blocks = gr.Blocks(
            title=f"{cog.name} webui",
            analytics_enabled=False,
            theme=self.theme,
            css=self.css,
        )

        self.dynamic_elements: list[Component] = []

        # Last text prompts
        self.lm_last_prompt = ""
        self.lm_last_message = ""
        self.lm_last_response = ""

        # Last imagen trigger/prompt/image
        self.img_last_request: str = ""
        self.img_last_tags: str = ""
        self.img_last_image: Path | None = None
        pass

    @property
    def bot(self) -> DiscoSnake:
        return self.cog.bot

    def lm_update(self, prompt, message, response) -> None:
        self.lm_last_prompt = prompt
        self.lm_last_message = message
        self.lm_last_response = response

    def imagen_update(self, lm_request: str, lm_tags: str, image_path: Path):
        self.img_last_request = lm_request
        self.img_last_tags = lm_tags
        self.img_last_image = image_path
        pass

    def _create_components(self):
        with self.blocks:

            def evt_reload():
                ret = []
                for elem in self.dynamic_elements:
                    elem_val = None
                    try:
                        elem_val = getattr(elem, "value", None)
                        if not hasattr(self.cog.provider_config.gensettings, elem.elem_id):
                            logger.warning(f"Element {elem.elem_id} not found in gensettings")
                        else:
                            current_val = getattr(self.cog.provider_config.gensettings, elem.elem_id)
                            if elem_val != current_val:
                                logger.debug(f"Updating {elem.elem_id} from {elem_val} to {current_val}")
                                elem_val = current_val
                    except Exception as e:
                        logger.exception(f"Failed to get value for {elem.elem_id}", e)
                    finally:
                        ret.append(elem_val)
                return ret

            def evt_set_param(element, evt: gr.EventData):
                try:
                    input_name = evt.target.elem_id  # type: ignore
                    input_value = element
                    logger.debug(f"Setting '{input_name}' to {input_value}")
                    setattr(self.cog.provider_config.gensettings, input_name, input_value)
                    return getattr(self.cog.provider_config.gensettings, input_name)
                except Exception:
                    logger.exception("Failed to set parameter")

            def get_self_attr(element, *args, **kwargs):
                target_id = getattr(element, "elem_id", element)
                try:
                    value = getattr(self, target_id, None)
                except Exception as e:
                    logger.exception(f"Failed to get self attr {target_id}")
                    raise e
                return value

            def get_ai_attr_json(element, *args, **kwargs):
                target_id = getattr(element, "elem_id", element)
                try:
                    value = getattr(self.cog, target_id, None)
                    if value is None:
                        value = {"error": f"Attribute {self.cog}.{target_id} is None"}
                    if hasattr(value, "to_json"):
                        value = value.to_json()
                    if hasattr(value, "json"):
                        value = value.json()
                except Exception as e:
                    logger.exception(f"Failed to get AI cog attr {target_id}")
                    raise e
                return value

            # title bar
            with gr.Row(equal_height=True):
                with gr.Column(scale=12, elem_id="header_col"):
                    self.header_title = gr.Markdown(
                        f"## {self.cog.name} webui",
                        elem_id="header_title",
                    )
                with gr.Column(scale=1, min_width=90, elem_id="button_col"):
                    with gr.Row(elem_id="button_row"):
                        self.reload_btn = gr.Button(
                            elem_id="refresh_btn",
                            value="🔄",
                            variant="primary",
                        )

            # Language model settings
            with gr.Tab(label="LM Settings"):
                with gr.Row(variant="panel", equal_height=True):
                    # Column 1
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Main parameters")
                                seed = gr.Number(
                                    value=self.lm_gensettings.seed,
                                    precision=0,
                                    label="Seed",
                                    elem_id="seed",
                                    info="Random seed for generation. Set to -1 for random seed.",
                                )
                                temperature = gr.Slider(
                                    value=self.lm_gensettings.temperature,
                                    minimum=0.01,
                                    maximum=1.99,
                                    step=0.01,
                                    label="Temperature",
                                    elem_id="temperature",
                                    info="Higher values = more random, lower values = more conservative.",
                                )
                                top_p = gr.Slider(
                                    minimum=0.0,
                                    maximum=1.0,
                                    value=self.lm_gensettings.top_p,
                                    step=0.01,
                                    label="Top P",
                                    elem_id="top_p",
                                    info="Select only from tokens with cumulative probability > `top_p`, given the prior text.",
                                )
                                top_k = gr.Slider(
                                    value=self.lm_gensettings.top_k,
                                    minimum=0,
                                    maximum=200,
                                    step=1,
                                    label="Top K",
                                    elem_id="top_k",
                                    info="Select only from the most probable N tokens, given the prior text.",
                                )
                                typical_p = gr.Slider(
                                    value=self.lm_gensettings.typical_p,
                                    minimum=0.0,
                                    maximum=1.0,
                                    step=0.01,
                                    label="Typical P",
                                    elem_id="typical_p",
                                    info="Select only from tokens with probability > N, given the prior text.",
                                )
                                repetition_penalty = gr.Slider(
                                    value=self.lm_gensettings.repetition_penalty,
                                    minimum=1.0,
                                    maximum=1.5,
                                    step=0.01,
                                    label="Repetition Penalty",
                                    elem_id="repetition_penalty",
                                    info="Penalty factor for tokens that have already appeared in the output. Higher values = less repetition.",
                                )
                                encoder_repetition_penalty = gr.Slider(
                                    value=self.lm_gensettings.encoder_repetition_penalty,
                                    minimum=0.8,
                                    maximum=1.5,
                                    step=0.01,
                                    label="Encoder Rep. Penalty",
                                    elem_id="encoder_repetition_penalty",
                                    info="Exponential penalty factor for tokens not found in the prompt text.",
                                )
                                no_repeat_ngram_size = gr.Slider(
                                    value=self.lm_gensettings.no_repeat_ngram_size,
                                    minimum=0,
                                    maximum=20,
                                    step=1,
                                    label="No-Repeat Ngram Size",
                                    elem_id="no_repeat_ngram_size",
                                    info="Sequences of this length or shorter will not be repeated in the output *at all*.",
                                )
                                epsilon_cutoff = gr.Slider(
                                    value=self.lm_gensettings.epsilon_cutoff,
                                    minimum=0,
                                    maximum=9,
                                    step=0.01,
                                    label="Epsilon Cutoff",
                                    elem_id="epsilon_cutoff",
                                    info="In units of 1e-4",
                                )
                                eta_cutoff = gr.Slider(
                                    value=self.lm_gensettings.eta_cutoff,
                                    minimum=0,
                                    maximum=20,
                                    step=0.01,
                                    label="Eta Cutoff",
                                    elem_id="eta_cutoff",
                                    info="In units of 1e-4",
                                )

                                seed.input(fn=evt_set_param, inputs=seed, outputs=None, api_name="seed")
                                temperature.input(
                                    fn=evt_set_param, inputs=temperature, api_name="temperature"
                                )
                                top_p.input(fn=evt_set_param, inputs=top_p, api_name="top_p")
                                top_k.input(fn=evt_set_param, inputs=top_k, api_name="top_k")
                                typical_p.input(fn=evt_set_param, inputs=typical_p, api_name="typical_p")
                                repetition_penalty.input(
                                    fn=evt_set_param, inputs=repetition_penalty, api_name="repetition_penalty"
                                )
                                encoder_repetition_penalty.input(
                                    fn=evt_set_param,
                                    inputs=encoder_repetition_penalty,
                                    api_name="encoder_repetition_penalty",
                                )
                                no_repeat_ngram_size.input(
                                    fn=evt_set_param,
                                    inputs=no_repeat_ngram_size,
                                    api_name="no_repeat_ngram_size",
                                )
                                epsilon_cutoff.input(
                                    fn=evt_set_param, inputs=epsilon_cutoff, api_name="epsilon_cutoff"
                                )
                                eta_cutoff.input(fn=evt_set_param, inputs=eta_cutoff, api_name="eta_cutoff")

                                self.dynamic_elements.extend(
                                    [
                                        seed,
                                        temperature,
                                        top_p,
                                        top_k,
                                        typical_p,
                                        repetition_penalty,
                                        encoder_repetition_penalty,
                                        no_repeat_ngram_size,
                                        epsilon_cutoff,
                                        eta_cutoff,
                                    ]
                                )

                    # Column 2
                    with gr.Column():
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Extra parameters")

                                do_sample = gr.Checkbox(
                                    value=self.lm_gensettings.do_sample,
                                    label="do_sample",
                                    elem_id="do_sample",
                                    info="Enable random sampling (disable for contrastive search)",
                                )
                                penalty_alpha = gr.Slider(
                                    value=self.lm_gensettings.penalty_alpha,
                                    minimum=0,
                                    maximum=5,
                                    step=0.01,
                                    label="penalty_alpha",
                                    elem_id="penalty_alpha",
                                    info="Disable do_sample and set a low top_k to use contrastive search mode.",
                                )

                                gr.Markdown("### Beam search (high memory usage)")
                                num_beams = gr.Slider(
                                    value=self.lm_gensettings.num_beams,
                                    minimum=1,
                                    maximum=20,
                                    step=1,
                                    label="num_beams",
                                    elem_id="num_beams",
                                )
                                length_penalty = gr.Slider(
                                    value=self.lm_gensettings.length_penalty,
                                    minimum=-5.0,
                                    maximum=5.0,
                                    step=0.01,
                                    label="length_penalty",
                                    elem_id="length_penalty",
                                )
                                early_stopping = gr.Checkbox(
                                    value=self.lm_gensettings.early_stopping,
                                    label="early_stopping",
                                    elem_id="early_stopping",
                                    info="Stop generation early if all beams have finished.",
                                )
                                do_sample.input(fn=evt_set_param, inputs=do_sample, api_name="do_sample")
                                penalty_alpha.input(
                                    fn=evt_set_param, inputs=penalty_alpha, api_name="penalty_alpha"
                                )
                                num_beams.input(fn=evt_set_param, inputs=num_beams, api_name="num_beams")
                                length_penalty.input(
                                    fn=evt_set_param, inputs=length_penalty, api_name="length_penalty"
                                )
                                early_stopping.input(
                                    fn=evt_set_param, inputs=early_stopping, api_name="early_stopping"
                                )

                                self.dynamic_elements.extend(
                                    [do_sample, penalty_alpha, num_beams, length_penalty, early_stopping]
                                )

                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Input/output parameters")
                                min_tokens = gr.Slider(
                                    value=self.lm_gensettings.min_tokens,
                                    minimum=0,
                                    maximum=2000,
                                    step=1,
                                    label="min_tokens",
                                    elem_id="min_tokens",
                                    info="Minimum generation length (tokens)",
                                )
                                max_tokens = gr.Slider(
                                    value=self.lm_gensettings.max_tokens,
                                    minimum=0,
                                    maximum=2000,
                                    step=1,
                                    label="max_tokens",
                                    elem_id="max_tokens",
                                    info="Maximum generation length (tokens)",
                                )
                                ban_eos_token = gr.Checkbox(
                                    value=self.lm_gensettings.ban_eos_token,
                                    label="ban_eos_token",
                                    elem_id="ban_eos_token",
                                    info="Ban the model from ending generation early.",
                                )
                                add_bos_token = gr.Checkbox(
                                    value=self.lm_gensettings.add_bos_token,
                                    label="add_bos_token",
                                    elem_id="add_bos_token",
                                    info="With some models, disabling this can give more creative results.",
                                )
                                skip_special_tokens = gr.Checkbox(
                                    value=self.lm_gensettings.skip_special_tokens,
                                    label="skip_special_tokens",
                                    elem_id="skip_special_tokens",
                                    info="Skip special tokens (such as <pad> or <unk>) when decoding output.",
                                )
                                truncation_length = gr.Slider(
                                    value=self.lm_gensettings.truncation_length,
                                    minimum=512,
                                    maximum=2048,
                                    step=1,
                                    label="truncation_length",
                                    elem_id="truncation_length",
                                    info="Maximum length of the input to the model. For most models, this is 2048.",
                                )
                                min_tokens.input(fn=evt_set_param, inputs=min_tokens, api_name="min_tokens")
                                max_tokens.input(fn=evt_set_param, inputs=max_tokens, api_name="max_tokens")
                                ban_eos_token.input(
                                    fn=evt_set_param, inputs=ban_eos_token, api_name="ban_eos_token"
                                )
                                add_bos_token.input(
                                    fn=evt_set_param, inputs=add_bos_token, api_name="add_bos_token"
                                )
                                skip_special_tokens.input(
                                    evt_set_param, inputs=skip_special_tokens, api_name="skip_special_tokens"
                                )
                                truncation_length.input(
                                    evt_set_param, inputs=truncation_length, api_name="truncation_length"
                                )

                                self.dynamic_elements.extend(
                                    [
                                        min_tokens,
                                        max_tokens,
                                        ban_eos_token,
                                        add_bos_token,
                                        skip_special_tokens,
                                        truncation_length,
                                    ]
                                )

            # Prompt info and imagen info
            with gr.Tab(label="Status"):
                with gr.Row(variant="panel", equal_height=True):
                    with gr.Column(elem_id="lm_messages"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### LM Messages")
                                self.gr_last_message = gr.Textbox(
                                    value=partial(get_self_attr, "lm_last_message"),
                                    max_lines=5,
                                    interactive=True,
                                    label="Message",
                                    info="Last message to trigger model generation.",
                                    elem_id="lm_last_message",
                                    every=4.0,
                                )
                                self.gr_last_response = gr.Textbox(
                                    value=partial(get_self_attr, "lm_last_response"),
                                    max_lines=5,
                                    interactive=True,
                                    label="Response",
                                    info="Last response from the model.",
                                    elem_id="lm_last_response",
                                    every=4.0,
                                )
                                self.gr_last_prompt = gr.Textbox(
                                    value=partial(get_self_attr, "lm_last_prompt"),
                                    max_lines=30,
                                    interactive=True,
                                    label="Prompt",
                                    info="Last prompt sent to the model.",
                                    elem_id="lm_last_prompt",
                                    every=4.0,
                                )

                    with gr.Column(elem_id="imagen_status"):
                        with gr.Row():
                            with gr.Column():
                                gr.Markdown("### Imagen")
                                with gr.Group():
                                    self.imagen_last_request = gr.Textbox(
                                        value=partial(get_self_attr, "img_last_request"),
                                        max_lines=10,
                                        interactive=True,
                                        label="Last input request",
                                        elem_id="img_last_request",
                                        every=4.0,
                                    )
                                    self.imagen_last_image = gr.Image(
                                        value=partial(get_self_attr, "img_last_image"),
                                        interactive=False,
                                        label="Last image",
                                        elem_id="img_last_image",
                                        every=4.0,
                                        height=640,
                                    )
                                    self.imagen_last_tags = gr.Textbox(
                                        value=partial(get_self_attr, "img_last_tags"),
                                        max_lines=10,
                                        interactive=True,
                                        label="Last output prompt",
                                        elem_id="img_last_tags",
                                        every=4.0,
                                    )

                with gr.Row(variant="panel", equal_height=True):
                    with gr.Column(elem_id="lm_debug"):
                        self._ai_trigger_cache = gr.JSON(
                            value=partial(get_ai_attr_json, "_last_debug_log"),
                            label="Last debug log",
                            elem_id="ai_last_debug_log",
                            every=4.0,
                        )
                    with gr.Column(elem_id="lm_extras"):
                        self._ai_trigger_cache = gr.JSON(
                            value=partial(get_ai_attr_json, "trigger_cache"),
                            label="Message trigger cache",
                            elem_id="ai_trigger_cache",
                            every=4.0,
                        )

            # set up the reload func
            self.reload_btn.click(
                evt_reload,
                inputs=[],
                outputs=self.dynamic_elements,
            )

    async def launch(self, **kwargs) -> None:
        if self.config.enabled:
            try:
                gradio_loop = asyncio.new_event_loop()
                asyncio.set_event_loop(gradio_loop)
                logger.info("Launching Gradio UI")
                logger.info("Creating components...")
                self._create_components()
                logger.info("Launching UI...")
                self.blocks.launch(
                    server_name=self.config.bind_host,
                    server_port=self.config.bind_port,
                    width=self.config.width,
                    prevent_thread_lock=True,
                    show_error=True,
                    root_path=self.config.root_path,
                    **kwargs,
                )
                logger.info("UI launched!")
            except Exception:
                logger.exception("Failed to launch UI")
        else:
            logger.info("UI disabled, skipping launch")

    def shutdown(self) -> None:
        logger.info("Shutting down UI...")
        self.blocks.close()
        logger.info("UI shutdown!")
