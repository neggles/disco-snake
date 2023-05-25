import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import gradio as gr
from gradio.themes.utils import colors

from ai.settings import GradioConfig
from disco_snake.bot import DiscoSnake

if TYPE_CHECKING:
    from ai import Ai

logger = logging.getLogger(__name__)
gr_logger = logging.getLogger("gradio")
gr_logger.setLevel(logging.DEBUG)


class GradioUi:
    def __init__(self, cog: "Ai", config: GradioConfig) -> None:
        logger.info("Initializing Gradio UI")
        self.cog = cog
        self.config = config
        self.theme = gr.themes.Base(
            primary_hue=colors.violet,
            secondary_hue=colors.indigo,
            neutral_hue=colors.slate,
            font=[gr.themes.GoogleFont("Fira Sans"), "ui-sans-serif", "system-ui", "sans-serif"],
            font_mono=[gr.themes.GoogleFont("Fira Code"), "ui-monospace", "Consolas", "monospace"],
        ).set(
            slider_color_dark="*primary_500",
        )

        self.lm_gensettings = cog.model_provider_cfg.gensettings
        self.components = {}

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

        # Last text prompts
        self.lm_last_prompt = ""
        self.lm_last_message = ""
        self.lm_last_response = ""

        # Last imagen trigger/prompt/image
        self.im_last_request: str = ""
        self.im_last_tags: str = ""
        self.im_last_image: Optional[Path] = None
        pass

    @property
    def bot(self) -> DiscoSnake:
        return self.cog.bot

    def lm_update(self, prompt, message, response) -> None:
        self.lm_last_prompt = prompt
        self.lm_last_message = message
        self.lm_last_response = response

    def imagen_update(self, lm_request: str, lm_tags: str, image: Path):
        self.im_last_request = lm_request
        self.im_last_tags = lm_tags
        self.im_last_image = image
        pass

    def _get_param(self, name: str) -> Any:
        return getattr(self.cog.model_provider_cfg.gensettings, name, None)

    def set_lm_setting(self, name: str, value: Any) -> None:
        if hasattr(self.lm_gensettings, name):
            setattr(self.lm_gensettings, name, value)
            logger.info(f"Set parameter {name} to {value}")

    def _create_components(self):
        with self.blocks:

            def get_parameter(element, evt: gr.EventData):
                logger.debug(f"Getting parameter {evt.target.elem_id}")
                try:
                    target = evt.target.elem_id
                    return getattr(self.cog.model_provider_cfg.gensettings, target, None)
                except Exception:
                    logger.exception("Failed to set parameter")

            def set_parameter(element, evt: gr.EventData):
                logger.info(f"Setting parameter {evt.target.elem_id} to {element}")
                try:
                    input_name = evt.target.elem_id
                    input_value = element
                    setattr(self.cog.model_provider_cfg.gensettings, input_name, input_value)
                    logger.info(f"Set parameter {input_name} to {input_value}")
                except Exception:
                    logger.exception("Failed to set parameter")

            with gr.Row():
                gr.Markdown(f"## {self.cog.name} webui")

            # Language model settings
            with gr.Tab(label="LM Settings"):
                with gr.Row(variant="panel").style(equal_height=True):
                    # Column 1
                    with gr.Column():
                        with gr.Box():
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

                                seed.change(fn=set_parameter, inputs=seed, outputs=None, api_name="seed")
                                seed.set_event_trigger(
                                    "load", get_parameter, inputs=seed, outputs=seed, every=2.0
                                )
                                temperature.change(
                                    fn=set_parameter, inputs=temperature, outputs=None, api_name="temperature"
                                )
                                temperature.set_event_trigger(
                                    "load", get_parameter, inputs=temperature, outputs=temperature, every=2.0
                                )

                                top_p.change(fn=set_parameter, inputs=top_p, outputs=None, api_name="top_p")
                                top_k.change(fn=set_parameter, inputs=top_k, outputs=None, api_name="top_k")
                                typical_p.change(
                                    fn=set_parameter, inputs=typical_p, outputs=None, api_name="typical_p"
                                )
                                repetition_penalty.change(
                                    fn=set_parameter,
                                    inputs=repetition_penalty,
                                    outputs=None,
                                    api_name="repetition_penalty",
                                )
                                encoder_repetition_penalty.change(
                                    fn=set_parameter,
                                    inputs=encoder_repetition_penalty,
                                    outputs=None,
                                    api_name="encoder_repetition_penalty",
                                )
                                no_repeat_ngram_size.change(
                                    fn=set_parameter,
                                    inputs=no_repeat_ngram_size,
                                    outputs=None,
                                    api_name="no_repeat_ngram_size",
                                )
                                epsilon_cutoff.change(
                                    fn=set_parameter,
                                    inputs=epsilon_cutoff,
                                    outputs=None,
                                    api_name="epsilon_cutoff",
                                )
                                eta_cutoff.change(
                                    fn=set_parameter, inputs=eta_cutoff, outputs=None, api_name="eta_cutoff"
                                )

                    # Column 2
                    with gr.Column():
                        with gr.Box():
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

                                do_sample.change(fn=set_parameter, inputs=do_sample, outputs=None)
                                penalty_alpha.change(fn=set_parameter, inputs=penalty_alpha, outputs=None)

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
                                num_beams.change(fn=set_parameter, inputs=num_beams, outputs=None)
                                length_penalty.change(fn=set_parameter, inputs=length_penalty, outputs=None)
                                early_stopping.change(fn=set_parameter, inputs=early_stopping, outputs=None)

                        with gr.Box():
                            with gr.Column():
                                gr.Markdown("### Input/output parameters")
                                min_length = gr.Slider(
                                    value=self.lm_gensettings.min_length,
                                    minimum=0,
                                    maximum=2000,
                                    step=1,
                                    label="min_length",
                                    elem_id="min_length",
                                    info="Minimum generation length (tokens)",
                                )
                                max_new_tokens = gr.Slider(
                                    value=self.lm_gensettings.max_new_tokens,
                                    minimum=0,
                                    maximum=2000,
                                    step=1,
                                    label="max_new_tokens",
                                    elem_id="max_new_tokens",
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
                                min_length.change(fn=set_parameter, inputs=min_length, outputs=None)
                                max_new_tokens.change(fn=set_parameter, inputs=max_new_tokens, outputs=None)
                                ban_eos_token.change(fn=set_parameter, inputs=ban_eos_token, outputs=None)
                                add_bos_token.change(fn=set_parameter, inputs=add_bos_token, outputs=None)
                                skip_special_tokens.change(
                                    set_parameter, inputs=skip_special_tokens, outputs=None
                                )
                                truncation_length.change(
                                    set_parameter, inputs=truncation_length, outputs=None
                                )

                # Prompt info and imagen info
            with gr.Tab(label="Status"):
                with gr.Row(variant="panel").style(equal_height=True):
                    with gr.Column():

                        def lm_last_prompt():
                            return self.lm_last_prompt

                        def lm_last_message():
                            return self.lm_last_message

                        def lm_last_response():
                            return self.lm_last_response

                        with gr.Box():
                            with gr.Column():
                                gr.Markdown("### LM Messages")
                                self.gr_last_prompt = gr.Textbox(
                                    value=lm_last_prompt,
                                    every=2.0,
                                    lines=15,
                                    interactive=False,
                                    label="Current Prompt",
                                ).style(show_copy_button=True)
                                self.gr_last_message = gr.Textbox(
                                    value=lm_last_message,
                                    every=2.0,
                                    lines=2,
                                    interactive=False,
                                    label="Last Message",
                                ).style(show_copy_button=True)
                                self.gr_last_response = gr.Textbox(
                                    value=lm_last_response,
                                    every=2.0,
                                    lines=5,
                                    interactive=False,
                                    label="Last Response",
                                ).style(show_copy_button=True)

                    with gr.Column():

                        def im_last_request():
                            return self.im_last_request

                        def im_last_image():
                            return self.im_last_image

                        def im_last_tags():
                            return self.im_last_tags

                        with gr.Box():
                            with gr.Column():
                                gr.Markdown("### Imagen")
                                self.imagen_last_request = gr.Textbox(
                                    value=im_last_request,
                                    every=2.0,
                                    lines=1,
                                    interactive=False,
                                    label="Last input request",
                                    elem_id="imagen_last_request",
                                ).style(show_copy_button=True)
                                self.imagen_last_image = gr.Image(
                                    value=im_last_image,
                                    every=2.0,
                                    interactive=False,
                                    label="Last image",
                                    elem_id="imagen_last_image",
                                ).style(height=600)
                                self.imagen_last_tags = gr.Textbox(
                                    value=im_last_tags,
                                    every=2.0,
                                    interactive=False,
                                    label="Last output prompt",
                                    elem_id="imagen_last_tags",
                                ).style(show_copy_button=True)

    async def launch(self, **kwargs):
        if self.config.enabled:
            try:
                logger.info("Launching Gradio UI")
                logger.info("Creating components...")
                self._create_components()
                logger.info("Launching UI...")
                self.blocks.launch(
                    server_name=self.config.bind_host,
                    server_port=self.config.bind_port,
                    enable_queue=self.config.enable_queue,
                    width=self.config.width,
                    prevent_thread_lock=True,
                )
                logger.info("UI launched!")
            except Exception:
                logger.exception("Failed to launch UI")
        else:
            logger.info("UI disabled, skipping launch")
