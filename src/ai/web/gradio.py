import logging
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import gradio as gr

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

        self.loop = cog.bot.loop
        self.gensettings = cog.model_provider_cfg.gensettings
        self.components = {}
        self.blocks = gr.Blocks(
            title=f"{cog.name} webui",
            analytics_enabled=False,
            theme="freddyaboulton/dracula_revamped",
        )

        # Last text prompts
        self.last_prompt = None
        self.last_response = None

        # Last imagen trigger/prompt/image
        self.imagen_last_trigger: str = ""
        self.imagen_last_tags: str = ""
        self.imagen_last_image: Optional[Path] = None
        pass

    @property
    def bot(self) -> DiscoSnake:
        return self.cog.bot

    def imagen_update(self, lm_trigger: str, lm_tags: str, image: Path):
        self.imagen_last_trigger = lm_trigger
        self.imagen_last_tags = lm_tags
        self.imagen_last_image = image
        pass

    def _create_components(self):
        with self.blocks:

            def set_parameter(data, evt: gr.EventData) -> None:
                logger.info(f"Setting parameter {evt.target.elem_id} to {data}")
                try:
                    input_name = evt.target.elem_id
                    input_value = data

                    setattr(self.cog.model_provider_cfg.gensettings, input_name, input_value)
                    logger.info(f"Set parameter {input_name} to {input_value}")
                except Exception:
                    logger.exception("Failed to set parameter")

            gr.Markdown(f"## {self.cog.name} webui")
            with gr.Row(variant="panel").style(equal_height=True):
                # Column 1
                with gr.Column():
                    with gr.Box():
                        with gr.Column():
                            gr.Markdown("### Main parameters")
                            seed = gr.Number(
                                value=self.gensettings.seed,
                                precision=0,
                                label="seed",
                                elem_id="seed",
                                info="Random seed for generation. Set to -1 for random seed.",
                            )
                            temperature = gr.Slider(
                                minimum=0.01,
                                maximum=1.99,
                                value=self.gensettings.temperature,
                                step=0.01,
                                label="temperature",
                                elem_id="temperature",
                                info="Higher value = more random results, lower value = more likely to stay in context.",
                            )
                            top_p = gr.Slider(
                                minimum=0.0,
                                maximum=1.0,
                                value=self.gensettings.top_p,
                                step=0.01,
                                label="top_p",
                                elem_id="top_p",
                                info="Select only from tokens with cumulative probability > `top_p`, given the prior text.",
                            )

                            top_k = gr.Slider(
                                value=self.gensettings.top_k,
                                minimum=0,
                                maximum=200,
                                step=1,
                                label="top_k",
                                elem_id="top_k",
                                info="Select only from the most probable `top_k` tokens, given the prior text.",
                            )

                            typical_p = gr.Slider(
                                value=self.gensettings.typical_p,
                                minimum=0.0,
                                maximum=1.0,
                                step=0.01,
                                label="typical_p",
                                elem_id="typical_p",
                                info="Select only from tokens with probability > `typical_p`, given the prior text.",
                            )
                            repetition_penalty = gr.Slider(
                                value=self.gensettings.repetition_penalty,
                                minimum=1.0,
                                maximum=1.5,
                                step=0.01,
                                label="repetition_penalty",
                                elem_id="repetition_penalty",
                                info="Penalty factor for tokens that have already appeared in the output. Higher values = less repetition.",
                            )
                            encoder_repetition_penalty = gr.Slider(
                                value=self.gensettings.encoder_repetition_penalty,
                                minimum=0.8,
                                maximum=1.5,
                                step=0.01,
                                label="encoder_repetition_penalty",
                                elem_id="encoder_repetition_penalty",
                                info="Exponential penalty factor for tokens not found in the prompt text.",
                            )
                            no_repeat_ngram_size = gr.Slider(
                                value=self.gensettings.no_repeat_ngram_size,
                                minimum=0,
                                maximum=20,
                                step=1,
                                label="no_repeat_ngram_size",
                                elem_id="no_repeat_ngram_size",
                                info="Sequences of this length or shorter will not be repeated in the output *at all*.",
                            )

                            epsilon_cutoff = gr.Slider(
                                value=self.gensettings.epsilon_cutoff,
                                minimum=0,
                                maximum=9,
                                step=0.01,
                                label="epsilon_cutoff",
                                elem_id="epsilon_cutoff",
                                info="In units of 1e-4",
                            )
                            eta_cutoff = gr.Slider(
                                value=self.gensettings.eta_cutoff,
                                minimum=0,
                                maximum=20,
                                step=0.01,
                                label="eta_cutoff",
                                elem_id="eta_cutoff",
                                info="In units of 1e-4",
                            )

                            seed.change(fn=set_parameter, inputs=seed, outputs=None)
                            temperature.change(fn=set_parameter, inputs=temperature, outputs=None)
                            top_p.change(fn=set_parameter, inputs=top_p, outputs=None)
                            top_k.change(fn=set_parameter, inputs=top_k, outputs=None)
                            typical_p.change(fn=set_parameter, inputs=typical_p, outputs=None)
                            repetition_penalty.change(
                                fn=set_parameter, inputs=repetition_penalty, outputs=None
                            )
                            encoder_repetition_penalty.change(
                                fn=set_parameter, inputs=encoder_repetition_penalty, outputs=None
                            )
                            no_repeat_ngram_size.change(
                                fn=set_parameter, inputs=no_repeat_ngram_size, outputs=None
                            )
                            epsilon_cutoff.change(fn=set_parameter, inputs=epsilon_cutoff, outputs=None)
                            eta_cutoff.change(fn=set_parameter, inputs=eta_cutoff, outputs=None)

                # Column 2
                with gr.Column():
                    with gr.Box():
                        with gr.Column():
                            gr.Markdown("### Extra parameters")

                            do_sample = gr.Checkbox(
                                value=self.gensettings.do_sample,
                                label="do_sample",
                                elem_id="do_sample",
                                info="Enable random sampling (disable for contrastive search)",
                            )
                            penalty_alpha = gr.Slider(
                                value=self.gensettings.penalty_alpha,
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
                                value=self.gensettings.num_beams,
                                minimum=1,
                                maximum=20,
                                step=1,
                                label="num_beams",
                                elem_id="num_beams",
                            )
                            length_penalty = gr.Slider(
                                value=self.gensettings.length_penalty,
                                minimum=-5.0,
                                maximum=5.0,
                                step=0.01,
                                label="length_penalty",
                                elem_id="length_penalty",
                            )
                            early_stopping = gr.Checkbox(
                                value=self.gensettings.early_stopping,
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
                                value=self.gensettings.min_length,
                                minimum=0,
                                maximum=2000,
                                step=1,
                                label="min_length",
                                elem_id="min_length",
                                info="Minimum generation length (tokens)",
                            )
                            max_new_tokens = gr.Slider(
                                value=self.gensettings.max_new_tokens,
                                minimum=0,
                                maximum=2000,
                                step=1,
                                label="max_new_tokens",
                                elem_id="max_new_tokens",
                                info="Maximum generation length (tokens)",
                            )
                            ban_eos_token = gr.Checkbox(
                                value=self.gensettings.ban_eos_token,
                                label="ban_eos_token",
                                elem_id="ban_eos_token",
                                info="Ban the model from ending generation early.",
                            )
                            add_bos_token = gr.Checkbox(
                                value=self.gensettings.add_bos_token,
                                label="add_bos_token",
                                elem_id="add_bos_token",
                                info="With some models, disabling this can give more creative results.",
                            )
                            skip_special_tokens = gr.Checkbox(
                                value=self.gensettings.skip_special_tokens,
                                label="skip_special_tokens",
                                elem_id="skip_special_tokens",
                                info="Skip special tokens (such as <pad> or <unk>) when decoding output.",
                            )
                            truncation_length = gr.Slider(
                                value=self.gensettings.truncation_length,
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
                            truncation_length.change(set_parameter, inputs=truncation_length, outputs=None)

            with gr.Row(variant="panel").style(equal_height=True):
                with gr.Column():
                    with gr.Box():
                        with gr.Column():
                            gr.Markdown("### Last messages")
                            last_prompt = gr.Textbox(
                                value=self.last_prompt,
                                lines=5,
                                interactive=False,
                                label="Last prompt",
                                every=2.0,
                            )
                            last_response = gr.Textbox(
                                value=self.last_response,
                                lines=5,
                                interactive=False,
                                label="Last response",
                                every=2.0,
                            )

                with gr.Column():

                    def get_last_trigger():
                        return self.imagen_last_trigger

                    def get_last_prompt():
                        return self.imagen_last_tags

                    def get_last_image():
                        return self.imagen_last_image

                    with gr.Box():
                        with gr.Column():
                            gr.Markdown("### Imagen")
                            imagen_last_trigger = gr.Textbox(
                                value=get_last_trigger,
                                interactive=False,
                                label="Last input prompt",
                                every=2.0,
                            )
                            imagen_last_image = gr.Image(
                                value=get_last_image,
                                interactive=False,
                                label="Last image",
                                every=2.0,
                            )
                            imagen_last_tags = gr.Textbox(
                                value=get_last_prompt,
                                interactive=False,
                                label="Last output prompt",
                                every=2.0,
                            )

                    pass

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
