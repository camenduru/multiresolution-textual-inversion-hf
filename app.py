#!/usr/bin/env python

from __future__ import annotations

import os

import gradio as gr
import torch

from model import Model

DESCRIPTION = '# [Multiresolution Textual Inversion](https://github.com/giannisdaras/multires_textual_inversion)'

DETAILS = '''
- To run the Semi Resolution-Dependent sampler, use the format: `<jane(number)>`.
- To run the Fully Resolution-Dependent sampler, use the format: `<jane[number]>`.
- To run the Fixed Resolution sampler, use the format: `<jane|number|>`.

For this demo, only `<jane>`, `<gta5-artwork>` and `<cat-toy>` are available.
Also, `number` should be an integer in [0, 9].
'''

CACHE_EXAMPLES = torch.cuda.is_available() and os.getenv(
    'CACHE_EXAMPLES') == '1'

model = Model()

with gr.Blocks(css='style.css') as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Group():
            with gr.Row():
                prompt = gr.Textbox(label='Prompt')
            with gr.Row():
                num_images = gr.Slider(
                    label='Number of images',
                    minimum=1,
                    maximum=9,
                    step=1,
                    value=1,
                )
            with gr.Row():
                num_steps = gr.Slider(label='Number of inference steps',
                                      minimum=1,
                                      maximum=50,
                                      step=1,
                                      value=10)
            with gr.Row():
                seed = gr.Slider(label='Seed',
                                 minimum=0,
                                 maximum=100000,
                                 step=1,
                                 value=100)
            with gr.Row():
                run_button = gr.Button('Run')

        with gr.Column():
            result = gr.Gallery(label='Result', object_fit='scale-down')

    with gr.Row():
        with gr.Group():
            fn = lambda x: model.run(x, 2, 10, 100)
            with gr.Row():
                gr.Examples(
                    label='Examples 1',
                    examples=[
                        ['an image of <gta5-artwork(0)>'],
                        ['an image of <jane(0)>'],
                        ['an image of <jane(3)>'],
                        ['an image of <cat-toy(0)>'],
                    ],
                    inputs=prompt,
                    outputs=result,
                    fn=fn,
                    cache_examples=CACHE_EXAMPLES,
                )
            with gr.Row():
                gr.Examples(
                    label='Examples 2',
                    examples=[
                        [
                            'an image of a cat in the style of <gta5-artwork(0)>'
                        ],
                        ['a painting of a dog in the style of <jane(0)>'],
                        ['a painting of a dog in the style of <jane(5)>'],
                        [
                            'a painting of a <cat-toy(0)> in the style of <jane(3)>'
                        ],
                    ],
                    inputs=prompt,
                    outputs=result,
                    fn=fn,
                    cache_examples=CACHE_EXAMPLES,
                )
            with gr.Row():
                gr.Examples(
                    label='Examples 3',
                    examples=[
                        ['an image of <jane[0]>'],
                        ['an image of <jane|0|>'],
                        ['an image of <jane|3|>'],
                    ],
                    inputs=prompt,
                    outputs=result,
                    fn=fn,
                    cache_examples=CACHE_EXAMPLES,
                )

        inputs = [
            prompt,
            num_images,
            num_steps,
            seed,
        ]
        prompt.submit(
            fn=model.run,
            inputs=inputs,
            outputs=result,
            api_name=False,
        )
        run_button.click(
            fn=model.run,
            inputs=inputs,
            outputs=result,
            api_name='run',
        )

    with gr.Accordion('About available prompts', open=False):
        gr.Markdown(DETAILS)

demo.queue(max_size=10).launch()
