#!/usr/bin/env python

from __future__ import annotations

import os

import gradio as gr

from model import Model

TITLE = '# Multiresolution Textual Inversion'
DESCRIPTION = 'An unofficial demo for [https://github.com/giannisdaras/multires_textual_inversion](https://github.com/giannisdaras/multires_textual_inversion).'

DETAILS = '''
- To run the Semi Resolution-Dependent sampler, use the format: `<jane(number)>`.
- To run the Fully Resolution-Dependent sampler, use the format: `<jane[number]>`.
- To run the Fixed Resolution sampler, use the format: `<jane|number|>`.

For this demo, only `<jane>`, `<gta5-artwork>` and `<cat-toy>` are available.
Also, `number` should be an integer in [0, 9].
'''
FOOTER = '<img id="visitor-badge" src="https://visitor-badge.glitch.me/badge?page_id=hysts.multires-textual-inversion" alt="visitor badge" />'

CACHE_EXAMPLES = os.getenv('SYSTEM') == 'spaces'


def main():
    model = Model()

    with gr.Blocks(css='style.css') as demo:
        gr.Markdown(TITLE)
        gr.Markdown(DESCRIPTION)

        with gr.Row():
            with gr.Group():
                with gr.Row():
                    prompt = gr.Textbox(label='Prompt')
                with gr.Row():
                    num_images = gr.Slider(1,
                                           9,
                                           value=1,
                                           step=1,
                                           label='Number of images')
                with gr.Row():
                    num_steps = gr.Slider(1,
                                          50,
                                          value=10,
                                          step=1,
                                          label='Number of inference steps')
                with gr.Row():
                    seed = gr.Slider(0,
                                     100000,
                                     value=100,
                                     step=1,
                                     label='Seed')
                with gr.Row():
                    run_button = gr.Button('Run')

            with gr.Column():
                result = gr.Gallery(label='Result')

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
                        inputs=[prompt],
                        outputs=[result],
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
                        inputs=[prompt],
                        outputs=[result],
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
                        inputs=[prompt],
                        outputs=[result],
                        fn=fn,
                        cache_examples=CACHE_EXAMPLES,
                    )

            prompt.submit(
                fn=model.run,
                inputs=[prompt, num_images, num_steps, seed],
                outputs=[result],
            )
            run_button.click(
                fn=model.run,
                inputs=[prompt, num_images, num_steps, seed],
                outputs=[result],
            )

        with gr.Accordion('About available prompts', open=False):
            gr.Markdown(DETAILS)
        gr.Markdown(FOOTER)
    demo.launch(enable_queue=True, share=False)


if __name__ == '__main__':
    main()
