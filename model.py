from __future__ import annotations

import os
import subprocess
import sys

import PIL.Image
import torch
from diffusers import DPMSolverMultistepScheduler

if os.getenv('SYSTEM') == 'spaces':
    with open('patch') as f:
        subprocess.run('patch -p1'.split(),
                       cwd='multires_textual_inversion',
                       stdin=f)

sys.path.insert(0, 'multires_textual_inversion')

from pipeline import MultiResPipeline, load_learned_concepts

HF_TOKEN = os.environ.get('HF_TOKEN')


class Model:
    def __init__(self):
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        model_id = 'runwayml/stable-diffusion-v1-5'
        if self.device.type == 'cpu':
            pipe = MultiResPipeline.from_pretrained(model_id,
                                                    use_auth_token=HF_TOKEN)
        else:
            pipe = MultiResPipeline.from_pretrained(model_id,
                                                    torch_dtype=torch.float16,
                                                    revision='fp16',
                                                    use_auth_token=HF_TOKEN)
        self.pipe = pipe.to(self.device)
        self.pipe.scheduler = DPMSolverMultistepScheduler(
            beta_start=0.00085,
            beta_end=0.012,
            beta_schedule='scaled_linear',
            num_train_timesteps=1000,
            trained_betas=None,
            predict_epsilon=True,
            thresholding=False,
            algorithm_type='dpmsolver++',
            solver_type='midpoint',
            lower_order_final=True,
        )
        self.string_to_param_dict = load_learned_concepts(
            self.pipe, 'textual_inversion_outputs/')

    def run(self, prompt: str, n_images: int, n_steps: int,
            seed: int) -> list[PIL.Image.Image]:
        generator = torch.Generator(device=self.device).manual_seed(seed)
        return self.pipe([prompt] * n_images,
                         self.string_to_param_dict,
                         num_inference_steps=n_steps,
                         generator=generator)
