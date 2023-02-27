#!/usr/bin/env python3

import re
import cmd
import os
import argparse
import torch

# disable telemetry
os.environ['DISABLE_TELEMETRY'] = 'YES'
# enable local mode
os.environ['HF_HUB_OFFLINE'] = 'YES'

from diffusers import DiffusionPipeline

class FileGen(object):
    def __init__(self, suffix='', basename='', digits=4, maximum=9999):
        self.counter = 0
        self.suffix = suffix
        self.basename = basename
        self.digits = digits
        self.maximum = maximum

    def _next(self):
        return os.path.abspath(f'{self.basename}{self.counter:0{self.digits}d}{self.suffix}')

    def make_dir(self):
        directory = os.path.dirname(self._next())
        os.makedirs(directory, exist_ok=True)

    def next_file(self):
        exists = True
        while exists:
            filename = self._next()
            exists = os.path.exists(filename)
            if exists:
                self.counter += 1
                if self.counter > self.maximum:
                    raise RuntimeError('Maxiumum counter value {self.maximum} exceeded')
        return filename


class DiffusionShell(cmd.Cmd):
    prompt = 'prompt: '
    intro = '''
Welcome to the Diffusion Shell. Type any prompts and wait for the output to be written to the outputs directory.

Press <enter> right now to generate an example prompt. Press <enter> later to repeat the last prompt.

Prompts should be separated by commas, whitespace is (mostly) ignored. Use underscore to join compound phrases (not all models support this well).
A ! in front of a phrase adds a negative prompt. TODO: don't split on commas, prefer underscore notation instead.

The first run might take a bit longer due to compute kernel compilation and model loading.
Network access should only be needed to download custom pipelines or models. They are cached afterwards.
'''
    splitter = re.compile(r'\s*,\s*')
    notprefix = '!'

    def __init__(self, pipe, filegen, generator=None, batch=None, steps=None, guidance=None):
        super().__init__()
        self.pipe = pipe
        self.filegen = filegen
        self.generator = generator
        self.batch = batch
        self.steps = steps
        self.guidance = guidance
        self.lastcmd = "1girl, aqua eyes, baseball cap, blonde hair, closed mouth, earrings, green background, hat, hoop earrings, jewelry, looking at viewer, shirt, short hair, simple background, solo, upper body, yellow shirt"

    def default(self, line):
        if line == 'EOF':
            raise KeyboardInterrupt('Exiting on EOF')
        if ',' in line:
            prompts = self.splitter.split(line)
        else:
            prompts = [line]
        pos, neg = [], []
        for prompt in prompts:
            if prompt.startswith(self.notprefix):
                neg.append(prompt.removeprefix(self.notprefix))
            else:
                pos.append(prompt)
        print(f"positive prompt: {', '.join(pos)}\nnegative prompt: {', '.join([f'!{el}' for el in neg])}")
        result = self.pipe(
            prompt=' '.join(pos),
            negative_prompt=' '.join(neg),
            num_images_per_prompt=self.batch,
            num_inference_steps=self.steps,
            guidance_scale=self.guidance,
            generator=self.generator,
        )
        for i, image in enumerate(result.images):
            if result.nsfw_content_detected is not None and result.nsfw_content_detected[i]:
                print("skipping NSFW image")
            else:
                filename = self.filegen.next_file()
                print(f"saving to {os.path.relpath(filename)}")
                image.save(filename)


def parse_args():
    parser = argparse.ArgumentParser(description='Stable Diffusion command shell, to quickly try prompts and generate large numbers of images')
    parser.add_argument('--full', action='store_true',
                        help='calculate in full precision (float32, default is float16)')
    parser.add_argument('--batch', type=int, default=1,
                        help='batch size (may increase performance at the cost of GPU memory usage)')
    parser.add_argument('--output', default='outputs',
                        help='output directory')
    parser.add_argument('--nsfw', action='store_true',
                        help='disable the NSFW checker (use at your own risk)')
    parser.add_argument('--model', default='.',
                        help='directory containing the models')
    parser.add_argument('--seed', type=int,
                        help='seed value to make results reproducible (use a random seed instead)')
    parser.add_argument('--steps', type=int, default=50,
                        help='number of denoising steps (increases quality at the cost of speed)')
    parser.add_argument('--guidance', type=float, default=7.5,
                        help='increase adherence to the prompt (at the cost of image quality, less than 1 disables the feature)')
    parser.add_argument('--semantic', action='store_true',
                        help='enable semantic guidance, providing more fine-grained control over the prompt')
    # https://github.com/huggingface/diffusers/blob/main/examples/community/lpw_stable_diffusion.py
    parser.add_argument('--custom',
                        help='load a custom pipeline (community identifier or local path - try lpw_stable_diffusion for good results)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.semantic:
        # https://huggingface.co/docs/diffusers/api/pipelines/semantic_stable_diffusion for usage.
        # Use SemanticStableDiffusionPipeline instead of StableDiffusionPipeline.
        print("Semantic Guidance (SEGA) isn't supported yet.")

    filegen = FileGen(basename=os.path.join(args.output, 'image-'), suffix='.png')
    filegen.make_dir()

    generator = None
    if args.seed is not None:
        generator = torch.Generator("cuda").manual_seed(args.seed)

    params = {
        'pretrained_model_name_or_path': args.model,
        'torch_dtype': torch.float32 if args.full else torch.float16,
    }
    if args.nsfw:
        # can't pass this directly due to the way the SafetyChecker is instantiated
        params['safety_checker'] = None
    if args.custom is not None:
        params['custom_pipeline'] = args.custom
    pipe = DiffusionPipeline.from_pretrained(**params).to('cuda')

    try:
        DiffusionShell(pipe, filegen, generator, args.batch, args.steps, args.guidance).cmdloop()
    except KeyboardInterrupt:
        print("Goodbye!")
