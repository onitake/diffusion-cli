#!/usr/bin/env python3

import re
import cmd
import os
import argparse
import torch
import PIL.Image

# disable telemetry
os.environ['DISABLE_TELEMETRY'] = 'YES'
# enable local mode
os.environ['HF_HUB_OFFLINE'] = 'YES'

from diffusers import DiffusionPipeline  # noqa: E402


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
        filename = self._next()
        while os.path.exists(filename):
            self.counter += 1
            if self.counter > self.maximum:
                raise RuntimeError('Maxiumum counter value {self.maximum} exceeded')
            filename = self._next()
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

    def __init__(self, pipe, filegen, width=None, height=None, generator=None, batch=None, steps=None, guidance=None, image=None, strength=None):
        super().__init__()
        self.pipe = pipe
        self.filegen = filegen
        self.generator = generator
        self.batch = batch
        self.steps = steps
        self.guidance = guidance
        self.image = image
        self.width = width
        self.height = height
        self.strength = strength
        if self.image is None:
            # skip the example prompt for image2image
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
            image=self.image,
            strength=self.strength,
            width=self.width,
            height=self.height,
        )
        for i, image in enumerate(result.images):
            if result.nsfw_content_detected is not None and result.nsfw_content_detected[i]:
                print("skipping NSFW image")
            else:
                filename = self.filegen.next_file()
                print(f"saving to {os.path.relpath(filename)}")
                image.save(filename)

    def __str__(self):
        return f"<DiffusionShell batch={self.batch} steps={self.steps} guidance={self.guidance} strength={self.strength} width={self.width} height={self.height}>"

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
    parser.add_argument('--custom',
                        help='load a custom pipeline (community identifier or local path - try lpw_stable_diffusion for good results)')
    parser.add_argument('--width', type=int, default=512,
                        help='width of output images')
    parser.add_argument('--height', type=int, default=512,
                        help='height of output images')
    parser.add_argument('--image',
                        help='optional input image, for image2image generation')
    parser.add_argument('--strength', type=float, default=0.75,
                        help='influence of input image on result (ignored if no input image given, lower values = stronger influence)')
    parser.add_argument('--cpu', action='store_true',
                        help='process on CPU (instead of GPU)')
    parser.add_argument('--paranoia', action='store_true',
                        help='enable paranoia mode (requires models in safetensors format)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    if args.paranoia:
        def _raise():
            raise RuntimeError("Pickled models are not supported. See https://huggingsface.co/docs/diffusers/main/en/using-diffusers/using_safetensors and https://rentry.co/safetensorsguide for more information on how to obtain or convert existing models.")
        torch.load = lambda *args, **kwargs: _raise()

    if args.semantic:
        # https://huggingface.co/docs/diffusers/api/pipelines/semantic_stable_diffusion for usage.
        # Use SemanticStableDiffusionPipeline instead of StableDiffusionPipeline.
        print("Semantic Guidance (SEGA) isn't supported yet.")

    filegen = FileGen(basename=os.path.join(args.output, 'image-'), suffix='.png')
    filegen.make_dir()

    image=None
    if args.image is not None:
        print(f"Loading input image {args.image}")
        image = PIL.Image.open(args.image).convert("RGB")
        image = image.resize((args.width, args.height))

    generator = None
    if args.seed is not None:
        if args.cpu:
            generator = torch.Generator("cpu")
        else:
            generator = torch.Generator("cuda")
        generator = generator.manual_seed(args.seed)

    params = {
        'pretrained_model_name_or_path': args.model,
        'use_safetensors': args.paranoia,
    }
    if args.full or args.cpu:
        # Torch doesn't support FP16 on CPU
        params['torch_dtype'] = torch.float32
    else:
        params['torch_dtype'] = torch.float16
    if args.nsfw:
        params['safety_checker'] = None
    if args.custom is not None:
        params['custom_pipeline'] = args.custom
    pipe = DiffusionPipeline.from_pretrained(**params)
    if not args.cpu:
        pipe = pipe.to('cuda')

    try:
        DiffusionShell(
            pipe=pipe,
            filegen=filegen,
            generator=generator,
            batch=args.batch,
            steps=args.steps,
            guidance=args.guidance,
            image=image,
            width=args.width,
            height=args.height,
            strength=args.strength,
        ).cmdloop()
    except KeyboardInterrupt:
        print("Goodbye!")
