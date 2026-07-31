"""YAML parser restricted to the RSF-DETR paper architecture."""

import ast
import contextlib

import torch
import torch.nn as nn

from ultralytics.nn.modules import AIFI, BasicBlock, Blocks, Concat, Conv, ConvNormLayer, RepC3, RTDETRDecoder
from ultralytics.utils import LOGGER, colorstr
from ultralytics.utils.torch_utils import make_divisible

from . import CERB, MESA, SFFN

PAPER_MODULES = {
    "AIFI": AIFI,
    "BasicBlock": BasicBlock,
    "Blocks": Blocks,
    "CERB": CERB,
    "Concat": Concat,
    "Conv": Conv,
    "ConvNormLayer": ConvNormLayer,
    "MESA": MESA,
    "RepC3": RepC3,
    "RTDETRDecoder": RTDETRDecoder,
    "SFFN": SFFN,
}


def parse_rsf_detr_model(config: dict, channels: int, verbose: bool = True):
    """Build an RSF-DETR or paper ablation model from a YAML dictionary."""
    max_channels = float("inf")
    num_classes = config.get("nc")
    activation = config.get("activation")
    scales = config.get("scales")
    depth = config.get("depth_multiple", 1.0)
    width = config.get("width_multiple", 1.0)
    if scales:
        scale = config.get("scale") or next(iter(scales))
        depth, width, max_channels = scales[scale]

    if activation:
        Conv.default_act = eval(activation)
        if verbose:
            LOGGER.info(f"{colorstr('activation:')} {activation}")

    if verbose:
        LOGGER.info(f"\n{'':>3}{'from':>20}{'n':>3}{'params':>10}  {'module':<30}{'arguments':<30}")

    channel_list = [channels]
    layers, saved_outputs = [], []
    repeat_modules = {MESA, RepC3}
    channel_modules = {CERB, Conv, ConvNormLayer, MESA, RepC3}

    for index, (source, repeats, module_name, arguments) in enumerate(config["backbone"] + config["head"]):
        if module_name.startswith("nn."):
            module = getattr(torch.nn, module_name[3:])
        else:
            if module_name not in PAPER_MODULES:
                raise ValueError(
                    f"Unsupported module '{module_name}' in RSF-DETR config. "
                    f"Allowed modules: {', '.join(sorted(PAPER_MODULES))}, nn.*"
                )
            module = PAPER_MODULES[module_name]

        arguments = list(arguments)
        for argument_index, argument in enumerate(arguments):
            if isinstance(argument, str):
                with contextlib.suppress(ValueError, SyntaxError):
                    arguments[argument_index] = (
                        num_classes if argument == "nc" else ast.literal_eval(argument)
                    )

        original_repeats = repeats
        repeats = max(round(repeats * depth), 1) if repeats > 1 else repeats
        output_channels = channel_list[source] if isinstance(source, int) else channel_list[source[0]]

        if module in channel_modules:
            input_channels = channel_list[source]
            output_channels = arguments[0]
            if output_channels != num_classes:
                output_channels = make_divisible(min(output_channels, max_channels) * width, 8)
            arguments = [input_channels, output_channels, *arguments[1:]]
            if module in repeat_modules:
                arguments.insert(2, repeats)
                repeats = 1
        elif module in {AIFI, SFFN}:
            output_channels = channel_list[source]
            arguments = [output_channels, *arguments]
        elif module is Blocks:
            block_type = PAPER_MODULES[arguments[1]]
            input_channels = channel_list[source]
            output_channels = arguments[0] * block_type.expansion
            arguments = [input_channels, arguments[0], block_type, *arguments[2:]]
        elif module is Concat:
            output_channels = sum(channel_list[item] for item in source)
        elif module is RTDETRDecoder:
            arguments.insert(1, [channel_list[item] for item in source])

        layer = (
            nn.Sequential(*(module(*arguments) for _ in range(repeats)))
            if repeats > 1
            else module(*arguments)
        )
        module_type = module.__name__
        layer.np = sum(parameter.numel() for parameter in layer.parameters())
        layer.i, layer.f, layer.type = index, source, module_type
        if verbose:
            LOGGER.info(
                f"{index:>3}{str(source):>20}{original_repeats:>3}{layer.np:10.0f}  "
                f"{module_type:<30}{str(arguments):<30}"
            )
        saved_outputs.extend(
            item % index
            for item in ([source] if isinstance(source, int) else source)
            if item != -1
        )
        layers.append(layer)
        if index == 0:
            channel_list = []
        channel_list.append(output_channels)

    return nn.Sequential(*layers), sorted(saved_outputs)
