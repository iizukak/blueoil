# -*- coding: utf-8 -*-
# Copyright 2019 The Blueoil Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# =============================================================================
"""
Script that automatically runs all of the folllowing steps.

- Import onnx, lmnet's export and config.
- Generate all cpp source headers and other control files like Makefile.
"""
import click
from os import path
import shutil

from core.config import Config
from core.graph import Graph
from core.model import Model
from core.params import Params
from core.optimizer import Optimizer
from code_generater import CodeGenerater
from frontend import TensorFlowIO

import utils

SCRITPS_DIR = path.abspath(path.dirname(__file__))
DLK_ROOT_DIR = path.abspath(path.join(SCRITPS_DIR, '..'))
ROOT_DIR = path.abspath(path.join(SCRITPS_DIR, '../../..'))


def optimize_graph_step(model: Model, config: Config) -> None:
    """Optimze graph in the model.

    Parameters
    ----------
    model : Model
        Model that contains the graph

    config : Config
        Collection of configurations

    """
    graph: Graph = model.graph
    optim = Optimizer()
    optim.transpose_NHWC(graph)
    optim.precompute(graph, config.activate_hard_quantization)
    if config.threshold_skipping:
        optim.threshold_skipping(graph)


def generate_code_step(model: Model, config: Config) -> None:
    """Generate code for the model.

    Parameters
    ----------
    model : Model
        Model the code generation is based on

    config : Config
        Collection of configurations

    """
    graph: Graph = model.graph
    params = Params(graph, config)

    builder = CodeGenerater(graph,
                            params,
                            config)

    builder.reuse_output_buffers()
    builder.generate_files_from_template()
    builder.generate_inputs()

    if config.activate_hard_quantization:
        builder.generate_scaling_factors()

    if config.threshold_skipping:
        builder.generate_thresholds()

    if config.use_tvm:
        builder.generate_tvm_libraries()


def run(input_path: str,
        dest_dir_path: str,
        project_name: str,
        activate_hard_quantization: bool,
        threshold_skipping: bool = False,
        num_pe: int = 16,
        use_tvm: bool = False,
        use_onnx: bool = False,
        debug: bool = False,
        cache_dma: bool = False):

    output_dlk_test_dir = path.join(dest_dir_path, f'{project_name}.test')
    optimized_pb_path = path.join(dest_dir_path, f'{project_name}')
    optimized_pb_path += '.onnx' if use_onnx else '.pb'
    output_project_path = path.join(dest_dir_path, f'{project_name}.prj')

    config = Config(num_pe=num_pe,
                    activate_hard_quantization=activate_hard_quantization,
                    threshold_skipping=threshold_skipping,
                    tvm_path=(path.join(ROOT_DIR, 'tvm') if use_tvm else None),
                    test_dir=output_dlk_test_dir,
                    optimized_pb_path=optimized_pb_path,
                    output_pj_path=output_project_path,
                    debug=debug,
                    cache_dma=cache_dma
                    )

    dest_dir_path = path.abspath(dest_dir_path)
    utils.make_dirs(dest_dir_path)

    click.echo('import pb file')
    if use_onnx:
        try:
            __import__('onnx')
        except ImportError:
            raise ImportError('ONNX is required but not installed.')
        from frontend.base import BaseIO
        from frontend.onnx import OnnxIO
        io: BaseIO = OnnxIO()

    else:
        io = TensorFlowIO()
    model: Model = io.read(input_path)

    click.echo('optimize graph step: start')
    optimize_graph_step(model, config)
    click.echo('optimize graph step: done!')

    click.echo('generate code step: start')
    generate_code_step(model, config)
    click.echo(f'generate code step: done!')


@click.command(context_settings=dict(help_option_names=['-h', '--help']))
@click.option(
    "-i",
    "--input_path",
    type=click.Path(exists=True),
    help="onnx protobuf path which you want to convert to C codes",
)
@click.option(
    "-o",
    "--output_path",
    help="output path which you want to export any generated files",
)
@click.option(
    "-p",
    "--project_name",
    help="project name which you'll generate",
)
@click.option(
    "-hq",
    "--activate_hard_quantization",
    is_flag=True,
    default=False,
    help="activate hard quantization optimization",
)
@click.option(
    "-ts",
    "--threshold_skipping",
    is_flag=True,
    default=False,
    help="activate threshold skip optimization",
)
@click.option(
    '-pe',
    '--num_pe',
    type=int,
    default=16,
    help='set number of PE used in FPGA IP',
)
@click.option(
    "-tvm",
    "--use_tvm",
    is_flag=True,
    default=False,
    help="optimize CPU/GPU operations using TVM",
)
@click.option(
    "-onnx",
    "--use_onnx",
    is_flag=True,
    default=False,
    help="if the input file is in ONNX format"
)
@click.option(
    "-dbg",
    "--debug",
    is_flag=True,
    default=False,
    help="add debug code to the generated project",
)
@click.option(
    "-cache",
    "--cache_dma",
    is_flag=True,
    default=False,
    help="use cached DMA buffers",
)
def main(input_path,
         output_path,
         project_name,
         activate_hard_quantization,
         threshold_skipping,
         num_pe,
         use_tvm,
         use_onnx,
         debug,
         cache_dma):

    click.echo('start running')
    run(input_path=input_path,
        dest_dir_path=output_path,
        project_name=project_name,
        activate_hard_quantization=activate_hard_quantization,
        threshold_skipping=threshold_skipping,
        num_pe=num_pe,
        use_tvm=use_tvm,
        use_onnx=use_onnx,
        debug=debug,
        cache_dma=cache_dma)


if __name__ == '__main__':
    main()
