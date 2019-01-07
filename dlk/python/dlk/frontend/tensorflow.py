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
"""Module of TensorFlow IO."""
from .base import BaseIO
from core.model import Model
from plugins.tf import Importer
import tensorflow as tf
from tensorflow.core.framework import graph_pb2
from tensorflow.python.lib.io import file_io
from frontend.tf_export import Exporter

from os import path


class TensorFlowIO(BaseIO):
    """IO class that reads/writes a model from/to TensorFlow pb."""

    def read(self, pb_path: str) -> Model:
        """Read TF file and load model.

        Parameters
        ----------
        pb_path : str
            Path to TF file

        Returns
        -------
        model : Model
            Loaded model

        """
        model = Model()

        # load tensorflow model
        graph_def = graph_pb2.GraphDef()
        try:
            f = open(path.abspath(pb_path), "rb")
            graph_def.ParseFromString(f.read())
            f.close()
        except IOError:
            print("Could not open file. Creating a new one.")

        # import graph
        model.graph = Importer.make_graph(graph_def)

        return model

    def write(self, model: Model, path: str) -> None:
        graph: tf.Graph = Exporter.export_graph(model)
        graph_def = graph.as_graph_def(add_shapes=True)

        file_io.atomic_write_string_to_file(path, graph_def.SerializeToString())
