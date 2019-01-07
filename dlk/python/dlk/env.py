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
from os import path


MAX_CPU_COUNT = 8

ROOT_DIR = path.abspath(path.dirname(__file__))
PYTHON_DIR = path.join(ROOT_DIR, 'python')
DLK_DIR = path.join(PYTHON_DIR, 'dlk')
DLK_CORE_DIR = path.join(DLK_DIR, 'core')


__all__ = [
    'MAX_CPU_COUNT',
    'ROOT_DIR',
    'PYTHON_DIR',
    'DLK_DIR',
]
