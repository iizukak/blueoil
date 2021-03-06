/* Copyright 2018 The Blueoil Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef DLK_MATRIX_ROW_MAJOR_TO_COL_MAJOR_H_INCLUDED
#define DLK_MATRIX_ROW_MAJOR_TO_COL_MAJOR_H_INCLUDED

#include "matrix_view.h"
#include "matrix/transpose.h"

namespace dlk {

template<typename T>
MatrixView<T, MatrixOrder::ColMajor> row_major_to_col_major(MatrixView<T, MatrixOrder::RowMajor>& m) {
  T* buf = new T[m.rows()*m.cols()];
  auto buf_mv = MatrixView<T, MatrixOrder::RowMajor>(buf, m.cols(), m.rows());
  matrix_transpose(m, buf_mv);

  return MatrixView<T, MatrixOrder::ColMajor>(buf, m.rows(), m.cols());
}

} // namespace dlk

#endif // DLK_MATRIX_ROW_MAJOR_TO_COL_MAJOR_H_INCLUDED
