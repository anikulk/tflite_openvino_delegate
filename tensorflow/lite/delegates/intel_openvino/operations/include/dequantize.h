/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_DEQUANTIZE_H_
#define DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_DEQUANTIZE_H_

#include "../operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Dequantize final : public OperationsBase {
 public:
  Dequantize() {}
  TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_DEQUANTIZE_H_
