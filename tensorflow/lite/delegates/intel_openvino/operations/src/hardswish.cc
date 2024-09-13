/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/hardswish.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus HardSwish::CreateNode() {
  auto input_node = getInputNode(tensor_indices_[INPUT_NODE_1]);
  if (input_node == nullptr) {
    return kTfLiteError;
  }
  output_node_ = std::make_shared<ov::op::v4::HSwish>(input_node);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
