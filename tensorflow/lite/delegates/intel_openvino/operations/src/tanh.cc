/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/tanh.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Tanh::CreateNode() {
  auto input_node = getInputNode(tensor_indices_[INPUT_NODE_1]);
  if (input_node == nullptr) {
    // // // TFLITE_LOG(ERROR) << "input node is null";
    return kTfLiteError;
  }

  output_node_ = ApplyActivation(input_node, kTfLiteActTanh);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
