/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "../include/logistic.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus Logistic::CreateNode() {
  auto input_node = getInputNode(tensor_indices_[INPUT_NODE_1]);
  if (input_node == nullptr) {
    return kTfLiteError;
  }
  output_node_ = ApplyActivation(input_node, kTfLiteActSigmoid);
  return kTfLiteOk;
}

}  // namespace openvinodelegate
}  // namespace tflite
