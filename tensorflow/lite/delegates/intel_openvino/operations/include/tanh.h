/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_TANH_H_
#define DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_TANH_H_

#include "../operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Tanh final : public OperationsBase {
 public:
  Tanh() {}
  TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_TANH_H_
