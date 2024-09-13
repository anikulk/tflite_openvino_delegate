/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_RESHAPE_H_
#define DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_RESHAPE_H_

#include "../operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Reshape final : public OperationsBase {
 public:
  Reshape() {}
  TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_RESHAPE_H_
