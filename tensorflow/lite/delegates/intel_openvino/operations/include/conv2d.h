/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_CONV_2D_H_
#define DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_CONV_2D_H_

#include "../operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Conv2D final : public OperationsBase {
 public:
  Conv2D() {}
  TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_CONV_2D_H_
