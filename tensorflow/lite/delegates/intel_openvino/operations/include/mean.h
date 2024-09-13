/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_MEAN_H_
#define DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_MEAN_H_

#include "../operations_base.h"

namespace tflite {
namespace openvinodelegate {

class Mean final : public OperationsBase {
 public:
  Mean() {}
  TfLiteStatus CreateNode() override;
};

}  // namespace openvinodelegate
}  // namespace tflite
#endif  // DELEGATE_INTEL_OPENVINO_OPERATIONS_INCLUDE_MEAN_H_
