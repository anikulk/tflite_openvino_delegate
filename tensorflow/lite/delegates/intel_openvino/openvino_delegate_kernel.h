/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_KERNEL_H_
#define DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_KERNEL_H_

#include <openvino/openvino.hpp>

#include <memory>

#include "tensorflow/lite/delegates/intel_openvino/openvino_delegate.h"
#include "tensorflow/lite/delegates/intel_openvino/openvino_delegate_core.h"
#include "tensorflow/lite/delegates/utils/simple_opaque_delegate.h"

namespace tflite {
namespace openvinodelegate {
class OpenVINODelegateKernel : public SimpleOpaqueDelegateKernelInterface {
 public:
  OpenVINODelegateKernel(TfLiteOpenVINODelegateOptions options)
      : ov_delegate_core_(std::make_unique<OpenVINODelegateCore>(
            "/etc/openvino/plugins.xml")) {
    options_ = options;
  }

  TfLiteStatus Init(TfLiteOpaqueContext *context,
                    const TfLiteOpaqueDelegateParams *params) override;

  TfLiteStatus Prepare(TfLiteOpaqueContext *context,
                       TfLiteOpaqueNode *node) override;

  TfLiteStatus Eval(TfLiteOpaqueContext *context,
                    TfLiteOpaqueNode *node) override;

 private:
  std::unique_ptr<OpenVINODelegateCore> ov_delegate_core_;
  TfLiteOpenVINODelegateOptions options_;
};

}  // namespace openvinodelegate
}  // namespace tflite

#endif  // DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_KERNEL_H_
