/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_CORE_H_
#define DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_CORE_H_

#include <openvino/openvino.hpp>

#include <memory>
#include <string>
#include <vector>

#include "tensorflow/lite/delegates/intel_openvino/openvino_delegate.h"
#include "tensorflow/lite/delegates/intel_openvino/openvino_graph_builder.h"

namespace tflite {
namespace openvinodelegate {
class OpenVINODelegateCore {
 public:
  explicit OpenVINODelegateCore(std::string plugins_path)
      : ov_core_(ov::Core(plugins_path)) {}
  TfLiteStatus Init();

  std::vector<int> getComputeInputs() { return compute_inputs_; }

  std::vector<int> getOutputs() { return outputs_; }

  ov::InferRequest getInferRequest() const { return infer_request_; }

  TfLiteStatus CreateModel(TfLiteOpaqueContext *context,
                           const TfLiteOpaqueDelegateParams *params,
                           const TfLiteOpenVINODelegateOptions *options);
  TfLiteStatus CompileAndInfer();

 private:
  TfLiteStatus BuildModelFromCache(TfLiteOpaqueContext *context,
                                   const TfLiteOpaqueDelegateParams *params,
                                   std::string cached_ir);
  TfLiteStatus BuildModel();
  TfLiteStatus InitializeBuilder(TfLiteOpaqueContext *context,
                                 const TfLiteOpaqueDelegateParams *params);
  std::unique_ptr<OpenVINOGraphBuilder> openvino_graph_builder_;
  ov::Core ov_core_;
  std::shared_ptr<ov::Model> model_;
  ov::CompiledModel compiled_model_;
  std::vector<int> compute_inputs_;
  std::vector<int> outputs_;
  ov::InferRequest infer_request_;
};

}  // namespace openvinodelegate
}  // namespace tflite

#endif  // DELEGATE_INTEL_OPENVINO_OPENVINO_DELEGATE_CORE_H_
