/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#include "tensorflow/lite/delegates/intel_openvino/openvino_delegate_core.h"

#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <unistd.h>
#include <unordered_set>

#include "graph_iterator_delegate.h"
#include "openvino/frontend/tensorflow_lite/frontend.hpp"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/delegates/intel_openvino/operations/openvino_node_manager.h"

namespace tflite {
namespace openvinodelegate {

TfLiteStatus OpenVINODelegateCore::Init() {
  std::vector<std::string> ov_devices = ov_core_.get_available_devices();
  if (std::find(ov_devices.begin(), ov_devices.end(), "CPU") ==
      ov_devices.end()) {
    // TFLITE_LOG(ERROR) << "Could not find plugin for CPU";
    return kTfLiteDelegateError;
  } else {
    return kTfLiteOk;
  }
}

TfLiteStatus OpenVINODelegateCore::InitializeBuilder(
    TfLiteOpaqueContext *context, const TfLiteOpaqueDelegateParams *params) {
  if (context == nullptr || params == nullptr)
    return kTfLiteError;

  auto tflite_fe = std::make_shared<ov::frontend::tensorflow_lite::FrontEnd>();
  std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator> graph_delegate =
      std::make_shared<GraphIteratorDelegate>(context, params);
  std::cout << "Num entries in graph_delegate " << graph_delegate->size();
  auto input_model = tflite_fe->load(graph_delegate);
  auto model = tflite_fe->convert(input_model);
  
  const std::unordered_set<int> inputs(
      &params->input_tensors->data[0],
      &params->input_tensors->data[params->input_tensors->size]);

  for (int o = 0; o < params->output_tensors->size; o++) {
    const int output_tensor_idx = params->output_tensors->data[o];
    outputs_.push_back(output_tensor_idx);
  }

  for (int i = 0; i < params->nodes_to_replace->size; i++) {
    const int delegate_node_id = params->nodes_to_replace->data[i];
    TfLiteOpaqueNode *delegate_node;
    TfLiteRegistrationExternal *delegate_node_registration;
    if (TfLiteOpaqueContextGetNodeAndRegistration(
            context, delegate_node_id, &delegate_node,
            &delegate_node_registration) != kTfLiteOk)
      return kTfLiteError;

    const int *inputs_data = nullptr;
    int num_inputs = 0;
    if (TfLiteOpaqueNodeInputs(delegate_node, &inputs_data, &num_inputs) !=
        kTfLiteOk)
      return kTfLiteError;
    for (int k = 0; k < num_inputs; k++) {
      const int t = inputs_data[k];
      if (t == kTfLiteOptionalTensor)
        continue;
      const void *data = nullptr;
      auto opaque_tensor = TfLiteOpaqueContextGetOpaqueTensor(context, t);
      auto allocation_type = TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
      if (allocation_type == kTfLiteMmapRo) {
        data = TfLiteOpaqueTensorData(opaque_tensor);
      }
      if (inputs.count(t) != 0) {
        if (data == nullptr) {
          compute_inputs_.push_back(t);
        }
      }
    }

  }

  model_ = model;
  return kTfLiteOk;
}

TfLiteStatus OpenVINODelegateCore::BuildModel() {
  if (!openvino_graph_builder_)
    return kTfLiteError;
  model_ =
      std::make_shared<ov::Model>(openvino_graph_builder_->getResultNodes(),
                                  openvino_graph_builder_->getInputParams());
  return kTfLiteOk;
}

TfLiteStatus OpenVINODelegateCore::BuildModelFromCache(
    TfLiteOpaqueContext *context, const TfLiteOpaqueDelegateParams *params,
    std::string cached_openvino_ir) {

  const std::unordered_set<int> inputs(
      &params->input_tensors->data[0],
      &params->input_tensors->data[params->input_tensors->size]);
  for (int i = 0; i < params->nodes_to_replace->size; i++) {
    const int delegate_node_id = params->nodes_to_replace->data[i];
    TfLiteOpaqueNode *delegate_node;
    TfLiteRegistrationExternal *delegate_node_registration;
    if (TfLiteOpaqueContextGetNodeAndRegistration(
            context, delegate_node_id, &delegate_node,
            &delegate_node_registration) != kTfLiteOk)
      return kTfLiteError;

    const int *inputs_data = nullptr;
    int num_inputs = 0;
    if (TfLiteOpaqueNodeInputs(delegate_node, &inputs_data, &num_inputs) !=
        kTfLiteOk)
      return kTfLiteError;
    for (int k = 0; k < num_inputs; k++) {
      const int t = inputs_data[k];
      const void *data = nullptr;
      auto opaque_tensor = TfLiteOpaqueContextGetOpaqueTensor(context, t);
      auto allocation_type = TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
      if (allocation_type == kTfLiteMmapRo) {
        data = TfLiteOpaqueTensorData(opaque_tensor);
      }
      if (inputs.count(t) != 0) {
        if (data == nullptr)
          compute_inputs_.push_back(t);
      }
    }
  }
  for (int o = 0; o < params->output_tensors->size; o++) {
    const int output_tensor_idx = params->output_tensors->data[o];
    outputs_.push_back(output_tensor_idx);
  }
  model_ = ov_core_.read_model(cached_openvino_ir);
  if (!model_)
    return kTfLiteError;
  return kTfLiteOk;
}

TfLiteStatus OpenVINODelegateCore::CompileAndInfer() {
  // TODO: get device string from flags
  std::string deviceStr = "CPU";
  ov::AnyMap config;
  // Below param helps accelerate inference on NPU device. It helps in HW
  // acceleration.
  // config["NPU_COMPILATION_MODE_PARAMS"] = "enable-se-ptrs-operations=true";

  compiled_model_ = ov_core_.compile_model(model_, deviceStr);
  //, config);
  infer_request_ = compiled_model_.create_infer_request();
  return kTfLiteOk;
}

TfLiteStatus OpenVINODelegateCore::CreateModel(
    TfLiteOpaqueContext *context, const TfLiteOpaqueDelegateParams *params,
    const TfLiteOpenVINODelegateOptions *delegate_options) {
  // If cache_dir is set, and
  //    if cached model exists BuildModelFromCache
  //    else initialize and build model from tflite runtime
  if (!delegate_options->cache_dir.empty() &&
      !delegate_options->model_token.empty()) {
    std::string cache_file_name = delegate_options->cache_dir + "/" +
                                  delegate_options->model_token + ".xml";

    ov_core_.set_property(ov::cache_dir(delegate_options->cache_dir));
    if (access(delegate_options->cache_dir.c_str(), R_OK) == 0) {
      // TFLITE_LOG(ERROR) << "Read access is there\n";
      if (std::filesystem::exists(cache_file_name)) {
        auto status = BuildModelFromCache(context, params, cache_file_name);
        if (status == kTfLiteOk)
          return status;
      }
      // TFLITE_LOG(ERROR) << "File absent\n";
    }
  }
  // If cache file is absent or caching is not enabled
  // Initialize model from TFLite runtime

  auto status = InitializeBuilder(context, params);
  if (status != kTfLiteOk)
    return status;

  //status = BuildModel();
  //if (status != kTfLiteOk)
  //    return status;
  if (!delegate_options->cache_dir.empty() &&
      !delegate_options->model_token.empty()) {
    std::string cache_file_name = delegate_options->cache_dir + "/" +
                                  delegate_options->model_token + ".xml";
    if (access(delegate_options->cache_dir.c_str(), W_OK) == 0) {
      ov::serialize(model_, cache_file_name);
    } else {
      // TFLITE_LOG(ERROR) << "Serialization failed\n Continue from built
      // model\n";
    }
  }
  return kTfLiteOk;
}
} // namespace openvinodelegate
} // namespace tflite
