/*
 * Copyright (C) 2024 Intel Corporation
 * SPDX-License-Identifier: Apache-2.0
 */

#ifndef DELEGATE_INTEL_OPENVINO_OPERATIONS_OPENVINO_NODE_MANAGER_H_
#define DELEGATE_INTEL_OPENVINO_OPERATIONS_OPENVINO_NODE_MANAGER_H_

#include <openvino/openvino.hpp>

#include <map>
#include <memory>
#include <unordered_set>

class NodeManager {
 public:
  NodeManager() {}
  std::shared_ptr<ov::Node> getInterimNodeOutput(int index) {
    auto node = output_at_op_index_[index];
    return node.get_node_shared_ptr();
  }
  void setOutputAtOperandIndex(int index, ov::Output<ov::Node> output) {
    output_at_op_index_.emplace(index, output);
  }

  size_t getNodeCount() const { return output_at_op_index_.size(); }

  bool isIndexAParam(int index) { return index_parameters_.count(index); }
  void insertIndexParameters(int index) { index_parameters_.insert(index); }

 private:
  std::map<int, ov::Output<ov::Node>> output_at_op_index_;
  std::unordered_set<int> index_parameters_;
};
#endif  // DELEGATE_INTEL_OPENVINO_OPERATIONS_OPENVINO_NODE_MANAGER_H_
