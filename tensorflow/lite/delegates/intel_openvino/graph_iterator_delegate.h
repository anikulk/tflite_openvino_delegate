#include <unordered_set>

#include "openvino/frontend/tensorflow_lite/graph_iterator.hpp"
#include "tensorflow/lite/c/c_api_opaque.h"
#include "tensorflow/lite/c/c_api_types.h"
#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace openvinodelegate {
class GraphIteratorDelegate
    : public ov::frontend::tensorflow_lite::GraphIterator {
 public:
  GraphIteratorDelegate(TfLiteOpaqueContext* context,
                        const TfLiteOpaqueDelegateParams* params) {
    context_ = context;
    params_ = params;

    const std::unordered_set<int> inputs(
        &params->input_tensors->data[0],
        &params->input_tensors->data[params->input_tensors->size]);
    for (int i = 0; i < params->nodes_to_replace->size; i++) {
      const int delegate_node_id = params->nodes_to_replace->data[i];
      std::cout << "delegate_node_id = " << delegate_node_id << "\n";
      TfLiteOpaqueNode* delegate_node;
      TfLiteRegistrationExternal* delegate_node_registration;
      if (TfLiteOpaqueContextGetNodeAndRegistration(
              context, delegate_node_id, &delegate_node,
              &delegate_node_registration))
        return;

      int inputs_size = TfLiteOpaqueNodeNumberOfInputs(delegate_node);
      for (int k = 0; k < inputs_size; k++) {
        if (TfLiteRegistrationExternalGetBuiltInCode(
                delegate_node_registration) == kTfLiteBuiltinTransposeConv &&
            k == 0) {
          continue;
        }
        const int* inputs_data = nullptr;
        int num_inputs = 0;
        TfLiteStatus tf_status =
            TfLiteOpaqueNodeInputs(delegate_node, &inputs_data, &num_inputs);
        const int t = inputs_data[k];
        std::cout << "delegate_node_id = " << delegate_node_id << "has inputs "
                  << t << "\n";
        const void* data = nullptr;
        auto opaque_tensor = TfLiteOpaqueContextGetOpaqueTensor(context, t);
        auto allocation_type =
            TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
        if (allocation_type == kTfLiteMmapRo) {
          data = TfLiteOpaqueTensorData(opaque_tensor);

          /*     if (graph_delegate->CreateConstNode(context, t) != kTfLiteOk)
                   return kTfLiteError; */
        }
        if (inputs.count(t) != 0) {
          if (data == nullptr) {
            graph_nodes_.push_back(t);  // input : 0
          }
        }
      }

      for (int o = 0; o < params->output_tensors->size; o++) {
        const int output_tensor_idx = params->output_tensors->data[o];
        graph_nodes_.push_back(output_tensor_idx);  // output-Id : 1
        std::cout << "delegate_node_id = " << delegate_node_id << "has o/p "
                  << output_tensor_idx << "\n";
      }
      graph_nodes_.push_back(delegate_node_id);  // Operation : 0
    }
  }

  ~GraphIteratorDelegate() = default;

  /* void add_const_tensor_nodes() {
       const_nodes.insert(data);
   };*/

  /*void fill_operation_list() {

  }*/
  /// \brief Get a number of operation nodes in the graph
  size_t size() const override;

  /// \brief Set iterator to the start position
  void reset() override;

  /// \brief Move to the next node in the graph
  void next() override;

  /// \brief Returns true if iterator goes out of the range of available nodes
  bool is_end() const override;

  /// \brief Return a pointer to a decoder of the current node
  std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase> get_decoder()
      const override;

  /// \brief Returns the number of sub-graphs that can be enumerated with
  /// get_subgraph
  size_t get_subgraph_size() const override;

  /// \brief Returns iterator for a subgraph created on demand
  /// If there is no query for specific sub-graph iterator shouldn't be created
  /// idx should be in range 0..get_subgraph_size()-1
  std::shared_ptr<ov::frontend::tensorflow_lite::GraphIterator> get_subgraph(
      size_t idx) const override{};

 private:
  size_t node_index_ = 0;
  std::vector<int32_t> graph_nodes_;
  // std::vector<ov::Any> op_nodes_;
  std::vector<ov::Any> output_nodes_;
  std::vector<ov::Any> input_nodes_;
  TfLiteOpaqueContext* context_;
  const TfLiteOpaqueDelegateParams* params_;
  std::unordered_set<int32_t> inputs_;
};
}  // namespace openvinodelegate
}  // namespace tflite