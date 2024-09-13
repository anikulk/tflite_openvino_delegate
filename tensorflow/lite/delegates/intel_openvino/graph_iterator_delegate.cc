#include "graph_iterator_delegate.h"

#include "delegate_decoder.h"
#include "operations/utility.h"

namespace tflite {
namespace openvinodelegate {
size_t GraphIteratorDelegate::size() const {
  return graph_nodes_.size();
  // +input_nodes_.size() + output_nodes_.size();
}

void GraphIteratorDelegate::reset() { node_index_ = 0; }

void GraphIteratorDelegate::next() { node_index_++; }

bool GraphIteratorDelegate::is_end() const {
  if (node_index_ == size())
    return true;
  else
    return false;
}

std::shared_ptr<ov::frontend::tensorflow_lite::DecoderBase>
GraphIteratorDelegate::get_decoder() const {
  // operation
  auto delegate_node_id = graph_nodes_[node_index_];
  //
  TfLiteOpaqueNode* delegate_node;
  TfLiteRegistrationExternal* delegate_node_registration;
  if (node_index_ == 0 || node_index_ == 2) {
    // Decoder Tensor
    auto opaque_tensor =
        TfLiteOpaqueContextGetOpaqueTensor(context_, delegate_node_id);
    TfLiteType type = TfLiteOpaqueTensorType(opaque_tensor);
    auto ov_element_type = GetOVElementType(type);
    int32_t num_dims = TfLiteOpaqueTensorNumDims(opaque_tensor);
    ov::PartialShape tensor_shape = ov::PartialShape::dynamic(num_dims);
    for (int i = 0; i < num_dims; i++) {
      tensor_shape[i] =
          static_cast<int64_t>(TfLiteOpaqueTensorDim(opaque_tensor, i));
    }
    ov::frontend::tensorflow_lite::TensorMetaInfo tensor_meta_info;
    tensor_meta_info.m_partial_shape = tensor_shape;
    tensor_meta_info.m_element_type = ov_element_type;

    int64_t input_index;
    int64_t output_index;
    if (node_index_ == 0) {
      input_index = 0;
      output_index = -1;
      tensor_meta_info.m_tensor_name = "input";
    } else if (node_index_ == 1) {
      input_index = -1;
      output_index = 0;
      tensor_meta_info.m_tensor_name = "output";
    }
    return std::make_shared<DelegateDecoderTensor>(
        tensor_meta_info, input_index /*node*/, output_index);
  } else if (node_index_ == 2) {
    // Decore Operation
    TfLiteOpaqueContextGetNodeAndRegistration(context_, delegate_node_id,
                                              &delegate_node,
                                              &delegate_node_registration);

    auto builtin_code =
        TfLiteRegistrationExternalGetBuiltInCode(delegate_node_registration);
    std::string op_type, op_name;
    if (builtin_code == kTfLiteBuiltinLogistic) {
      op_type = "LOGISTIC";
      op_name = op_type + "_" + "1";
      std::cout << " Logistic\n";
    }
    int num_inputs = 0;
    const int* input_data = nullptr;
    int intput_index;
    TfLiteOpaqueNodeInputs(delegate_node, &input_data, &num_inputs);
    std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo> input_meta_info;
    std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo> output_meta_info;
    for (int k = 0; k < num_inputs; k++) {
      auto opaque_tensor =
          TfLiteOpaqueContextGetOpaqueTensor(context_, delegate_node_id);
      TfLiteType type = TfLiteOpaqueTensorType(opaque_tensor);
      auto ov_element_type = GetOVElementType(type);
      int32_t num_dims = TfLiteOpaqueTensorNumDims(opaque_tensor);
      ov::PartialShape tensor_shape = ov::PartialShape::dynamic(num_dims);
      for (int i = 0; i < num_dims; i++) {
        tensor_shape[i] =
            static_cast<int64_t>(TfLiteOpaqueTensorDim(opaque_tensor, i));
      }
      ov::frontend::tensorflow_lite::TensorMetaInfo tensor_meta_info;
      tensor_meta_info.m_partial_shape = tensor_shape;
      tensor_meta_info.m_element_type = ov_element_type;
      tensor_meta_info.m_tensor_name = "input";

      input_meta_info.push_back(tensor_meta_info);
    }
    int num_outputs = 0;
    const int* output_data = nullptr;
    int output_index;
    TfLiteOpaqueNodeOutputs(delegate_node, &output_data, &num_outputs);
    for (int k = 0; k < num_outputs; k++) {
      auto opaque_tensor =
          TfLiteOpaqueContextGetOpaqueTensor(context_, delegate_node_id);
      TfLiteType type = TfLiteOpaqueTensorType(opaque_tensor);
      auto ov_element_type = GetOVElementType(type);
      int32_t num_dims = TfLiteOpaqueTensorNumDims(opaque_tensor);
      ov::PartialShape tensor_shape = ov::PartialShape::dynamic(num_dims);
      for (int i = 0; i < num_dims; i++) {
        tensor_shape[i] =
            static_cast<int64_t>(TfLiteOpaqueTensorDim(opaque_tensor, i));
      }
      ov::frontend::tensorflow_lite::TensorMetaInfo tensor_meta_info;
      tensor_meta_info.m_partial_shape = tensor_shape;
      tensor_meta_info.m_element_type = ov_element_type;
      tensor_meta_info.m_tensor_name = "output";
      output_meta_info.push_back(tensor_meta_info);
    }
    return std::make_shared<DelegateDecoderOperation>(
        op_type, op_name, input_meta_info, output_meta_info);
  }
}
// if (node_index_ < graph_nodes_size()) {

/*
const int *inputs_data = nullptr;
int num_inputs = 0;
int input_index;
TfLiteOpaqueNodeInputs(delegate_node, &inputs_data, &num_inputs);
for (int k = 0; k < num_inputs; k++) {
  const int t = inputs_data[k];
  const void *data = nullptr;
  auto opaque_tensor = TfLiteOpaqueContextGetOpaqueTensor(context_, t);
  auto allocation_type = TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
  if (allocation_type == kTfLiteMmapRo) {
    data = TfLiteOpaqueTensorData(opaque_tensor);
  }
  std::cout << "Calling get_decoder\n";


  const int *inputs_data = nullptr;
  int num_inputs = 0;
  int input_index;
  TfLiteOpaqueNodeInputs(delegate_node, &inputs_data, &num_inputs);
  for (int k = 0; k < num_inputs; k++) {
    const int t = inputs_data[k];
    const void *data = nullptr;
    auto opaque_tensor = TfLiteOpaqueContextGetOpaqueTensor(context_, t);
    auto allocation_type =
        TfLiteOpaqueTensorGetAllocationType(opaque_tensor);
    if (allocation_type == kTfLiteMmapRo) {
      data = TfLiteOpaqueTensorData(opaque_tensor);
    }
    input_index = t;
  }

  int num_outputs = 0;
  const int *output_data = nullptr;
  int output_index;
  if (TfLiteOpaqueNodeOutputs(delegate_node, &output_data, &num_outputs) ==
      kTfLiteOk) {

    // TODO: Assume there is only one output from the op. Handle multiple
    // outputs from one op in next version.
    if (num_outputs == 1) {
      output_index = output_data[0];
    }
  }

  return std::make_shared<DelegateDecoderTensor>(tensor_meta_info,
                                                 input_index, output_index); */

size_t GraphIteratorDelegate::get_subgraph_size() const { return 0; }
}  // namespace openvinodelegate
}  // namespace tflite
