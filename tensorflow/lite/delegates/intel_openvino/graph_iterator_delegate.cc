#include "graph_iterator_delegate.h"

#include "delegate_decoder.h"
#include "operations/utility.h"

namespace tflite {
namespace openvinodelegate {
size_t GraphIteratorDelegate::size() const {
  return graph_nodes_.size() + input_nodes_.size() + output_nodes_.size();
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
  // size() of all decoders 
  // return all_decoders[node_index]

  // operation
  if (node_index_ >= output_nodes_.size() + input_nodes_.size() + const_nodes_.size() ) {
    auto delegate_node_id = graph_nodes_[node_index_ - output_nodes_.size() - input_nodes_.size() - const_nodes_.size()];
    TfLiteOpaqueNode* delegate_node;
    TfLiteRegistrationExternal* delegate_node_registration;
    TfLiteOpaqueContextGetNodeAndRegistration(context_, delegate_node_id,
                                              &delegate_node,
                                              &delegate_node_registration);

    auto builtin_code =
        TfLiteRegistrationExternalGetBuiltInCode(delegate_node_registration);
    std::string op_type, op_name;

    // move below logic to helper function;
    if (builtin_code == kTfLiteBuiltinLogistic) {
      op_type = "LOGISTIC";
    }
    else if(builtin_code == kTfLiteBuiltinReshape) {
      op_type = "RESHAPE";
    }
    else if(builtin_code == kTfLiteBuiltinAdd) {
      op_type = "ADD";   
    }
    else if(builtin_code == kTfLiteBuiltinDequantize) {
      op_type = "DEQUANTIZE";   
    }
    else if(builtin_code == kTfLiteBuiltinConv2d) {
      op_type = "CONV_2D";   
    }
    else if(builtin_code == kTfLiteBuiltinDepthwiseConv2d) {
      op_type = "DEPTHWISE_CONV_2D";   
    }
    else if(builtin_code == kTfLiteBuiltinConv2d) {
      op_type = "CONV_2D";   
    }
    else if(builtin_code == kTfLiteBuiltinResizeBilinear) {
      op_type = "RESIZE_BILINEAR";   
    }
    else if(builtin_code == kTfLiteBuiltinConcatenation) {
      op_type = "CONCATENATION";   
    }
    else if(builtin_code == kTfLiteBuiltinAveragePool2d) {
      op_type = "AVERAGE_POOL_2D";   
    }
    else if(builtin_code == kTfLiteBuiltinSoftmax) {
      op_type = "SOFTMAX";   
    }

    op_name = op_type + "_" + std::to_string(node_index_);
    int num_inputs = 0;
    const int* input_data = nullptr;
    int intput_index;
    TfLiteOpaqueNodeInputs(delegate_node, &input_data, &num_inputs);
    std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo> input_meta_info;
    std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo> output_meta_info;
    for (int k = 0; k < num_inputs; k++) {
      std::cout << "Line number : " << __LINE__ << " in file " << __FILE__<< ":::> input: " << input_data[k] << "\n";
      auto opaque_tensor =
          TfLiteOpaqueContextGetOpaqueTensor(context_, input_data[k]);
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
      tensor_meta_info.m_tensor_name = TfLiteOpaqueTensorName(opaque_tensor); // "input";
      tensor_meta_info.m_tensor_data  = (const uint8_t*)TfLiteOpaqueTensorData(opaque_tensor);
      if (tensor_meta_info.m_tensor_data == NULL) {
        std::cout << "Line number : " << __LINE__ << " in file " << __FILE__<< ":::> nullptr : node_index = " << node_index_ << " delegate_node_id = " << delegate_node_id << "\n";
      }
      else {
        std::cout << "Line number : " << __LINE__ << " in file " << __FILE__<< ":::> not a  nullptr : node_index = " << node_index_ << " delegate_node_id = " << delegate_node_id << "\n";
      } 


      input_meta_info.push_back(tensor_meta_info);
      std::cout << __FILE__ << " " << __LINE__<< " input_meta_info[" << k << "] = " << input_meta_info[k].m_tensor_name << "\n";
    }
    int num_outputs = 0;
    const int* output_data = nullptr;
    int output_index;
    TfLiteOpaqueNodeOutputs(delegate_node, &output_data, &num_outputs);
    for (int k = 0; k < num_outputs; k++) {
      auto opaque_tensor =
          TfLiteOpaqueContextGetOpaqueTensor(context_, output_data[k]);
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
      tensor_meta_info.m_tensor_name = TfLiteOpaqueTensorName(opaque_tensor); // "input";

      output_meta_info.push_back(tensor_meta_info);
    }

    return std::make_shared<DelegateDecoderOperation>(
        op_type, op_name, input_meta_info, output_meta_info, TfLiteOpaqueNodeGetBuiltinData(delegate_node));
  }
  else if (node_index_ < input_nodes_.size()) {
    auto delegate_node_id = input_nodes_[node_index_];
    std::cout << __FILE__ << " " << __PRETTY_FUNCTION__ << " " << __LINE__ << " input_node = " << node_index_ << " " << delegate_node_id << " \n";
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

    int64_t input_index = node_index_;
    int64_t output_index = -1;
    tensor_meta_info.m_tensor_name = TfLiteOpaqueTensorName(opaque_tensor); //"input" + std::to_string(input_index);
    // input_index_ = input_index_ + 1;
    std::cout << __LINE__ << " " << __FILE__ << "Creating " << tensor_meta_info.m_tensor_name << " Decoder tensor ip \n";
        std::cout << __LINE__ << " " << __FILE__ << input_index << " " << output_index << " Decoder tensor\n";  
    return std::make_shared<DelegateDecoderTensor>(
        tensor_meta_info, input_index /*node*/, output_index);
  } else if (node_index_ >= input_nodes_.size() && node_index_ < (input_nodes_.size() + output_nodes_.size())) {
    auto delegate_node_id = output_nodes_[node_index_ - input_nodes_.size()];
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
      int64_t input_index = -1;
      int64_t output_index = node_index_ - input_nodes_.size();
      tensor_meta_info.m_tensor_name = TfLiteOpaqueTensorName(opaque_tensor); // "output" + std::to_string(output_index);
    std::cout << __LINE__ << " " << __FILE__ << "Creating " << tensor_meta_info.m_tensor_name << " Decoder tensor op\n";
    std::cout << __LINE__ << " " << __FILE__ << input_index << " " << output_index << " Decoder tensor\n";
    return std::make_shared<DelegateDecoderTensor>(
        tensor_meta_info, input_index /*node*/, output_index);
  } 
}


size_t GraphIteratorDelegate::get_subgraph_size() const { return 0; }
}  // namespace openvinodelegate
}  // namespace tflite
