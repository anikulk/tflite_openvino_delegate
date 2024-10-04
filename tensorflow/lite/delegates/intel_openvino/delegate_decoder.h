#include "openvino/frontend/tensorflow_lite/decoder.hpp"
#include "tensorflow/lite/tools/logging.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "operations/utility.h"

struct TensorMetaInfo {
  std::shared_ptr<ov::frontend::tensorflow_lite::QuantizationInfo>
      m_quantization_info;
  std::shared_ptr<ov::frontend::tensorflow_lite::SparsityInfo> m_sparsity_info;
  ov::PartialShape m_partial_shape;
  ov::element::Type m_element_type;
  const uint8_t *m_tensor_data;
  std::string m_tensor_name;
};

class DelegateDecoderOperation
    : public ov::frontend::tensorflow_lite::DecoderBaseOperation {
 public:
  explicit DelegateDecoderOperation(
      const std::string &type, const std::string &name,
      std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>
          input_tensor_info,
      std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>
          output_tensor_info, void* builtin_data) {
    op_type_ = type;
    op_name_ = name;
    input_tensor_info_ = input_tensor_info;
    output_tensor_info_ = output_tensor_info;
    builtin_data_ = builtin_data;

  };

  /// \brief Get input tensor info
  ov::frontend::tensorflow_lite::TensorMetaInfo get_input_tensor_info(
      size_t idx) const override {
    return input_tensor_info_[idx];
  }  //

  std::string get_input_tensor_name(size_t idx) const override {
    return input_tensor_info_[idx].m_tensor_name;
  }
  /// \brief Get input tensor type by index
  ov::element::Type get_input_tensor_type(size_t idx) const override {
    return input_tensor_info_[idx].m_element_type;
  };

  size_t get_input_size() const override { return input_tensor_info_.size(); }

  /// \brief Get output tensor info_
  ov::frontend::tensorflow_lite::TensorMetaInfo get_output_tensor_info(
      size_t idx) const override {
    return output_tensor_info_[idx];
  }  //

  /// \brief Get a number of outputs
  size_t get_output_size() const override { return output_tensor_info_.size(); }

  /// \brief Get output tensor name by index
  std::string get_output_tensor_name(size_t idx) const override {
    return output_tensor_info_[idx].m_tensor_name;
  }

  /// \brief Get output tensor type by index
  ov::element::Type get_output_tensor_type(size_t idx) const override {
    return output_tensor_info_[idx].m_element_type;
  }

  
  ov::Any get_attribute(const std::string &name) const override {
   if (name == "fused_activation_function" && op_type_ == "ADD")  {
    TfLiteAddParams* data  = reinterpret_cast<TfLiteAddParams*>(builtin_data_);
    return tflite::openvinodelegate::get_activation_string(data->activation);
   }
   else if (name == "new_shape" && op_type_ == "RESHAPE") {
        TfLiteReshapeParams* data = reinterpret_cast<TfLiteReshapeParams*>(builtin_data_);
        if (data->num_dimensions == 0) {
          return {};
        }
        else {
          const auto new_shape = std::vector<int32_t>(data->shape,  data->shape + data->num_dimensions);
          return new_shape;
        }
   } else if (name == "strides" && op_type_ == "CONV_2D") {
    TfLiteConvParams* data = reinterpret_cast<TfLiteConvParams*>(builtin_data_);
        return std::vector<int64_t>{1,
                                    static_cast<int64_t>(data->stride_height),
                                    static_cast<int64_t>(data->stride_width),
                                    1};
    } else if (name == "padding" && op_type_ == "CONV_2D") {
         TfLiteConvParams* data = reinterpret_cast<TfLiteConvParams*>(builtin_data_);
        return "SAME";
    } else if (name == "dilations" && op_type_ == "CONV_2D") {
       TfLiteConvParams* data = reinterpret_cast<TfLiteConvParams*>(builtin_data_);
        return std::vector<int64_t>{1,
                                    static_cast<int64_t>(data->dilation_height_factor),
                                    static_cast<int64_t>(data->dilation_width_factor),
                                    1};
    } else if (name == "data_format" && op_type_ == "CONV_2D") {
        return "NHWC";
    } else if (name == "activation" && op_type_ == "CONV_2D") {
      TfLiteConvParams* data = reinterpret_cast<TfLiteConvParams*>(builtin_data_);
    return tflite::openvinodelegate::get_activation_string(data->activation);
    } 
     else if (name == "strides" && op_type_ == "DEPTHWISE_CONV_2D") {
    TfLiteDepthwiseConvParams* data = reinterpret_cast<TfLiteDepthwiseConvParams*>(builtin_data_);
        return std::vector<int64_t>{1,
                                    static_cast<int64_t>(data->stride_height),
                                    static_cast<int64_t>(data->stride_width),
                                    1};
    } else if (name == "padding" && op_type_ == "DEPTHWISE_CONV_2D") {
         TfLiteDepthwiseConvParams* data = reinterpret_cast<TfLiteDepthwiseConvParams*>(builtin_data_);
        return "SAME";
    } else if (name == "dilations" && op_type_ == "DEPTHWISE_CONV_2D") {
       TfLiteDepthwiseConvParams* data = reinterpret_cast<TfLiteDepthwiseConvParams*>(builtin_data_);
        return std::vector<int64_t>{1,
                                    static_cast<int64_t>(data->dilation_height_factor),
                                    static_cast<int64_t>(data->dilation_width_factor),
                                    1};
    } else if (name == "data_format" && op_type_ == "DEPTHWISE_CONV_2D") {
        return "NHWC";
    } else if (name == "activation" && op_type_ == "DEPTHWISE_CONV_2D") {
      TfLiteDepthwiseConvParams* data = reinterpret_cast<TfLiteDepthwiseConvParams*>(builtin_data_);
    return tflite::openvinodelegate::get_activation_string(data->activation);
    } else if (name == "group" && op_type_ == "DEPTHWISE_CONV_2D") {
        TfLiteDepthwiseConvParams* data = reinterpret_cast<TfLiteDepthwiseConvParams*>(builtin_data_);
        return data->depth_multiplier;
    } else if (name == "align_corners" && op_type_ == "RESIZE_BILINEAR") {
        TfLiteResizeBilinearParams* data = reinterpret_cast<TfLiteResizeBilinearParams*>(builtin_data_);
        return data->align_corners;
    } else if (name == "half_pixel_centers" && op_type_ == "RESIZE_BILINEAR") {
        TfLiteResizeBilinearParams* data = reinterpret_cast<TfLiteResizeBilinearParams*>(builtin_data_);
        return data->half_pixel_centers;
    } else if (name == "axis" && op_type_ == "CONCATENATION") {
         TfLiteConcatenationParams* data = reinterpret_cast<TfLiteConcatenationParams*>(builtin_data_);
        return static_cast<int64_t>(data->axis);
    }
         else if (name == "strides" && op_type_ == "AVERAGE_POOL_2D") {
    TfLitePoolParams* data = reinterpret_cast<TfLitePoolParams*>(builtin_data_);
        return std::vector<int64_t>{1,
                                    static_cast<int64_t>(data->stride_height),
                                    static_cast<int64_t>(data->stride_width),
                                    1};
    } else if (name == "padding" && op_type_ == "AVERAGE_POOL_2D") {
         TfLitePoolParams* data = reinterpret_cast<TfLitePoolParams*>(builtin_data_);
        return "SAME";
    } else if (name == "ksize" && op_type_ == "AVERAGE_POOL_2D") {
       TfLitePoolParams* data = reinterpret_cast<TfLitePoolParams*>(builtin_data_);
        return std::vector<int64_t>{1,
                                    static_cast<int64_t>(data->filter_height),
                                    static_cast<int64_t>(data->filter_width),
                                    1};
    } else if (name == "data_format" && op_type_ == "AVERAGE_POOL_2D") {
        return "NHWC";
    } else if (name == "activation" && op_type_ == "AVERAGE_POOL_2D") {
      TfLitePoolParams* data = reinterpret_cast<TfLitePoolParams*>(builtin_data_);
    return tflite::openvinodelegate::get_activation_string(data->activation);
    } else if (name == "beta" && op_type_ == "SOFTMAX") {
      TfLiteSoftmaxParams* data = reinterpret_cast<TfLiteSoftmaxParams*>(builtin_data_);
        return data->beta;
    }
   
  }

  void set_op_builtin_data(void* builtin_data) {
    builtin_data_ = builtin_data;
  }

  void* get_op_builtin_data() {
    return builtin_data_;
  }

  const std::string &get_op_type() const override { return op_type_; };
  const std::string &get_op_name() const override { return op_name_; };

  void get_input_node(size_t input_port_idx, std::string &producer_name,
                      std::string &producer_output_port_name,
                      size_t &producer_output_port_index) const override {
    return;
  };

 private:
  std::string op_type_;
  std::string op_name_;
  std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo> input_tensor_info_;
  std::vector<ov::frontend::tensorflow_lite::TensorMetaInfo>
      output_tensor_info_;
  void* builtin_data_;
};

class DelegateDecoderTensor
    : public ov::frontend::tensorflow_lite::DecoderBaseTensor {
 public:
  // Need to create TensorMetaInfo

  explicit DelegateDecoderTensor(
      ov::frontend::tensorflow_lite::TensorMetaInfo m_tensor_meta_info,
      int64_t input_index, int64_t output_index)
      : m_tensor_meta_info(m_tensor_meta_info),
        input_index_(input_index),
        output_index_(output_index){};
  ov::frontend::tensorflow_lite::TensorMetaInfo get_tensor_info()
      const override {
    ov::frontend::tensorflow_lite::TensorMetaInfo temp;
    return m_tensor_meta_info;
  }

  /// \brief Get input index for tensor
  int64_t get_input_idx() const override { return input_index_; }

  /// \brief Get output index for tensor
  int64_t get_output_idx() const override { return output_index_; }

  ov::Any get_attribute(const std::string &name) const override {
    TFLITE_LOG(ERROR) << "get_attribute not implemented\n";
  }

  /// \brief Get a number of inputs
  size_t get_input_size() const override {
    TFLITE_LOG(ERROR) << "get_input_size not implemented\n";
  }

  /// \brief Get a producer name and its output port index
  ///
  /// \param input_port_idx              Input port index by which data is
  /// consumed \param producer_name               A producer name \param
  /// producer_output_port_name   Output port name if exists \param
  /// producer_output_port_index  Output port index from which data is generated
  void get_input_node(size_t input_port_idx, std::string &producer_name,
                      std::string &producer_output_port_name,
                      size_t &producer_output_port_index) const override {
    TFLITE_LOG(ERROR) << "get_input_node not implemented\n";
  }

  /// \brief Get operation type
  const std::string &get_op_type() const override {
    TFLITE_LOG(ERROR) << "get_op_type not implemented\n";
  };

  /// \brief Get node name
  const std::string &get_op_name() const override {
    TFLITE_LOG(ERROR) << "get_op_name not implemented\n";
  };

 private:
  ov::frontend::tensorflow_lite::TensorMetaInfo m_tensor_meta_info;
  int64_t input_index_;
  int64_t output_index_;
};
