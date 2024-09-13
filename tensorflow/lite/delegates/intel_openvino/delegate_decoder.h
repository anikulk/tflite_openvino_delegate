#include "openvino/frontend/tensorflow_lite/decoder.hpp"
#include "tensorflow/lite/tools/logging.h"

struct TensorMetaInfo {
  std::shared_ptr<ov::frontend::tensorflow_lite::QuantizationInfo>
      m_quantization_info;
  std::shared_ptr<ov::frontend::tensorflow_lite::SparsityInfo> m_sparsity_info;
  ov::PartialShape m_partial_shape;  // Can be set up
  ov::element::Type m_element_type;  // Can be set up
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
          output_tensor_info) {
    op_type_ = type;
    op_name_ = name;
    input_tensor_info_ = input_tensor_info;
    output_tensor_info_ = output_tensor_info;
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
  size_t get_output_size() const override { return input_tensor_info_.size(); }

  /// \brief Get output tensor name by index
  std::string get_output_tensor_name(size_t idx) const override {
    return output_tensor_info_[idx].m_tensor_name;
  }

  /// \brief Get output tensor type by index
  ov::element::Type get_output_tensor_type(size_t idx) const override {
    return output_tensor_info_[idx].m_element_type;
  }
  ov::Any get_attribute(const std::string &name) const override {
    std::cout << "Not implemented";
  };

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
