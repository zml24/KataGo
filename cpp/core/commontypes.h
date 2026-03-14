#ifndef COMMONTYPES_H
#define COMMONTYPES_H

#include <string>

struct enabled_t {
  enum value { False, True, Auto };
  value x;

  enabled_t() = default;
  constexpr enabled_t(value a) : x(a) { }
  explicit operator bool() = delete;
  constexpr bool operator==(enabled_t a) const { return x == a.x; }
  constexpr bool operator!=(enabled_t a) const { return x != a.x; }

  std::string toString() {
    return x == True ? "true" : x == False ? "false" : "auto";
  }

  static bool tryParse(const std::string& v, enabled_t& buf) {
    if(v == "1" || v == "t" || v == "true" || v == "enabled" || v == "y" || v == "yes")
      buf = True;
    else if(v == "0" || v == "f" || v == "false" || v == "disabled" || v == "n" || v == "no")
      buf = False;
    else if(v == "auto")
      buf = Auto;
    else
      return false;
    return true;
  }

};

struct compute_precision_t {
  enum value { FP32, FP16, BF16, Auto };
  value x;

  compute_precision_t() = default;
  constexpr compute_precision_t(value a) : x(a) { }
  explicit operator bool() = delete;
  constexpr bool operator==(compute_precision_t a) const { return x == a.x; }
  constexpr bool operator!=(compute_precision_t a) const { return x != a.x; }

  std::string toString() const {
    return x == FP32 ? "fp32" : x == FP16 ? "fp16" : x == BF16 ? "bf16" : "auto";
  }

  bool isExplicit() const {
    return x != Auto;
  }

  static bool tryParse(const std::string& v, compute_precision_t& buf) {
    if(v == "fp32" || v == "float32")
      buf = FP32;
    else if(v == "fp16" || v == "float16")
      buf = FP16;
    else if(v == "bf16" || v == "bfloat16")
      buf = BF16;
    else if(v == "auto")
      buf = Auto;
    else
      return false;
    return true;
  }
};

struct nn_model_type_t {
  enum value { CNN, TF };
  value x;

  constexpr nn_model_type_t() : x(CNN) { }
  constexpr nn_model_type_t(value a) : x(a) { }
  explicit operator bool() = delete;
  constexpr bool operator==(nn_model_type_t a) const { return x == a.x; }
  constexpr bool operator!=(nn_model_type_t a) const { return x != a.x; }

  std::string toString() const {
    return x == CNN ? "cnn" : "tf";
  }

  static bool tryParse(const std::string& v, nn_model_type_t& buf) {
    if(v == "cnn")
      buf = CNN;
    else if(v == "tf" || v == "transformer")
      buf = TF;
    else
      return false;
    return true;
  }
};

#endif //COMMONTYPES_H
