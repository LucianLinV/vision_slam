#ifndef CONFIG_HPP
#define CONFIG_HPP
#include <iostream>
#include <memory>
#include <opencv2/core.hpp>
#include <stdexcept>
#include <string>

class Config {

private:
  inline static std::shared_ptr<Config> config_ = nullptr;
  cv::FileStorage files_;

  Config() {}

public:
  ~Config() = default;

  static void setParameterFile(const std::string &filename) {
    if (config_ == nullptr) {
      config_ = std::shared_ptr<Config>(new Config);
    }

    config_->files_ = cv::FileStorage(filename.c_str(), cv::FileStorage::READ);
    if (config_->files_.isOpened() == false) {
      std::cerr << "parameter file " << filename << " does not exist."
                << std::endl;
      config_->files_.release();
      return;
    }
  }

  template <typename T> static T get(const std::string &key) {
    if (config_ == nullptr || !config_->files_.isOpened()) {
      throw std::runtime_error("parameter file has not been opened");
    }
    return T(Config::config_->files_[key]);
  }
};

#endif
