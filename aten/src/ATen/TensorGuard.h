#pragma once

#include <ATen/ATen.h>
#include <ATen/ScalarType.h>
#include <ATen/Tensor.h>

#include <cstddef>
#include <vector>

namespace at {

struct TensorGuard {
  TensorGuard() = default;

  explicit TensorGuard(const Tensor& tensor) {
    if (tensor.has_storage()) {
      StorageImpl* storage = tensor.storage().unsafeGetStorageImpl();
      if (storage->lms_enabled()) {
        storage->lms_pin();
        storage_ = storage;
      }
    }
  }

  ~TensorGuard() {
    if (storage_ != nullptr)
      storage_->lms_unpin();
  }

 private:
  StorageImpl* storage_ = nullptr;
};

struct TensorListGuard {
  TensorListGuard() = default;

  explicit TensorListGuard(const TensorList& tensors) {
    int len = tensors.size();
    for (int i = 0; i < len; i++) {
      const Tensor &tensor = tensors[i];
      if (tensor.has_storage()) {
        StorageImpl* storage = tensor.storage().unsafeGetStorageImpl();
        if (storage->lms_enabled()) {
          storage->lms_pin();
          storage_.push_back(storage);
        }
      }
    }
  }

  ~TensorListGuard() {
    for (auto storage : storage_) {
      storage->lms_unpin();
    }
  }

 private:
  std::vector<StorageImpl*> storage_;
};

} // namespace at
