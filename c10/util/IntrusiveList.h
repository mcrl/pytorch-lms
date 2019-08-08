//===--- IntrusiveList.h - --------------------------------------*- C++ -*-===//

#pragma once

#include "c10/util/Exception.h"

namespace c10 {

// Element objects embed the IntrustiveListHook, which provides the
// following properties:
//   1. Insertion and removal operations are O(1) and require no
//      memory allocation or deletion.
//   2. Element destruction is valid and can be performed safely
//      regardless of list membership.
template <typename T>
class IntrusiveListHook {
 public:
  IntrusiveListHook(T *elem) : elem_(elem) {
    next_ = prev_ = this;
  }
  ~IntrusiveListHook() {
    remove();
  }

  bool attached() const { return next_ != this; }
  bool detached() const { return next_ == this; }

  void insertbefore(IntrusiveListHook<T>* x) {
    if (x->attached()) {
      AT_ERROR("Double insertion of IntrusiveListHook");
    }
    x->prev_ = prev_;
    x->next_ = this;
    prev_->next_ = x;
    prev_ = x;
  }

  bool remove() {
    if (!attached()) return false;

    prev_->next_ = next_;
    next_->prev_ = prev_;
    next_ = prev_ = this;
    return true;
  }
  IntrusiveListHook<T>* next() const { return next_; }
  IntrusiveListHook<T>* prev() const { return prev_; }
  T* elem() const { return elem_; }

 private:
  IntrusiveListHook<T>* next_;
  IntrusiveListHook<T>* prev_;
  T* elem_;
};

template <typename T>
class IntrusiveList {
 public:
  IntrusiveList() : anchor_(nullptr) {}
  ~IntrusiveList() {}
  bool empty() const { return anchor_.detached(); }
  void append(IntrusiveListHook<T>* x) { anchor_.insertbefore(x); }
  void prepend(IntrusiveListHook<T>* x) { anchor_.next()->insertbefore(x); }
  IntrusiveListHook<T>* head() const { return anchor_.next(); }
  IntrusiveListHook<T>* tail() const { return anchor_.prev(); }
  const IntrusiveListHook<T>* terminator() const { return &anchor_; }

 private:
  IntrusiveListHook<T> anchor_;
};

} // end namespace c10
