#pragma once

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <future>
#include <memory>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <vector>

namespace splash::model {

// A listener may outlive a step that failed before submitting its signal.
// Cancellation and the listener compete for Pending: only the winner may
// fulfill the promise. If the listener won, a waiter still drains its work.
class Qwen4PlePrefetchCompletion final {
public:
  std::future<void> future() { return done_.get_future(); }

  template <class Work> void run(Work &&work) noexcept {
    static_assert(std::is_nothrow_invocable_v<Work>);
    State expected = State::Pending;
    if (!state_.compare_exchange_strong(expected, State::Running)) return;
    std::forward<Work>(work)();
    done_.set_value();
  }

  void cancel() noexcept {
    State expected = State::Pending;
    if (state_.compare_exchange_strong(expected, State::Cancelled)) done_.set_value();
  }

private:
  enum class State : uint8_t { Pending, Running, Cancelled };
  std::atomic<State> state_{State::Pending};
  std::promise<void> done_;
};

// Covers graph construction, preparation and submission, including listener
// registration itself. Release only once the signal's submission succeeded.
class Qwen4PlePrefetchSubmission final {
public:
  explicit Qwen4PlePrefetchSubmission(std::shared_ptr<Qwen4PlePrefetchCompletion> &pending) : pending_(pending) {}
  Qwen4PlePrefetchSubmission(const Qwen4PlePrefetchSubmission &) = delete;
  Qwen4PlePrefetchSubmission &operator=(const Qwen4PlePrefetchSubmission &) = delete;
  ~Qwen4PlePrefetchSubmission() {
    if (pending_) pending_->cancel();
    submitted();
  }
  void submitted() noexcept { pending_.reset(); }

private:
  std::shared_ptr<Qwen4PlePrefetchCompletion> &pending_;
};

// The history captured for a draft's speculative PLE row hints, with the
// current anchor appended before the proposals arrive.
struct Qwen4PlePrefetchInput final {
  uint64_t position;
  std::vector<uint32_t> history;
};

inline Qwen4PlePrefetchInput qwen4PlePrefetchInput(uint64_t position, std::span<const uint32_t> history,
                                                uint32_t anchor, uint32_t reach) {
  if (position > history.size() || !reach)
    throw std::invalid_argument("invalid PLE prefetch history");
  const size_t end = static_cast<size_t>(position);
  const size_t begin = end - std::min<size_t>(end, reach);
  std::vector<uint32_t> suffix(history.begin() + begin, history.begin() + end);
  suffix.push_back(anchor);
  // A proposal reaches back at most ngram - 1 tokens. The anchor plus
  // up to ngram prior tokens cover that reach, so shifting both position
  // and history to this suffix preserves every hash, even at sequence start.
  const uint64_t localPosition = suffix.size();
  return {localPosition, std::move(suffix)};
}

} // namespace splash::model
