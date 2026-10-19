#include "model/Qwen4PlePrefetch.hpp"

#include <chrono>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <memory>
#include <thread>

using namespace std::chrono_literals;
using splash::model::Qwen4PlePrefetchCompletion;
using splash::model::Qwen4PlePrefetchSubmission;

namespace {

void require(bool condition, const char *message) {
  if (condition) return;
  std::cerr << message << std::endl;
  std::_Exit(1); // A failed timeout must not wait for an async future's destructor.
}

template <class T> void ready(std::future<T> &future) {
  require(future.wait_for(2s) == std::future_status::ready, "PLE prefetch timed out");
  future.get();
}

// CPU stand-in for the shared event: retain a listener even when submission
// fails, then deliver it again as a later, higher event value would.
struct Event final {
  std::function<void()> listener;
  void signal() const { if (listener) listener(); }
};

struct Prefetch final {
  std::shared_ptr<Qwen4PlePrefetchCompletion> pending;
  std::future<void> prefetched;
  uint32_t &reads;
  bool produced = false;

  explicit Prefetch(uint32_t &reads) : reads(reads) {}
  ~Prefetch() { waitPlePrefetch(); }
  void waitPlePrefetch() { if (prefetched.valid()) prefetched.get(); }

  void arm(Event &event, std::function<void()> work = {}, bool failRegistration = false) {
    waitPlePrefetch();
    auto done = std::make_shared<Qwen4PlePrefetchCompletion>();
    prefetched = done->future();
    pending = done;
    if (failRegistration) throw std::runtime_error("injected listener registration failure");
    event.listener = [this, done, work = std::move(work)]() noexcept {
      done->run([&]() noexcept {
        require(produced, "read unproduced proposals");
        if (work) work();
        ++reads;
      });
    };
  }

  void submit(Event &event, uint32_t failAt, bool signalBeforeReturn = false) {
    Qwen4PlePrefetchSubmission submission(pending);
    arm(event, {}, failAt == 0);
    // Fail after arming at registration, graph construction, pipeline
    // preparation, command-buffer creation or submission, before any signal.
    for (uint32_t stage = 1; stage < 5; ++stage)
      if (stage == failAt) throw std::runtime_error("injected submission failure");
    produced = true;
    if (signalBeforeReturn) event.signal();
    submission.submitted();
  }
};

void failure(uint32_t stage, bool waitFirst) {
  Event event;
  uint32_t reads = 0;
  auto prefetch = std::make_unique<Prefetch>(reads);
  try {
    prefetch->submit(event, stage);
    require(false, "submission did not fail");
  } catch (const std::runtime_error &) {
  }
  if (waitFirst) {
    auto waited = std::async(std::launch::async, [&] { prefetch->waitPlePrefetch(); });
    ready(waited);
  }
  auto destroyed = std::async(std::launch::async, [owner = std::move(prefetch)]() mutable { owner.reset(); });
  ready(destroyed);
  // The owner and its proposal storage are gone. A cancelled callback must
  // neither access them nor fulfill its already completed promise again.
  event.signal();
  event.signal();
  require(reads == 0, "cancelled listener accessed the owner");
}

void drainRunning() {
  Event event;
  uint32_t reads = 0;
  auto prefetch = std::make_unique<Prefetch>(reads);
  std::promise<void> entered, release;
  auto started = entered.get_future();
  auto released = release.get_future();
  std::future<void> callback;
  {
    Qwen4PlePrefetchSubmission submission(prefetch->pending);
    prefetch->arm(event, [&] { entered.set_value(); released.wait(); });
    prefetch->produced = true;
    callback = std::async(std::launch::async, [&] { event.signal(); });
    ready(started);
    // Submission fails after the callback has claimed its work.
  }
  std::promise<void> destroying;
  auto destroyingStarted = destroying.get_future();
  auto destroyed = std::async(std::launch::async, [owner = std::move(prefetch), &destroying]() mutable {
    destroying.set_value();
    owner.reset();
  });
  ready(destroyingStarted);
  require(destroyed.wait_for(30ms) == std::future_status::timeout, "teardown did not drain a running listener");
  release.set_value();
  ready(callback);
  ready(destroyed);
  event.signal();
  require(reads == 1, "running listener completed more than once");
}

void rearm() {
  Event event;
  uint32_t reads = 0;
  Prefetch prefetch(reads);
  try {
    prefetch.submit(event, 4);
  } catch (const std::runtime_error &) {
  }
  auto cancelled = event.listener;
  auto submitted = std::async(std::launch::async, [&] { prefetch.submit(event, 5); });
  ready(submitted); // The next arm drains the cancelled prefetch without a signal.
  cancelled();
  event.signal();
  prefetch.waitPlePrefetch();
  require(reads == 1, "a later signal ran the cancelled prefetch");
}

void race() {
  for (uint32_t iteration = 0; iteration < 256; ++iteration) {
    auto done = std::make_shared<Qwen4PlePrefetchCompletion>();
    auto finished = done->future();
    uint32_t reads = 0;
    std::thread callback([&] { done->run([&]() noexcept { ++reads; }); });
    done->cancel();
    ready(finished);
    callback.join();
    done->cancel();
    done->run([&]() noexcept { ++reads; });
    require(reads <= 1, "completion race ran the listener twice");
  }
}

} // namespace

int main() {
  for (uint32_t stage = 0; stage < 5; ++stage)
    for (bool waitFirst : {false, true}) failure(stage, waitFirst);
  for (bool signalBeforeReturn : {false, true}) {
    Event event;
    uint32_t reads = 0;
    {
      Prefetch prefetch(reads);
      prefetch.submit(event, 5, signalBeforeReturn);
      event.signal();
      event.signal();
      prefetch.waitPlePrefetch();
    }
    event.signal();
    require(reads == 1, "successful listener did not run exactly once");
  }
  drainRunning();
  rearm();
  race();
  std::cout << "PLE prefetch cancellation: submission failures, delayed signals, drain and races passed\n";
}
