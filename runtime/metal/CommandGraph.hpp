#pragma once

#include "MetalBackend.hpp"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <deque>
#include <span>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace splash::metal {

// An ordered dispatch list for one command, with the event steps that order
// it against another agent (EventStep); MetalBackend splits a command into
// Metal command buffers at its event signals. Buffers bind at indices
// 0..n-1; an optional parameter struct binds at index n and is copied into
// graph-owned storage until submission.
class CommandGraph final {
public:
  static constexpr uint32_t kDefaultThreads = 256;

  CommandGraph() = default;
  // Dispatches point into payloads_; a copy would keep pointing at the source.
  CommandGraph(const CommandGraph &) = delete;
  CommandGraph &operator=(const CommandGraph &) = delete;
  CommandGraph(CommandGraph &&) noexcept = default;
  CommandGraph &operator=(CommandGraph &&) noexcept = default;

  void add(std::string pipeline, std::vector<MetalBuffer> buffers,
           DispatchSize groups, DispatchSize threads = {kDefaultThreads, 1, 1}) {
    push(std::move(pipeline), std::move(buffers), groups, threads);
  }

  template <class Params>
  void add(std::string pipeline, std::vector<MetalBuffer> buffers,
           const Params &params, DispatchSize groups,
           DispatchSize threads = {kDefaultThreads, 1, 1}) {
    static_assert(std::is_trivially_copyable_v<Params>,
                  "dispatch parameters must be plain data");
    payloads_.emplace_back(sizeof(Params));
    std::memcpy(payloads_.back().data(), &params, sizeof(Params));
    ComputeDispatch &dispatch =
        push(std::move(pipeline), std::move(buffers), groups, threads);
    dispatch.bytes.push_back({static_cast<uint32_t>(dispatch.buffers.size()),
                              payloads_.back().data(), sizeof(Params)});
  }

  // A dispatch of the grid in `grid`, which an earlier dispatch writes
  // within `groups` (ComputeDispatch::indirectThreadgroups).
  template <class Params>
  void add(std::string pipeline, std::vector<MetalBuffer> buffers,
           const Params &params, DispatchSize groups, IndirectGrid grid,
           DispatchSize threads) {
    add(std::move(pipeline), std::move(buffers), params, groups, threads);
    dispatches_.back().indirectThreadgroups = std::move(grid);
  }

  // A dispatch whose `params` follow its n buffers, then `after` from index
  // n + 1 and `tail` after them: a kernel whose parameter struct precedes
  // further bindings.
  template <class Params, class Tail>
  void add(std::string pipeline, std::vector<MetalBuffer> buffers, const Params &params,
           std::vector<MetalBuffer> after, const Tail &tail, DispatchSize groups,
           DispatchSize threads = {kDefaultThreads, 1, 1}) {
    static_assert(std::is_trivially_copyable_v<Params> && std::is_trivially_copyable_v<Tail>,
                  "dispatch parameters must be plain data");
    add(std::move(pipeline), std::move(buffers), params, groups, threads);
    ComputeDispatch &dispatch = dispatches_.back();
    uint32_t index = static_cast<uint32_t>(dispatch.buffers.size()) + 1;
    for (MetalBuffer &buffer : after) dispatch.buffers.push_back({index++, std::move(buffer)});
    payloads_.emplace_back(sizeof(Tail));
    std::memcpy(payloads_.back().data(), &tail, sizeof(Tail));
    dispatch.bytes.push_back({index, payloads_.back().data(), sizeof(Tail)});
  }

  // Event steps after the dispatches added so far (EventStep): signal once
  // all earlier work has completed, or hold all later work until the event
  // reaches value.
  void signal(SharedEvent event, uint64_t value) {
    step(std::move(event), value, EventStep::Kind::Signal);
  }
  void wait(SharedEvent event, uint64_t value) {
    step(std::move(event), value, EventStep::Kind::Wait);
  }

  // The caller proves that [begin, end) has no RAW, WAR or WAW dependency,
  // including aliased views, scratch and buffers reached through pointers.
  // All earlier work completes before the group; all its work completes
  // before the next ordinary dispatch. No access-mode inference is made.
  void independent(size_t begin, size_t end) {
    if (begin >= end || end > dispatches_.size())
      throw std::invalid_argument("invalid independent dispatch range");
    for (const EventStep &event : events_)
      if (event.before > begin && event.before < end)
        throw std::invalid_argument("independent dispatch range crosses an event");
    for (size_t i = begin; i < end; ++i)
      if (dispatches_[i].indirectThreadgroups.buffer || dispatches_[i].independentOfPrevious)
        throw std::invalid_argument("independent dispatch range overlaps a group or indirect grid");
    if (end < dispatches_.size() && dispatches_[end].independentOfPrevious)
      throw std::invalid_argument("independent dispatch range cuts a group");
    for (size_t i = begin + 1; i < end; ++i)
      dispatches_[i].independentOfPrevious = true;
  }

  // Safe command boundaries, recorded only between complete layers.
  void layerBoundary() {
    if (!dispatches_.empty() && (layers_.empty() || layers_.back() != dispatches_.size()))
      layers_.push_back(dispatches_.size());
  }
  [[nodiscard]] std::vector<Command> layerCommands(uint32_t count) const {
    if (count < 2 || layers_.size() < 2 || !events_.empty()) return {command()};
    count = std::min<uint32_t>(count, layers_.size());
    std::vector<Command> result;
    size_t begin = 0;
    for (uint32_t i = 1; i < count; ++i) {
      const size_t end = layers_[i * layers_.size() / count - 1];
      if (end < dispatches_.size() && dispatches_[end].independentOfPrevious)
        throw std::logic_error("layer boundary crosses an independent group");
      result.push_back({dispatches().subspan(begin, end - begin), {}});
      begin = end;
    }
    result.push_back({dispatches().subspan(begin), {}});
    return result;
  }

  [[nodiscard]] bool empty() const noexcept { return dispatches_.empty(); }
  // The dispatches alone; command() has the event steps too.
  [[nodiscard]] std::span<const ComputeDispatch> dispatches() const noexcept {
    return dispatches_;
  }
  [[nodiscard]] Command command() const noexcept {
    return {dispatches_, events_};
  }

private:
  ComputeDispatch &push(std::string pipeline, std::vector<MetalBuffer> buffers,
                        DispatchSize groups, DispatchSize threads) {
    ComputeDispatch dispatch;
    dispatch.pipelineName = std::move(pipeline);
    dispatch.threadgroups = groups;
    dispatch.threadsPerThreadgroup = threads;
    dispatch.buffers.reserve(buffers.size());
    for (uint32_t index = 0; index < buffers.size(); ++index) {
      dispatch.buffers.push_back({index, std::move(buffers[index])});
    }
    dispatches_.push_back(std::move(dispatch));
    return dispatches_.back();
  }

  void step(SharedEvent event, uint64_t value, EventStep::Kind kind) {
    events_.push_back({dispatches_.size(), std::move(event), value, kind});
  }

  std::deque<std::vector<std::byte>> payloads_;
  std::vector<ComputeDispatch> dispatches_;
  std::vector<EventStep> events_;
  std::vector<size_t> layers_;
};

} // namespace splash::metal
