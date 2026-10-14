#include "metal/CommandGraph.hpp"
#include "TestChecks.hpp"

#include <iostream>

using namespace splash::metal;
using splash::test::require;
using splash::test::rejects;

int main() {
  CommandGraph graph;
  for (int i = 0; i < 6; ++i) graph.add("unused", {}, {1, 1, 1});
  graph.independent(1, 4);
  for (size_t i = 0; i < 6; ++i)
    require(graph.dispatches()[i].independentOfPrevious == (i == 2 || i == 3),
            "independent group lost its entry or exit boundary");
  rejects([&] { graph.independent(0, 7); }, "independent dispatch range", "range past dispatches accepted");
  rejects([&] { graph.independent(2, 2); }, "independent dispatch range", "empty range accepted");
  rejects([&] { graph.independent(3, 5); }, "independent dispatch range", "overlapping range accepted");
  rejects([&] { graph.independent(0, 2); }, "independent dispatch range", "range cutting a group accepted");
  graph.independent(4, 6);
  require(!graph.dispatches()[4].independentOfPrevious, "adjacent groups merged");
  CommandGraph events;
  events.add("unused", {}, {1, 1, 1});
  events.signal({}, 1);
  events.add("unused", {}, {1, 1, 1});
  rejects([&] { events.independent(0, 2); }, "independent dispatch range", "range crossed an event");
  require(!events.dispatches()[1].independentOfPrevious, "failed range mutated graph");
  std::cout << "PASS command graph ordering (CPU only)\n";
}
