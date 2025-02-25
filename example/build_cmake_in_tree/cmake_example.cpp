#include <Kokkos_Core.hpp>
#include <iostream>

using Scalar = double;

struct SegReduce
{
  // we call the argument lhs because in a scan, it's always a previous intermediate
  // value updating the current value.
  KOKKOS_INLINE_FUNCTION SegReduce& operator+=(const SegReduce& lhs)
  {
    if(!flag) {
      val += lhs.val;
    }
    flag = flag || lhs.flag;
    return *this;
  }

  Scalar val = 0;
  bool flag = false;
};

using Exec = Kokkos::DefaultExecutionSpace;
using Mem = typename Exec::memory_space;

int main(int argc, char* argv[]) {
  Kokkos::initialize(argc, argv);
  {
    int seglen = 5;
    int nsegs = 4;
    int n = seglen * nsegs;
    std::vector<double> vals(n);
    for(int i = 0; i < n; i++)
      vals[i] = (rand() % 100) * 0.01;
    std::vector<double> gold;
    double accum = 0;
    for(int i = 0; i < n; i++) {
      // reset accumulator at the start of each segment
      if(i % seglen == 0)
        accum = 0;
      // inclusive, so add value at i before writing out result
      accum += vals[i];
      gold.push_back(accum);
    }
    std::cout << "gold:   ";
    for(int i = 0; i < n; i++)
      std::cout << gold[i] << " ";
    std::cout << '\n';

    Kokkos::View<Scalar*, Mem> input("input", n);
    auto inputHost = Kokkos::create_mirror(input);
    for(int i = 0; i < n; i++) inputHost(i) = vals[i];
    Kokkos::deep_copy(input, inputHost);
    Kokkos::View<Scalar*, Mem> result("result", n);

    Kokkos::parallel_scan(Kokkos::RangePolicy<>(0, n),
      KOKKOS_LAMBDA(int i, SegReduce& update, bool finalPass)
      {
        // Mark the beginning of each segment
        if(i % seglen == 0) {
          update.val = 0;
          update.flag = true;
        }
        update.val += input(i);
        if(finalPass) result(i) = update.val;
      });

    auto resultHost = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), result);
    std::cout << "actual: ";
    for(int i = 0; i < n; i++)
      std::cout << resultHost(i) << ' ';
    std::cout << '\n';
  }
  Kokkos::finalize();
  return 0;
}
