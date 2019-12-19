/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 2.0
//              Copyright (2014) Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Christian R. Trott (crtrott@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <iostream>
#include <sstream>

#include <Kokkos_Core.hpp>

struct Tag1 {};
struct Tag2 {};

//TaggedFunctor is just an example of what I want to use with CUDA graphs -
//it doesn't actually need to be executed to replicate the require() issue

template<typename ExecSpace, typename View>
struct TaggedFunctor
{
  KOKKOS_INLINE_FUNCTION void operator()(const Tag1&, const int i) const
  {
    v(i) = 1;
  }
  KOKKOS_INLINE_FUNCTION void operator()(const Tag2&, const int i) const
  {
    v(i) = i;
  }
  View v;
};

template<typename Device>
void demonstrate()
{
  using Kokkos::Experimental::require;
  using Exec = typename Device::execution_space;
  using Tag1Range = Kokkos::RangePolicy<Exec, Tag1>;
  using Tag2Range = Kokkos::RangePolicy<Exec, Tag2>;
  using Tag1Team = Kokkos::TeamPolicy<Exec, Tag1>;
  using Tag2Team = Kokkos::TeamPolicy<Exec, Tag2>;
  auto P = Kokkos::Experimental::WorkItemProperty::HintLightWeight;
  Exec execInstance;
  auto range1 = require(Tag1Range(execInstance, 0, 100), P);
  auto range2 = require(Tag2Range(execInstance, 0, 100), P);
  auto team1 = require(Tag1Team(execInstance, 25, 1), P);
  auto team2 = require(Tag2Team(execInstance, 25, 1), P);
}

//----------------------------------------------------------------------------
//----------------------------------------------------------------------------

int main(int argc, char** argv) {
  Kokkos::initialize();
#if defined(KOKKOS_ENABLE_SERIAL)
  {
    std::cout << "Kokkos::Serial" << std::endl;
    demonstrate<Kokkos::Serial>();
  }
#endif  // defined( KOKKOS_ENABLE_SERIAL )

#if defined(KOKKOS_ENABLE_THREADS)
  {
    std::cout << "Kokkos::Threads" << std::endl;
    demonstrate<Kokkos::Threads>();
  }
#endif

#if defined(KOKKOS_ENABLE_OPENMP)
  {
    std::cout << "Kokkos::OpenMP" << std::endl;
    demonstrate<Kokkos::OpenMP>();
  }
#endif

#if defined(KOKKOS_ENABLE_CUDA)
  {
    std::cout << "Kokkos::Cuda" << std::endl;
    demonstrate<Kokkos::Cuda>();
  }
#endif
  Kokkos::finalize();

  return 0;
}

