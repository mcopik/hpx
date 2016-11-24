///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/include/compute.hpp>

#include <hpx/hpx_init.hpp>

#include <numeric>
#include <iostream>
#include <string>
#include <vector>

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    int const N = 100;
    std::vector<int> host(N);

    std::iota(host.begin(), host.end(), (std::rand() % 100));

    typedef hpx::compute::sycl::allocator<int> allocator_type;
    hpx::compute::sycl::target target;
    allocator_type alloc(target);
    hpx::compute::vector<int, allocator_type> device(N, alloc);
    device[0] = 1;
    std::cout << device[0] << std::endl;
    return hpx::finalize();
}

int main(int argc, char* argv[])
{
    // add command line option which controls the random number generator seed
    using namespace boost::program_options;
    options_description desc_commandline(
        "Usage: " HPX_APPLICATION_STRING " [options]");

    desc_commandline.add_options()
        ("seed,s", value<unsigned int>(),
        "the random number generator seed to use for this run")
        ;

    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
