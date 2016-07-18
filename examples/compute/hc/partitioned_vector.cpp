//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/hpx_init.hpp>

#include <hpx/include/compute.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/compute/hc/detail/launch.hpp>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
// Define the partitioned vector types to be used.
typedef hpx::compute::hc::allocator<int> allocator_type;
typedef typename allocator_type::pointer pointer;
typedef hpx::compute::vector<int, allocator_type> target_vector;

HPX_REGISTER_PARTITIONED_VECTOR(int, target_vector);

///////////////////////////////////////////////////////////////////////////////
int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(0);
    unsigned int n = 100;
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();
    if (vm.count("n"))
        n = vm["n"].as<unsigned int>();
    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    hpx::compute::hc::target target;
    allocator_type alloc(target);
    target_vector d_A(n, alloc);

    std::cout << "Write: 1" << std::endl;
    d_A[0] = 1;
    std::cout << "Read: " << d_A[0] << std::endl;

    int x = 5;
    std::cout << "Write in kernel: 5" << std::endl;
    hpx::compute::hc::detail::launch(target, n/2, 2,
        [] (hpx::compute::hc::local_index<1> idx,
            const hpx::compute::hc::target_ptr<int> & p,
            int const & x) [[hc]]
        {
#if defined(__COMPUTE__ACCELERATOR__)
            p[ idx.global[0] ] = x;
#endif
            //p[ idx.global[0] ] = 1;
        },
        d_A.data(), x);
    target.synchronize();
    int read_val = d_A[0];
    std::cout << "Read : " << read_val << std::endl;

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

    desc_commandline.add_options()
        ("n", value<unsigned int>(),
        "data size")
        ;
    // Initialize and run HPX
    return hpx::init(desc_commandline, argc, argv);
}
