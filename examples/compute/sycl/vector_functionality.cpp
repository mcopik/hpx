///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#include <hpx/include/compute.hpp>
#include <hpx/include/partitioned_vector.hpp>
#include <hpx/include/parallel_copy.hpp>

#include <hpx/hpx_init.hpp>

#include <numeric>
#include <iostream>
#include <string>
#include <vector>

struct pfo
{
    void operator()(int & x) const
    {
        ++x;
        //int v = x;
        //x = ++v;
    }
    // otherwise result_of fails to match pfo against value_proxy &&
    void operator()(int && x) const
    {
        ++x;
        //int v = x;
        //x = ++v;
    }
};

int hpx_main(boost::program_options::variables_map& vm)
{
    unsigned int seed = (unsigned int)std::time(nullptr);
    if (vm.count("seed"))
        seed = vm["seed"].as<unsigned int>();

    std::cout << "using seed: " << seed << std::endl;
    std::srand(seed);

    int const N = 100;
    std::vector<int> host(N);
    std::vector<int> new_host(N);

    std::iota(host.begin(), host.end(), (std::rand() % 100));

    typedef hpx::compute::sycl::allocator<int> target_allocator;
    {
        hpx::compute::sycl::target target;
        target_allocator alloc(target);
        hpx::compute::vector<int, target_allocator> device(N, alloc);
        hpx::compute::vector<int, target_allocator> device2(N, alloc);
        std::cout << host[0] << " " << host[1] << std::endl;
        std::cout << "Write at 0 val " << host[0] << "\n";
        device[0] = host[0];
        std::cout << "Read at 0 val " << device[0] << "\n";

        hpx::parallel::copy(hpx::parallel::seq, host.begin(), host.end(), device.begin());
        hpx::parallel::copy(hpx::parallel::seq, device.begin(), device.end(), new_host.begin());
        if (!std::equal(host.begin(), host.end(), new_host.begin())) {
            std::cout << "Wrong copy - not equal!" << std::endl;
        }

        hpx::compute::sycl::detail::launch<class NewKernel>(target, N, 32,
                                                         [=](size_t idx, cl::sycl::global_ptr<int> ptr) mutable {
                                                             if (idx >= N)
                                                                 return;
                                                             ptr[idx] += 1;
                                                         }, device.data());
        hpx::future<void> f = target.get_future();
        f.get();
        hpx::compute::sycl::default_executor exec(target);
        hpx::parallel::for_each(hpx::parallel::par.on(exec), device.begin(), device.end(), pfo());
        hpx::parallel::for_each(hpx::parallel::par, host.begin(), host.end(), [](int & val) { val += 2;} );
        target.synchronize();
        hpx::parallel::copy(hpx::parallel::seq, device.begin(), device.end(), new_host.begin());
        target.synchronize();
        if (!std::equal(host.begin(), host.end(), new_host.begin())) {
            std::cout << "Wrong copy - not equal!" << std::endl;
        }
    }
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
