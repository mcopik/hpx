/*
 * executor_test.cpp
 *
 *  Created on: Jul 6, 2015
 *      Author: mcopik
 */

#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/parallel/algorithms/transform.hpp>
#include <hpx/parallel/executors/parallel_executor.hpp>
#include <hpx/util/lightweight_test.hpp>


#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>



int hpx_main(boost::program_options::variables_map& vm)
{


	boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();
	{
		hpx::util::high_resolution_timer t;

		std::vector<std::size_t> a(n);
		std::vector<std::size_t> b(n);
		std::vector<std::size_t> c(n);
		std::iota(boost::begin(a), boost::end(a),  std::rand());
		std::iota(boost::begin(b), boost::end(b),  std::rand());
		std::iota(boost::begin(c), boost::end(c),  std::rand());

	    auto buffera = hpx::parallel::gpu.executor().create_buffers(a.begin(), n);
	    auto bufferb = hpx::parallel::gpu.executor().create_buffers(b.begin(), n);
	    auto bufferc = hpx::parallel::gpu.executor().create_buffers(c.begin(), n);
		/**
		 * First, last
		 */
		hpx::parallel::for_each(hpx::parallel::gpu,//hpx::parallel::gpu,
			buffera.begin(), buffera.end(),
			[](std::size_t& v) {

				v = 400;
			});
		int k = 3;
		hpx::parallel::for_each(hpx::parallel::gpu.with(hpx::parallel::static_chunk_size(5)),
			bufferb.begin(), bufferb.end(),
			[](std::size_t& v) {
				//printf("%lu \n", v); 
				v = 43;
			});

		hpx::parallel::transform(hpx::parallel::gpu.with(hpx::parallel::static_chunk_size(32)),
                        buffera.begin(), buffera.end(), bufferb.begin(), bufferc.begin(),
                        [=](std::size_t v1, std::size_t v2) {
                                //printf("%lu \n", v);
                                return v1 + v2 *k;
                        });
		buffera.sync();
        bufferb.sync();
        bufferc.sync();

		// verify values
		std::size_t count = 0;
		std::for_each(boost::begin(a), boost::end(a),
			[&count](std::size_t v) -> void {
				HPX_TEST_EQ(v, std::size_t(400));
				++count;
			});
		HPX_TEST_EQ(count, a.size());

		count = 0;
		std::for_each(boost::begin(b), boost::end(b),
			[&count](std::size_t v) -> void {
				HPX_TEST_EQ(v, std::size_t(43));
				++count;
			});
		HPX_TEST_EQ(count, b.size());
		count = 0;
		std::for_each(boost::begin(c), boost::end(c),
				[&count](std::size_t v) -> void {
						HPX_TEST_EQ(v, std::size_t(400+43+300));
						++count;
				});
		HPX_TEST_EQ(count, c.size());

		double elapsed = t.elapsed();
		std::cout
			<< ( boost::format("elapsed time == %2% [s]\n")
			   % n  % elapsed);
	}

	return hpx::finalize();
}


int main(int argc, char* argv[])
{
	using boost::program_options::options_description;
	using boost::program_options::value;

	// Configure application-specific options
	options_description
	   desc_commandline("Usage: " HPX_APPLICATION_STRING " [options]");

	desc_commandline.add_options()
		( "n-value"
		, value<boost::uint64_t>()->default_value(10)
		, "n value for the factorial function")
		;

	// Initialize and run HPX
	return hpx::init(desc_commandline, argc, argv);
}
