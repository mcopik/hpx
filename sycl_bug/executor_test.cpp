#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/parallel/executors/parallel_executor.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/parallel/kernel.hpp>
#include <hpx/parallel/executors/kernel_name.hpp>

#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>



int hpx_main(boost::program_options::variables_map& vm)
{

	boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();
	{
		hpx::util::high_resolution_timer t;
		std::vector<std::size_t> c(n);
		std::vector<std::size_t> d(n);
		std::size_t count = 0;

		/**
		 * First, last
		 */
		std::iota(boost::begin(c), boost::end(c), std::rand());
		std::iota(boost::begin(d), boost::end(d), std::rand());
		
		/**
			Default
		**/
		hpx::parallel::for_each(hpx::parallel::gpu,
			boost::begin(c), boost::end(c),
			[](std::size_t& v) {

				v = 400;
			});

		// verify values
		std::for_each(boost::begin(c), boost::end(c),
			[&count](std::size_t v) -> void {
				HPX_TEST_EQ(v, std::size_t(400));
				++count;
			});
		HPX_TEST_EQ(count, c.size());

		count = 0;
		std::iota(boost::begin(c), boost::end(c), std::rand());		
		/**
			Kernel overrides parameter
		**/
		hpx::parallel::for_each(hpx::parallel::gpu.with(hpx::parallel::static_chunk_size(4), hpx::parallel::kernel_name<class FalseName_2>()),
			boost::begin(c), boost::end(c),
			hpx::parallel::make_kernel<class CorrectName_2>([](std::size_t& v) {

				v = 401;
			}));
		std::for_each(boost::begin(c), boost::end(c),
			[&count](std::size_t v) -> void {
				HPX_TEST_EQ(v, std::size_t(401));
				++count;
			});
		HPX_TEST_EQ(count, c.size());


		/**
		 * Second attempt
		 */
		std::iota(boost::begin(c), boost::end(c), std::rand());
		std::iota(boost::begin(d), boost::end(d), std::rand());
		std::vector< hpx::future<std::vector<std::size_t>::iterator> > tasks;
		/**
			Just parameter
		**/
		tasks.push_back( hpx::parallel::for_each_n(hpx::parallel::gpu(hpx::parallel::task).with(hpx::parallel::static_chunk_size(4), hpx::parallel::kernel_name<class CorrectName_3>()),
			boost::begin(c), n,
			[](std::size_t& v) {
				v = 42;
			} ) );
		/**
			Just kernel
		**/
		tasks.push_back( hpx::parallel::for_each_n(hpx::parallel::gpu(hpx::parallel::task),
			boost::begin(d), n,
			hpx::parallel::make_kernel<class CorrectName_4>([](std::size_t& v) {
				v = 43;
			})) );

		hpx::wait_all(tasks);
		// verify values
		count = 0;
		std::for_each(boost::begin(c), boost::end(c),
			[&count](std::size_t v) -> void {
				HPX_TEST_EQ(v, std::size_t(42));
				++count;
			});
		HPX_TEST_EQ(count, c.size());

		count = 0;
		std::for_each(boost::begin(d), boost::end(d),
			[&count](std::size_t v) -> void {
				HPX_TEST_EQ(v, std::size_t(43));
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
