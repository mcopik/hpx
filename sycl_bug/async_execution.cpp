#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/parallel/executors/parallel_executor.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>

#include <SYCL/sycl.hpp>

using namespace cl::sycl;

int hpx_main(boost::program_options::variables_map& vm)
{

	hpx::util::high_resolution_timer t;
	boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();
	int * data = new int[n];
  	for (int i = 0; i < n; i++) {
		data[i] = 0;
  	}

	try
	{
		queue myQueue;
		buffer<int, 1> buf(data, range<1>(n));
		if(false) {

			myQueue.submit([&](handler& cgh) {

				auto ptr = buf.get_access<access::mode::read_write>(cgh);
				auto myKernel = (range<1>(n), [=](id<1> idx) {
					ptr[idx[0]] = static_cast<int>(idx[0]);
				});
				cgh.parallel_for<class assign_elements>(range<1>(n), myKernel);
			});

		} else {

			auto future_obj = hpx::async(hpx::launch::async, [&] ()
				{
					myQueue.submit([&](handler& cgh) {
						auto ptr = buf.get_access<access::mode::read_write>(cgh);
						auto myKernel = (range<1>(n), [=](id<1> idx) {
							ptr[idx[0]] = static_cast<int>(idx[0]);
						});
						cgh.parallel_for<class async_assign_elements>(range<1>(n), myKernel);
					});
				});
			future_obj.wait();
		}

	} catch (exception e) {
		std::cout << "SYCL exception caught: " << e.what();
		return 2;
	}
 
	for (int i = 0; i < n; i++) {
		if (data[i] != i) {
			std::cout << "The results not are correct at: " << i << std::endl;
		}
	}

	double elapsed = t.elapsed();
	std::cout << ( boost::format("elapsed time == %2% [s]\n") % n  % elapsed);
	delete[] data;

	return hpx::finalize();
}


int main(int argc, char* argv[])
{
	using boost::program_options::options_description;
	using boost::program_options::value;
	using boost::program_options::variables_map;
	using boost::program_options::store;
	using boost::program_options::parse_command_line;

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
