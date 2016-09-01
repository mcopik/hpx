#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/parallel/executors/parallel_executor.hpp>
//#include <hpx/parallel/algorithms/transform.hpp>
#include <hpx/util/lightweight_test.hpp>
#include <hpx/parallel/kernel.hpp>
#include <hpx/parallel/executors/kernel_name.hpp>
#include <hpx/parallel/algorithms/transform.hpp>
#include <hpx/parallel/algorithms/copy.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <iostream>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>



int hpx_main(boost::program_options::variables_map& vm)
{

	boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();
	{
		hpx::util::high_resolution_timer t;
//		std::vector<std::size_t> c(n);
	//	std::vector<std::size_t> d(n);
		//std::vector<std::size_t> e(n);

        
		std::vector<float> a(n);
		std::vector<float> b(n);
		std::vector<float> c_(n);
		std::vector<float> a_init(n);
		std::vector<float> c_init(n);
=======
		std::vector<std::size_t> c(n);
		std::vector<std::size_t> d(n);
		std::vector<std::size_t> e(n);
>>>>>>> 86c71300dd768238d3d6a908ef3641a7a6f932dc
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
<<<<<<< HEAD
			hpx::parallel::make_kernel<class ExecutorTest>([](std::size_t& v) {
=======
			hpx::parallel::make_kernel<class CorrectName_1>([](std::size_t& v) {
>>>>>>> 86c71300dd768238d3d6a908ef3641a7a6f932dc

				v = 400;
			}));

		// verify values
		std::for_each(boost::begin(c), boost::end(c),
			[&count](std::size_t v) -> void {
				HPX_TEST_EQ(v, std::size_t(400));
				++count;
			});
		HPX_TEST_EQ(count, c.size());

		count = 0;
<<<<<<< HEAD
		//std::iota(boost::begin(c), boost::end(c), std::rand());
=======
>>>>>>> 86c71300dd768238d3d6a908ef3641a7a6f932dc
		/**
			Kernel overrides parameter
		**/
		hpx::parallel::for_each(hpx::parallel::gpu.with(hpx::parallel::static_chunk_size(4), hpx::parallel::kernel_name<class FalseName_2>()),
			boost::begin(d), boost::end(d),
			hpx::parallel::make_kernel<class CorrectName_2>([](std::size_t& v) {

				v = 401;
			}));
<<<<<<< HEAD
=======
		hpx::parallel::transform(hpx::parallel::gpu.with(hpx::parallel::static_chunk_size(32), hpx::parallel::kernel_name<class TransformKernel>()),
                        boost::begin(c), boost::end(c), boost::begin(d), boost::begin(e),
                        [](std::size_t v1, std::size_t v2) {
                                //printf("%lu \n", v);
                                return v1 + v2 + 300;
                        });
>>>>>>> 86c71300dd768238d3d6a908ef3641a7a6f932dc


		std::for_each(boost::begin(d), boost::end(d),
			[&count](std::size_t v) -> void {
				HPX_TEST_EQ(v, std::size_t(401));
				++count;
			});
		HPX_TEST_EQ(count, c.size());
        count = 0;
		std::for_each(boost::begin(e), boost::end(e),
			[&count](std::size_t v) -> void {
				HPX_TEST_EQ(v, std::size_t(401 + 400 + 300));
				++count;
			});
		HPX_TEST_EQ(count, c.size());


		int k = 3;
        {
        
/*		    std::iota(boost::begin(a), boost::end(a), std::rand());
		    std::iota(boost::begin(c_), boost::end(c_), std::rand());*/


            std::fill(boost::begin(a), boost::end(a), 1.0);
:q            std::fill(boost::begin(b), boost::end(b), 2.0);
            std::fill(boost::begin(c_), boost::end(c_), 0.0);



		    std::copy(boost::begin(a), boost::end(a), boost::begin(a_init));//std::rand());
		    std::copy(boost::begin(a), boost::end(a), boost::begin(c_init));//std::rand());
            auto buffera = hpx::parallel::gpu.executor().create_buffers(a.begin(), n);
            auto bufferb = hpx::parallel::gpu.executor().create_buffers(b.begin(), n);
            auto bufferc = hpx::parallel::gpu.executor().create_buffers(c_.begin(), n);


		    hpx::util::high_resolution_timer t;
		    hpx::parallel::copy(hpx::parallel::gpu.with(hpx::parallel::static_chunk_size(32), hpx::parallel::kernel_name<class Copy>()),
                            n, buffera, bufferc);

		    hpx::parallel::transform(hpx::parallel::gpu.with(hpx::parallel::static_chunk_size(32), hpx::parallel::kernel_name<class Scale>()),
                            n, bufferc, bufferb,
                            [=](float v1) {
                                    //printf("%lu \n", v);
                                    //return v1 + k*v2;
                                    return k * v1;
                            });

		    hpx::parallel::transform(hpx::parallel::gpu.with(hpx::parallel::static_chunk_size(32), hpx::parallel::kernel_name<class Add>()),
                            n, buffera, bufferb, bufferc,
                            [=](float v1, float v2) {
                                    //printf("%lu \n", v);
                                    //return v1 + k*v2;
                                    return v2 + v1;
                            });

		    hpx::parallel::transform(hpx::parallel::gpu.with(hpx::parallel::static_chunk_size(32), hpx::parallel::kernel_name<class Triad>()),
                            n, bufferb, bufferc, buffera,
                            [=](float v1, float v2) {
                                    //printf("%lu \n", v);
                                    //return v1 + k*v2;
                                    return k * v2 + v1;
                            });

	        double elapsed = t.elapsed();
	        std::cout
		        << ( boost::format("elapsed time == %2% [s]\n")
		           % n  % elapsed);
        }
		count = 0;
        std::for_each(boost::begin(a), boost::end(a),
                [&count,&c_init,&a_init,k](float v) -> void {

                        float cj = c_init[count];
                        float aj = a_init[count];
                        float bj = k*cj;
                        cj = aj+bj;
                        aj = bj+k*cj;
                        HPX_TEST_EQ(v, aj);
                        ++count;
                });
        HPX_TEST_EQ(count, e.size());


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
