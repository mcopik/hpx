#include <hpx/hpx.hpp>
#include <hpx/hpx_init.hpp>
#include <iostream>
#include <future>

#include <boost/cstdint.hpp>
#include <boost/format.hpp>

#include <SYCL/sycl.hpp>

using namespace cl::sycl;

namespace hpx { namespace parallel {

	struct DefaultKernelName;
	struct GenericKernelName;
}}

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{

namespace detail{
	template<typename F, typename KernelName>
	struct kernel
	{
		using functor_type = F;
		using kernel_name = KernelName;

		kernel(const F & f) : f(f) {}


		template<typename ...Args>
		typename std::result_of<F(Args...)>::type operator()(Args &&... args) const
		{
			return f( std::forward<Args>(args)... );
		}

	private:
		//F && f;
		F f;
	};


	template<typename Kernel, typename Enable = void>
	struct get_kernel_name
	{
		typedef GenericKernelName kernel_name;
	};

/*	template<
		typename Kernel,
		typename std::enable_if<std::is_same<typename Kernel::kernel_name, DefaultKernelName>::value>::type* = nullptr
	>
	struct get_kernel_name
	{
		typedef typename Kernel::functor_type kernel_name;
	};

	template<
		typename Kernel,
		typename std::enable_if<!std::is_same<typename Kernel::kernel_name, DefaultKernelName>::value>::type* = nullptr
	>
	struct get_kernel_name
	{
		typedef typename KernelName kernel_name;
	};*/

	template<typename F, typename KernelName>
	struct get_kernel_name<kernel<F, KernelName>, typename std::enable_if<std::is_same<KernelName, DefaultKernelName>::value>::type>
	{
		typedef typename kernel<F, KernelName>::functor_type kernel_name;
	};

	template<typename F, typename KernelName>
	struct get_kernel_name<kernel<F, KernelName>, typename std::enable_if<!std::is_same<KernelName, DefaultKernelName>::value>::type>
	{
		typedef typename kernel<F, KernelName>::kernel_name kernel_name;
	};
}
}}}

namespace hpx { namespace parallel
{

	template<typename KernelName = DefaultKernelName, typename F>
	v3::detail::kernel<F, KernelName> make_kernel(F && f)
	{
		return v3::detail::kernel<typename hpx::util::decay<F>::type, KernelName>(f);
	} 

    template <typename Kernel>
    struct get_kernel_name
      : v3::detail::get_kernel_name<typename hpx::util::decay<Kernel>::type>
    {};
}}

template<typename Range, typename Kernel, typename OrigKernel>
void call_kernel(handler& cgh, const Range & range, Kernel kernel, OrigKernel &)
{
	using kernel_name = typename hpx::parallel::get_kernel_name<OrigKernel>::kernel_name;
	std::cout << typeid(kernel_name).name() << std::endl;
	cgh.parallel_for< kernel_name >(range, kernel);
}

	
int hpx_main(/*boost::program_options::variables_map& vm*/)
{

	hpx::util::high_resolution_timer t;
	//boost::uint64_t n = vm["n-value"].as<boost::uint64_t>();
	int n = 10;
	int * data = new int[n];
  	for (int i = 0; i < n; i++) {
		data[i] = 0;
  	}

	// Should not work
	//auto my_kernel = hpx::parallel::make_kernel( [](int x) { return x + 1; } );
	// As you see...
	//auto my_kernel = hpx::parallel::make_kernel<class SuperKernelName>( [](int x) { return x + 1; } );
	// Another DefaultKernelName
	//auto my_kernel = [](int x) { return x + 1; };
	// genericKernelName
	struct Functor
	{
		int operator()(int x) const
		{
			return x+1;
		}
	};
	Functor my_kernel;
	//DefaultKernelName
	//auto my_kernel = hpx::parallel::make_kernel( Functor() );

//	auto my_kernel2 = hpx::parallel::make_kernel( [](int & x) { x += 1; } );
//	int x = 0;
//	my_kernel2(x);
	try
	{
		queue myQueue;
		buffer<int, 1> buf(data, range<1>(n));
		//if(true) {

			myQueue.submit([&](handler& cgh) {
				// Ok !
				using kernel_name = hpx::parallel::get_kernel_name<decltype(my_kernel)>::kernel_name;

				auto ptr = buf.get_access<access::mode::read_write>(cgh);
				auto myKernel = (range<1>(n), [=](id<1> idx) {
					ptr[idx[0]] = my_kernel(static_cast<int>(idx[0]));
				});
				//call_kernel(cgh, range<1>(n), myKernel, my_kernel);
				cgh.parallel_for< class Functor /*kernel_name*/ >(range<1>(n), myKernel);
			});

		/*} else {

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
		}*/

	} catch (exception e) {
		std::cout << "SYCL exception caught: " << e.what();
		return 2;
	}
 
	for (int i = 0; i < n; i++) {
		if (data[i] != i + 1) {
			std::cout << "The results not are correct at: " << i << std::endl;
		}
	}

	double elapsed = t.elapsed();
	std::cout << ( boost::format("elapsed time == %2% [s]\n") % n  % elapsed);
	delete[] data;
	return 0;
	//return hpx::finalize();
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
	//return hpx::init(desc_commandline, argc, argv);
	hpx_main();
}
