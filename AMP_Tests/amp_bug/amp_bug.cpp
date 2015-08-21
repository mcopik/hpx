
#include <iostream>
#include <cassert>
#include <numeric>
#include <utility>
#include <vector>

#include <boost/range.hpp>
#include <amp.h>


namespace hpx { namespace parallel { inline namespace v3 { namespace detail {

class ImProperExecutor {
public:
	template<typename X, typename F, typename Data>
	static void run(X && x, F && f, Data count)
	{
		Concurrency::extent<1> e(count);
		Concurrency::parallel_for_each(e, [=](Concurrency::index<1> idx) restrict(amp) {
			auto _x = std::make_pair(idx[0], 1);
			f(_x);
		});
	}
	template<typename X, typename F>
	static void run(X && x, F && f, std::size_t count)
	{
		Concurrency::extent<1> e(count);
		Concurrency::parallel_for_each(e, [=](Concurrency::index<1> idx) restrict(amp) {
			auto _x = std::make_pair(idx[0], 1);
			f(_x);
		});
	}
};

class ProperExecutor {
public:
	template<typename F, typename Data>
	static void run(F && f, Data count)
	{
		std::vector<int> x;
		ImProperExecutor::run(x, std::forward<F>(f),count);
/*		Concurrency::extent<1> e(count);
		Concurrency::parallel_for_each(e, [=](Concurrency::index<1> idx) restrict(amp) {
			auto _x = std::make_pair(idx[0], 1);
			f(_x);
		});*/
	}

	template<typename F>
	static void run(F && f, std::size_t count)
	{
		std::vector<int> x;
		ImProperExecutor::run(x, std::forward<F>(f),count);
		/*Concurrency::extent<1> e(count);
		Concurrency::parallel_for_each(e, [=](Concurrency::index<1> idx) restrict(amp) {
			auto _x = std::make_pair(idx[0], 1);
			f(_x);
		});*/
	}
};
} } } }

namespace hpx { namespace parallel { inline namespace v3 { namespace detail {
class Executor {
public:
	template<typename F>
	static void run(F && f1, std::size_t count)
	{
		auto f = [f1](std::pair<std::size_t, std::size_t> const& elem)
			{
				//ignore second in this example
				f1(elem.first, elem.second);
			};
		ProperExecutor::run(std::forward<decltype(f)>(f), count);
	}
};
} } } }

namespace hpx { namespace parallel { inline namespace v3 { namespace detail {
class Partitioner {
public:
	template<typename F>
	static void partition_run(F && f, std::size_t count)
	{
		Executor::run(std::forward<F>(f), count);
	}
};
} } } }

namespace hpx { namespace parallel { inline namespace v3 { namespace detail {
template<typename F>
void partition_run(F && f, std::size_t count)
{
	Partitioner::partition_run(f,count);//std::forward<F>(f), count);
}
} } } }

namespace hpx { namespace parallel {
template<typename F>
void run(F && f, std::size_t count)
{
	detail::partition_run([f](std::size_t pos, std::size_t chunk) {
		for(std::size_t i = 0;i < chunk;++i)
			f(pos + i);
	}, count);
}
}}


int main(int argc, char ** argv)
{
	int n = 10;
	auto fc = [](std::size_t & v) -> void { v = 42; };
	auto fd = [](std::size_t & v) -> void { v = 43; };
	std::vector<std::size_t> c(n);
	std::vector<std::size_t> d(n);	
	std::iota(std::begin(c), std::end(c), std::rand());
	std::iota(std::begin(d), std::end(d), std::rand());

	Concurrency::extent<1> extentc(n);
	Concurrency::extent<1> extentd(n);
	Concurrency::array<std::size_t> arc(extentc, boost::begin(c), boost::end(c));
	Concurrency::array<std::size_t> ard(extentd, boost::begin(d), boost::end(d));
	Concurrency::array_view<std::size_t> avc(arc);
	Concurrency::array_view<std::size_t> avd(ard);
	hpx::parallel::run([fc, &avc](std::size_t pos) { fc(avc[pos]); }, n);
	hpx::parallel::run([fd, &avd](std::size_t pos) { fd(avd[pos]); }, n);
	Concurrency::copy(avc, boost::begin(c));
	Concurrency::copy(avd, boost::begin(d));


	std::size_t count = 0;
	std::for_each(std::begin(c), std::end(c),
		[&count](std::size_t v) -> void {
			if(v != std::size_t(42)){ std::cout << "not equal in C " << v << std::endl; };
			++count;
		});
	assert(count == c.size());

	count = 0;
	std::for_each(std::begin(d), std::end(d),
		[&count](std::size_t v) -> void {
			if(v != std::size_t(43)){ std::cout << "not equal in D " << v << std::endl; };
			++count;
		});
	assert(count == d.size());

	return 0;
}
