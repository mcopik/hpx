#ifndef __AMP_TEMPLATE__
#define __AMP_TEMPLATE__
#include <functional>
#include <vector>
#include <list>

using std::function;


void call_amp(std::vector<int> &, size_t count, const function< int(double) > &);
void call_amp(std::list<int> &, size_t count, const function< int(double) > &);

template<typename Container, typename Size, typename F>
void run_amp(Container & first, Size count, F && f)
{
	//Concurrency::array<std::iterator_trais<Iter>::value_type ,1> array(count, first);
	call_amp(first, count, f);
}

#endif
