#include <amp.h>
#include <iostream>
#include "amp_template.h"
#include "lambda.h"

using namespace Concurrency;

void call_amp(std::vector<int> & v, size_t count, const function< int(double) > & f)
{
	//array_view<int> av(count, v);
//	auto lambda = [=](int v)  -> double { return 1;};
 	extent<1> e(count);
	auto it = v.begin();
	auto end = it;
	std::advance(end,count);
        array<int> arr(e, it, end);
        auto lambda = [=](int v)  -> double { return 1;};
        array_view<int> av(arr);


	parallel_for_each(av.get_extent(), [=](index<1> idx) restrict(amp) {
    		av[idx] += lambda(0); //Lambda::get_lambda()(0);
 	});
}

void call_amp(std::list<int> & v, size_t count, const function< int(double) > & f)
{
	extent<1> e(count);
        array<int> arr(e, v.begin(), v.end());
        auto lambda = [=](int v)  -> double { return 1;};
//	array_view<int> av(e, arr);
	array_view<int> av(arr);
        parallel_for_each(av.get_extent(), [=](index<1> idx) restrict(amp) {
                av[idx] += lambda(0); //Lambda::get_lambda()(0);
        });
	copy(arr, v.begin());
}


