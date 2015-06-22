#include <amp.h>
#include <iostream>
#include "amp_template.h"
#include "lambda.h"

using namespace Concurrency;

void call_amp(std::vector<int> & v, size_t count, const function< int(double) > & f)
{
	array_view<int> av(count, v);

	auto lambda = [=](int v)  -> double { return 1;};

	parallel_for_each(av.get_extent(), [=](index<1> idx) restrict(amp) {
    		av[idx] += lambda(0); //Lambda::get_lambda()(0);
 	});
}

