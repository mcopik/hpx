#ifndef __LAMBDA_H__
#define __LAMBDA_H__
#include <functional>

using std::function;

struct Lambda {
public:
	static const function< int(double) > & get_lambda()
	{
		static const function< int(double) > lambda = [=](int v)  -> double { return 1;};
		return lambda;
	}
};


#endif
