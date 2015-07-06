#include <iostream>

#include <amp.h>


void run_amp(std::vector<int> & v)
{
	auto f = [&](int & v) -> void { v += 1; };
	Concurrency::array_view<int> av(v.size(), v);
	Concurrency::parallel_for_each(av.get_extent(), [=](Concurrency::index<1> idx) restrict(amp) {
		f(av[idx]);
	});
}

int main(int argc, char ** argv)
{

	std::vector<int> v = { 'G', 'd', 'k', 'k', 'n', 31, 'v', 'n', 'q', 'k', 'c'};
	run_amp(v);
	for(unsigned int i = 0; i < v.size(); i++)
        std::cout << static_cast<char>(v.at(i));
	run_amp(v);

	return 0;
}
