//  Copyright (c) 2007-2015 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/sequential_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_GPU_AMP_EXECUTOR_JUNE_06_2015_0400AM)
#define HPX_PARALLEL_EXECUTORS_GPU_AMP_EXECUTOR_JUNE_06_2015_0400AM

#include <hpx/config.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/exception_list.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/unwrapped.hpp>

#include <type_traits>
#include <utility>
#include <iterator>

#include <boost/range/functions.hpp>
#include <boost/range/const_iterator.hpp>
#include <boost/type_traits/is_void.hpp>

#include <amp.h>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{


	///////////////////////////////////////////////////////////////////////////
    /// A \a sequential_executor creates groups of sequential execution agents
    /// which execute in the calling thread. The sequential order is given by
    /// the lexicographical order of indices in the index space.
    ///
    struct gpu_amp_executor
    {
		#if defined(DOXYGEN)
				/// Create a new sequential executor
				gpu_amp_executor() {}
		#endif

		template<typename Iter>
    	struct gpu_amp_buffer : detail::gpu_executor_buffer<Iter,
			Concurrency::array_view< typename std::iterator_traits<Iter>::value_type > >
		{
    		typedef typename std::iterator_traits<Iter>::value_type value_type;
    		typedef typename Concurrency::array<value_type> buffer_type;
    		typedef typename Concurrency::array_view<value_type> buffer_view_type;

    		Iter cpu_buffer;
    		Concurrency::extent<1> extent;
    		std::shared_ptr<buffer_type> buffer;
    		std::shared_ptr<buffer_view_type> _buffer_view;

    		gpu_amp_buffer(Iter first, std::size_t count) :
    			cpu_buffer( first ),
    			extent( count )
    		{
    			Iter last = first;
    			std::advance(last, count);

    			buffer.reset( new buffer_type(extent, first, last) );
    			_buffer_view.reset( new buffer_view_type( *buffer.get() ) );
    		}

    		buffer_view_type & buffer_view()
    		{
    			return *_buffer_view.get();
    		}

    		void sync()
    		{
				Concurrency::copy(*buffer.get(), cpu_buffer);
    		}
		};

    	template<typename Iter>
    	static gpu_amp_buffer<Iter> create_buffers(Iter first, std::size_t count)
		{
    		return gpu_amp_buffer<Iter>(first, count);
		}

        template <typename F>
        static typename hpx::util::result_of<
            typename hpx::util::decay<F>::type()
        >::type
        execute(F && f)
        {
        	throw std::runtime_error("Feature not supported in GPU AMP executor! Please, use bulk execute.");
        }

        template <typename F>
        static hpx::future<typename hpx::util::result_of<
            typename hpx::util::decay<F>::type()
        >::type>
        async_execute(F && f)
        {
        	throw std::runtime_error("Feature not supported in GPU AMP executor! Please, use bulk execute.");
        }

        template <typename F, typename Shape>
        static std::vector<hpx::future<
            typename detail::bulk_async_execute_result<F, Shape>::type
        > >
        bulk_async_execute(F && f, Shape const& shape)
        {
            typedef typename
                    detail::bulk_async_execute_result<F, Shape>::type
                result_type;
            std::vector<hpx::future<result_type> > results;
            std::cout << "ASYNC" << std::endl;
            /*try {
                for (auto const& elem: shape)
                    results.push_back(hpx::async(launch::deferred, f, elem));
            }
            catch (std::bad_alloc const& ba) {
                boost::throw_exception(ba);
            }
            catch (...) {
                boost::throw_exception(
                    exception_list(boost::current_exception())
                );
            }*/
        	throw std::runtime_error("Not implemented!");

           // return std::move(results);
        }

        template <typename F, typename Shape>
        static typename detail::bulk_execute_result<F, Shape>::type
        bulk_execute(F && f, Shape const& shape)
        {

        	/**
        	 * The elements of pair are:
        	 * begin at array, # of elements to process
        	 */
        	for(auto const & elem : shape) {
				std::size_t x = elem.first;
				std::size_t y = elem.second;
	        	Concurrency::extent<1> e(y);
        		Concurrency::parallel_for_each(e, [=](Concurrency::index<1> idx) restrict(amp) {
        			auto _x = std::make_pair(x + idx[0], y);
	        		f(_x);
				});
        	}
        }

        std::size_t os_thread_count()
        {
            return 1;
        }
        /// \endcond
    };

    namespace detail
    {
        /// \cond NOINTERNAL
        template <>
        struct is_executor<gpu_amp_executor>
          : std::true_type
        {};
        /// \endcond
    }
}}}

#endif
