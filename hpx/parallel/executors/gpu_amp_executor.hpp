//  Copyright (c) 2015 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/sequential_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_GPU_AMP_EXECUTOR_JUNE_06_2015_0400AM)
#define HPX_PARALLEL_EXECUTORS_GPU_AMP_EXECUTOR_JUNE_06_2015_0400AM

#include <hpx/config.hpp>
#include <hpx/traits/is_gpu_executor.hpp>
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
    struct gpu_amp_executor : gpu_executor_tag
    {
		#if defined(DOXYGEN)
				/// Create a new sequential executor
				gpu_amp_executor() {}
		#endif

        typedef gpu_execution_tag execution_category;
        template<typename value_type>
        struct buffer_iterator;
		template<typename Iter>
    	struct gpu_amp_buffer : detail::gpu_executor_buffer<Iter,
			Concurrency::array_view< typename std::iterator_traits<Iter>::value_type > >
		{
    		typedef typename std::iterator_traits<Iter>::value_type value_type;
    		typedef typename Concurrency::array<value_type> buffer_type;
    		typedef typename Concurrency::array_view<value_type> buffer_view_type;

    		Iter cpu_buffer;
    		//Concurrency::extent<1> extent;
            std::size_t count;
    		std::shared_ptr<buffer_type> buffer;
    		std::shared_ptr<buffer_view_type> _buffer_view;

    		gpu_amp_buffer(Iter first, std::size_t count) :
    			cpu_buffer( first ),
                count(count)
    			//extent( count )
    		{
    			Iter last = first;
    			std::advance(last, count);

    		    Concurrency::extent<1> extent(count);
    			buffer.reset( new buffer_type(extent, first, last) );
    			_buffer_view.reset( new buffer_view_type( *buffer.get() ) );
    		}

    		gpu_amp_buffer(Iter first, Iter last) :
    			cpu_buffer( first )
    		{
    			count = std::distance(first, last);
                std::cout << "Input count: " << count << std::endl;
    		    Concurrency::extent<1> extent(count);
                
    			buffer.reset( new buffer_type(extent, first, last) );
    			_buffer_view.reset( new buffer_view_type( *buffer.get() ) );
    		}

    		buffer_view_type * buffer_view()
    		{
    			return _buffer_view.get();
    		}

    		void sync()
    		{
			    Concurrency::copy(*buffer.get(), cpu_buffer);
    		}

    		void print()
    		{
    			std::cout << "buffer: " << *cpu_buffer << std::endl;
    		}
            
    		buffer_iterator<value_type> begin()
            {
                return buffer_iterator<value_type>(*buffer.get(), 0, count);
            }

            buffer_iterator<value_type> end()
            {
                return buffer_iterator<value_type>(*buffer.get(), count, count);
            }
		};
   
        template<typename value_type>
        struct buffer_iterator
        {
        private:
            Concurrency::array_view<value_type> array_view;
            uint32_t idx_, size;
        public:
            explicit buffer_iterator(Concurrency::array<value_type> & array, uint32_t idx, uint32_t size) :
                array_view(array), idx_(idx), size(size) {}

            explicit buffer_iterator(const Concurrency::array_view<value_type> & array_view, uint32_t idx, uint32_t size) :
                array_view(array_view), idx_(idx), size(size) {}

            template<typename ValueType>
            explicit buffer_iterator(const Concurrency::array_view<ValueType> & array_view, uint32_t idx, uint32_t size) :
                array_view(array_view), idx_(idx), size(size) {}

            template<typename ValueType>
            buffer_iterator(const buffer_iterator<ValueType> & other) :
                array_view(other.array_view), idx_(other.idx_), size(other.size) {}

            buffer_iterator(const buffer_iterator & other) :
                array_view(other.array_view), idx_(other.idx_), size(other.size) {}
            
            buffer_iterator & operator=(const buffer_iterator & a) restrict(amp, cpu)
            {
                return *this;
            }

            Concurrency::array_view<value_type> get_array_view() { return array_view; }

            buffer_iterator const& operator++() restrict(amp, cpu)
            {
                ++idx_;
                return *this;
            }

            buffer_iterator const& operator--() restrict(amp, cpu)
            {
                --idx_;
                return *this;
            }

            buffer_iterator operator++(int) restrict(amp, cpu)
            {
            	buffer_iterator copy(*this);
                ++idx_;
                return copy;
            }

            buffer_iterator operator--(int) restrict(amp, cpu)
            {
            	buffer_iterator copy(*this);
                --idx_;
                return copy;
            }

            explicit operator bool() const restrict(amp, cpu)
            {
                return idx_;
            }

            bool operator==(buffer_iterator const& other) const restrict(amp, cpu)
            {
                return idx_ == other.idx_;
            }

            bool operator!=(buffer_iterator const& other) const restrict(amp, cpu)
            {
                return idx_ != other.idx_;
            }

            bool operator<(buffer_iterator const& other) const restrict(amp, cpu)
            {
                return idx_ < other.idx_;
            }

            bool operator>(buffer_iterator const& other) const restrict(amp, cpu)
            {
                return idx_ > other.idx_;
            }

            bool operator<=(buffer_iterator const& other) const restrict(amp, cpu)
            {
                return idx_ <= other.idx_;
            }

            bool operator>=(buffer_iterator const& other) const restrict(amp, cpu)
            {
                return idx_ >= other.idx_;
            }

            buffer_iterator& operator+=(std::ptrdiff_t offset) restrict(amp, cpu)
            {
                idx_ += offset;
                return *this;
            }

            buffer_iterator & operator-=(std::ptrdiff_t offset) restrict(amp, cpu)
            {
                idx_ -= offset;
                return *this;
            }

            std::ptrdiff_t operator-(buffer_iterator const& other) const restrict(amp, cpu)
            {
                return idx_ - other.idx_;
            }

            buffer_iterator operator-(std::ptrdiff_t offset) const restrict(amp, cpu)
            {
                return buffer_iterator(array_view, idx_ - offset, size);
            }

            buffer_iterator operator+(std::ptrdiff_t offset) const restrict(amp, cpu)
            {
                return buffer_iterator(array_view, idx_ + offset, size);
            }

            value_type & operator*() restrict(amp, cpu)
            {
                return array_view[idx_];
            }

            value_type const& operator[](std::ptrdiff_t offset) const restrict(amp, cpu)
            {
                return array_view[idx_ + offset];
            }

            value_type& operator[](std::ptrdiff_t offset) restrict(amp, cpu)
            {
                return array_view[idx_ + offset];
            }

            operator value_type*() const
            {
                return array_view[idx_];
            }

            value_type* operator->() const
            {
                return &array_view[idx_];
            }
        };

    	template<typename Iter>
    	static gpu_amp_buffer<Iter> create_buffers(Iter first, std::size_t count)
		{
    		return gpu_amp_buffer<Iter>(first, count);
		}

    	template<typename Iter>
    	static gpu_amp_buffer<Iter> create_buffers(Iter first, Iter end)
		{
    		return gpu_amp_buffer<Iter>(first, end);
		}

    	template<typename Iter>
    	static std::shared_ptr<gpu_amp_buffer<Iter>> create_buffers_shared(Iter first, std::size_t count)
		{
    		return std::make_shared<gpu_amp_buffer<Iter>>(first, count);
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

		/*template <typename F, typename Shape>
		static std::vector<hpx::future<
			typename detail::bulk_async_execute_result<F, Shape>::type
		> >
		bulk_async_execute(F && f, Shape const& shape)
		{
			typedef typename
				    detail::bulk_async_execute_result<F, Shape>::type
				result_type;
			std::vector< hpx::future<result_type> > results;

			try {	
				for (auto const& elem: shape) {
					std::size_t data_count = elem.first;
					std::size_t chunk_size = elem.second;

					std::size_t threads_to_run = data_count / chunk_size;
					std::size_t last_thread_chunk = data_count - (threads_to_run - 1)*chunk_size;

					results.push_back(hpx::async(launch::async,
						/**
						 * Lambda calling the AMP parallel execution.
						 */
						/*[=]() {
							Concurrency::extent<1> e(threads_to_run);
							Concurrency::parallel_for_each(e, [=](Concurrency::index<1> idx) restrict(amp)
							{
								auto _x = std::make_pair(idx[0] * chunk_size, idx[0] != static_cast<int>(threads_to_run - 1) ? chunk_size : last_thread_chunk);
								f(_x);
							});
						}));
				}
			}
			catch (std::bad_alloc const& ba) {
				boost::throw_exception(ba);
			}
			catch (...) {
				boost::throw_exception(
				    exception_list(boost::current_exception())
				);
			}

			return std::move(results);
		}*/

		template <typename F, typename Shape>
		static std::vector<hpx::future<
			typename detail::bulk_async_execute_result<F, Shape>::type
		> >
		bulk_async_execute(F && f, Shape const& shape)
		{
			typedef typename
				    detail::bulk_async_execute_result<F, Shape>::type
				result_type;
			std::vector< hpx::future<result_type> > results;

			try {	
				for (auto const& elem: shape) {
				auto iter = hpx::util::get<0>(elem);
                std::size_t offset = iter.idx();
                //Concurrency::array_view<std::size_t> array_view = iter.get_array_view();
				std::size_t data_count = hpx::util::get<1>(elem);
                //deal with it later
                std::size_t chunk_size = hpx::util::get<2>(elem);
				
				std::size_t threads_to_run = data_count / chunk_size;
				std::size_t last_thread_chunk = data_count - (threads_to_run - 1)*chunk_size;

					results.push_back(hpx::async(launch::async,
						/**
						 * Lambda calling the AMP parallel execution.
						 */
						[=]() {
							Concurrency::extent<1> e(threads_to_run);
							Concurrency::parallel_for_each(e, [=](Concurrency::index<1> idx) restrict(amp)
							{
                				std::size_t part_size = idx[0] != static_cast<int>(threads_to_run - 1) ? chunk_size : last_thread_chunk;
								auto it = iter;
						        //it.advance(idx[0]*chunk_size);
						        it.idx() += idx[0]*chunk_size;
						        hpx::util::tuple<decltype(it), std::size_t, std::size_t> tuple(it, 0, part_size);                    
						        f( tuple );
							});
						}));
				}
			}
			catch (std::bad_alloc const& ba) {
				boost::throw_exception(ba);
			}
			catch (...) {
				boost::throw_exception(
				    exception_list(boost::current_exception())
				);
			}

			return std::move(results);
		}

		template <typename F, typename Iter>
		static void bulk_execute(Iter first, std::size_t count, F && f)
		{
			std::size_t data_count = count;
			std::size_t chunk_size = 1;
			
			std::size_t threads_to_run = data_count / chunk_size;
			std::size_t last_thread_chunk = data_count - (threads_to_run - 1)*chunk_size;
            std::size_t offset = first.idx();
            auto array_view = first.get_array_view();


			Concurrency::extent<1> e(threads_to_run);
			Concurrency::parallel_for_each(e, [=](Concurrency::index<1> idx) restrict(amp) 
			{
                //Iter start(first);
                //std::advance(start, idx[0] * chunk_size);
                std::size_t part_begin = idx[0] * chunk_size;
                std::size_t part_size = part_begin + (idx[0] != static_cast<int>(threads_to_run - 1) ? chunk_size : last_thread_chunk);
                for(std::size_t i = part_begin; i < part_size ; ++i)
					f( array_view[offset + i] );
			});

            /** Synchronize **/
            /** TODO: put acc_view in an executor **/
            first.get_array_view().get_source_accelerator_view().wait();
            /**
                This should not be in an executor
            **/
            //Concurrency::copy(*buffer.get(), cpu_buffer);
		}

		template <typename Shape, typename F>
		static typename detail::bulk_execute_result<F, Shape>::type
		bulk_execute(F && f, Shape const& shape)
		{
			typedef typename Shape::value_type tuple_t;
			/**
			 * The elements of pair are:
			 * begin at array, # of elements to process
			 */
			for(auto const & elem : shape) {
				auto iter = hpx::util::get<0>(elem);
				//using value_type = typename iter::value_type;
                //std::size_t offset = iter.idx();
                //auto array_view = iter.get_array_view();
				std::size_t data_count = hpx::util::get<1>(elem);
                //deal with it later
                std::size_t chunk_size = hpx::util::get<2>(elem);
				
				std::size_t threads_to_run = data_count / chunk_size;
				std::size_t last_thread_chunk = data_count - (threads_to_run - 1)*chunk_size;

                // This works
                //gpu_amp_buffer_iterator<std::size_t> it(array_view, 1,1);
				Concurrency::extent<1> e(threads_to_run);
				Concurrency::parallel_for_each(e, [=](Concurrency::index<1> idx) restrict(amp)
				{
                    //gpu_amp_buffer_iterator<std::size_t> it(array_view, (uint32_t)offset + idx[0]*chunk_size, (uint32_t)data_count);
                    //gpu_amp_buffer_iterator<std::size_t> it(iter);
                	std::size_t part_size = idx[0] != static_cast<int>(threads_to_run - 1) ? chunk_size : last_thread_chunk;
                    auto it = iter;
                    //it.advance(idx[0]*chunk_size);
                    //it.idx() += idx[0]*chunk_size;
					//std::advance(it, idx[0]*chunk_size);                 
					tuple_t tuple(it, 0, part_size);                    
                    f( tuple );
				});
            	//array_view.get_source_accelerator_view().wait();
			}
		}

        template <typename F, typename Parameters>
        static void bulk_execute(Parameters & params, F && f, std::size_t data_count, std::size_t chunk_size, Concurrency::accelerator_view& accl_view)
        {

	        /**
	         * The elements of pair are:
	         * begin at array, # of elements to process
	         */
	        //for(auto const & elem : shape) {
			
			std::size_t threads_to_run = data_count / chunk_size;
			std::size_t last_thread_chunk = data_count - (threads_to_run - 1)*chunk_size;
			//std::cout << "Sync: " << chunk_size << " " << data_count << " " << threads_to_run << " " << last_thread_chunk << std::endl;

			Concurrency::extent<1> e(threads_to_run);
			Concurrency::parallel_for_each(accl_view, e, [=](Concurrency::index<1> idx) restrict(amp) 
			{
				//auto _x = std::make_pair(idx[0] * chunk_size, idx[0] != static_cast<int>(threads_to_run - 1) ? chunk_size : last_thread_chunk);
				//f(_x);
	            f(idx[0], 1);
			});
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
