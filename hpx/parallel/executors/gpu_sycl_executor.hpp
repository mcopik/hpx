//  Copyright (c) 2015 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/sequential_executor.hpp

#if !defined(HPX_PARALLEL_EXECUTORS_GPU_SYCL_EXECUTOR)
#define HPX_PARALLEL_EXECUTORS_GPU_SYCL_EXECUTOR

#include <SYCL/sycl.hpp>

#include <hpx/config.hpp>
#include <hpx/parallel/kernel.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>
#include <hpx/parallel/exception_list.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/executors/dynamic_chunk_size.hpp>
#include <hpx/runtime/threads/thread_executor.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/iterator_adaptor.hpp>
#include <hpx/util/result_of.hpp>
#include <hpx/util/unwrapped.hpp>

#include <type_traits>
#include <utility>
#include <iterator>

#include <boost/range/functions.hpp>
#include <boost/range/const_iterator.hpp>
#include <boost/type_traits/is_void.hpp>

//#include <SYCL/sycl.hpp>

namespace hpx { namespace parallel { HPX_INLINE_NAMESPACE(v3)
{

    template<typename T>    	
    struct sycl_buffer
	{
        typedef cl::sycl::buffer<T> buffer_t;
        typedef decltype(std::declval<buffer_t>().template get_access<cl::sycl::access::mode::read_write>(std::declval<cl::sycl::handler &>())) device_acc_t;
        typedef decltype(std::declval<buffer_t>().template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>()) host_acc_t;
        std::shared_ptr<buffer_t> buffer;

        template<typename Iter, typename std::enable_if< std::is_same<typename std::iterator_traits<Iter>::value_type, T>::value >::type* = nullptr>
        sycl_buffer(Iter begin, Iter end):
            buffer( new buffer_t(begin, end) )
        {}

        //TODO: implement iterators
        device_acc_t get_access(cl::sycl::handler & cgh) const
        {
            return buffer->template get_access<cl::sycl::access::mode::read_write>(cgh);
        }

        host_acc_t get_access() const
        {
            return buffer->template get_access<cl::sycl::access::mode::read, cl::sycl::access::target::host_buffer>();
        }
	};

    namespace detail
    {
        struct sycl_iterator_tag {};

        template <typename T>
        struct device_iterator
          : hpx::util::iterator_facade<
                device_iterator<T>,
                cl::sycl::global_ptr<T>,
                std::random_access_iterator_tag,
                typename cl::sycl::global_ptr<T>::reference_t,
                //typename cl::sycl::global_ptr<T>::difference_type
                std::ptrdiff_t
            >
        {
            typedef hpx::util::iterator_facade<
                    device_iterator<T>,
                    cl::sycl::global_ptr<T>,
                    std::random_access_iterator_tag,
                    typename cl::sycl::global_ptr<T>::reference_t,
                    //typename cl::sycl::global_ptr<T>::difference_type
                    std::ptrdiff_t
            > base_type;

            typedef typename cl::sycl::global_ptr<T>::reference_t Reference;
            //typedef typename cl::sycl::global_ptr<T>::difference_type Distance;
            typedef typename std::ptrdiff_t Distance;

            HPX_HOST_DEVICE device_iterator()
              : base_type(), ptr(nullptr)
            {}

            HPX_HOST_DEVICE
            device_iterator(cl::sycl::global_ptr<T> p, std::size_t pos = 0)
              : ptr(p + pos)
            {}
            
            template<typename U>
            HPX_HOST_DEVICE
            device_iterator(cl::sycl::global_ptr<U> p, std::size_t pos)
              : ptr(p + pos)
            {}

            HPX_HOST_DEVICE device_iterator(device_iterator const& other)
              : base_type(other), ptr(other.ptr)
            {}

            template <typename Iterator1>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            bool equal(Iterator1 const& other)
            {
                return ptr == other.ptr;
            }

            HPX_HOST_DEVICE HPX_FORCEINLINE
            void increment()
            {
                ++ptr;
            }

            HPX_HOST_DEVICE HPX_FORCEINLINE
            void decrement()
            {
                --ptr;
            }

            HPX_HOST_DEVICE HPX_FORCEINLINE
            Reference dereference() const
            {
                return *ptr;
            }

            HPX_HOST_DEVICE HPX_FORCEINLINE
            void advance(Distance n)
            {
                ptr += n;
            }

            template <typename Iterator1>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            Distance distance_to(Iterator1 const& other)
            {
                return ptr.pointer() - other.ptr.pointer();
            }

        private:
            mutable cl::sycl::global_ptr<T> ptr;
        };

        template <typename T>
        struct host_iterator
          : hpx::util::iterator_facade<
                host_iterator<T>,
                T*,
                std::random_access_iterator_tag,
                typename cl::sycl::global_ptr<T>::reference_t,
                //typename cl::sycl::global_ptr<T>::difference_type
                std::ptrdiff_t
            >
        {
            typedef hpx::util::iterator_facade<
                    host_iterator<T>,
                    cl::sycl::global_ptr<T>,
                    std::random_access_iterator_tag,
                    typename cl::sycl::global_ptr<T>::reference_t,
                    std::ptrdiff_t
            > base_type;

            typedef sycl_buffer<T> buffer_t;
            typedef typename buffer_t::host_acc_t buffer_acc_t;
            typedef typename buffer_t::reference_t Reference;
            typedef typename std::ptrdiff_t Distance;

            HPX_HOST_DEVICE host_iterator()
              : base_type(), ptr(nullptr), pos_(0)
            {}

            HPX_HOST_DEVICE
            host_iterator(buffer_acc_t p, Distance pos)
              : ptr(p), pos_(pos_)
            {}

            HPX_HOST_DEVICE host_iterator(host_iterator const& other)
              : base_type(other), ptr(other.ptr), pos_(other.pos_)
            {}

            template <typename Iterator1>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            bool equal(Iterator1 const& other)
            {
                return ptr == other.ptr && pos_ == other.pos_;
            }

            HPX_HOST_DEVICE HPX_FORCEINLINE
            void increment()
            {
                ++pos_;
            }

            HPX_HOST_DEVICE HPX_FORCEINLINE
            void decrement()
            {
                --pos_;
            }

            HPX_HOST_DEVICE HPX_FORCEINLINE
            Reference dereference() const
            {
                if(!ptr) {
                    ptr.reset(new buffer_acc_t(buffer.get_access()));               
                }
                return (*ptr)[pos_];
            }

            HPX_HOST_DEVICE HPX_FORCEINLINE
            void advance(Distance n)
            {
                pos_ += n;
            }

            template <typename Iterator1>
            HPX_HOST_DEVICE HPX_FORCEINLINE
            Distance distance_to(Iterator1 const& other)
            {
                return pos_ - other.pos_;
            }

            Distance pos() const
            {
                return pos_;
            }

        private:
            Distance pos_;
            std::shared_ptr<buffer_acc_t> ptr;
            buffer_t buffer;
        };

        template<typename Iter, typename value_type = typename std::iterator_traits<Iter>::value_type>
        sycl_buffer<value_type> get_buffer(Iter begin, std::size_t count)
        {
            Iter end = begin;
            std::advance(end, count);
            return sycl_buffer<value_type>(begin, end);
        }
        
        template<typename T>
        sycl_buffer<T> get_buffer(detail::host_iterator<T> begin, std::size_t count)
        {
            return begin.get_buffer();
        }
        
        template<typename T>
        std::pair<typename sycl_buffer<T>::device_acc_t, std::ptrdiff_t>
        get_device_acc(cl::sycl::handler & cgh, const sycl_buffer<T> & buf, detail::host_iterator<T> begin)
        {
            return std::make_pair(buf.get_access(cgh), begin.pos());
        }
        
        template<typename T, typename Iter>
        typename sycl_buffer<T>::device_acc_t
        get_device_acc(cl::sycl::handler & cgh, const sycl_buffer<T> & buf, Iter begin)
        {
            return buf.get_access(cgh);
        }
        
        template<typename T>
        device_iterator<T>
        get_device_it(const std::pair<typename sycl_buffer<T>::device_acc_t, std::ptrdiff_t> & data, std::size_t idx)
        {
            return device_iterator<T>(data.first, data.second + idx);
        }
        
        template<typename T>
        device_iterator<T>
        get_device_it(const typename sycl_buffer<T>::device_acc_t & data, std::size_t idx)
        {
            return device_iterator<T>(data, idx);
        }
    }

	///////////////////////////////////////////////////////////////////////////
    /// A \a sequential_executor creates groups of sequential execution agents
    /// which execute in the calling thread. The sequential order is given by
    /// the lexicographical order of indices in the index space.
    ///
    struct gpu_sycl_executor
    {
		/// Returns always 1 if user doesn't provide a chunk size
		typedef hpx::parallel::dynamic_chunk_size executor_parameters_type;

		cl::sycl::default_selector selector;
        std::shared_ptr<cl::sycl::queue> queue;

		#if defined(DOXYGEN)
				/// Create a new sequential executor
				//gpu_sycl_executor() {}
		#endif
        //TODO: device selection
        gpu_sycl_executor() :
            queue(new cl::sycl::queue(selector))
        {

        }
		
		template<typename ValueType, typename BufferType>
		struct gpu_sycl_buffer_view_wrapper
		{
			ValueType & operator[](std::size_t idx);
		};

		template<typename Iter,
				typename value_type = typename std::iterator_traits<Iter>::value_type,
    			typename buffer_type = cl::sycl::buffer<typename std::iterator_traits<Iter>::value_type, 1>,
				typename _buffer_view_type =  decltype( std::declval< buffer_type >().template get_access<cl::sycl::access::mode::read_write>( std::declval<cl::sycl::handler &>() ) )>
    	struct gpu_sycl_buffer// : detail::gpu_executor_buffer<Iter, _buffer_view_type>
		{
		cl::sycl::default_selector selector;
			cl::sycl::queue queue;
    		Iter cpu_buffer;
    		std::shared_ptr<buffer_type> buffer;
			typedef _buffer_view_type buffer_view_type;
			// TODO: possibly not safe, change letter
    		buffer_view_type * _buffer_view;

    		gpu_sycl_buffer(Iter first, std::size_t count) :
				queue( selector ),
    			cpu_buffer( first ),
				_buffer_view( nullptr )
    		{
    			Iter last = first;
    			std::advance(last, count);
				std::shared_ptr<value_type> buf{new value_type[count], [first, count](value_type * ptr) {
								std::cout << "Copy values: " << *ptr << " " << count << std::endl;								
								std::copy(ptr, ptr + count, first);
								delete[] ptr;
							}};
				std::copy(first, last, buf.get());
				buffer.reset( new buffer_type(buf, cl::sycl::range<1>(count)) );
				buffer.get()->set_final_data(buf);
    		}

            buffer_type * buffer_obj()
            {
                return buffer.get();
            }

    		buffer_view_type * buffer_view()
    		{
    			return _buffer_view;
    		}

    		void sync()
    		{
				buffer.reset();
    		}

    		void print()
    		{
    			std::cout << "buffer: " << *cpu_buffer << std::endl;
    		}
		};
        

		//TODO: move to buffer parent class?
		template<typename Iter>
		struct buffer_traits {
			typedef typename gpu_sycl_buffer<Iter>::buffer_view_type type;		
		};

		template<typename buffer_type>
		struct buffer_view_type {
			typedef decltype( std::declval< buffer_type >().template get_access<cl::sycl::access::mode::read_write>( std::declval<cl::sycl::handler &>() ) ) type;		
		};

    	template<typename Iter>
    	static gpu_sycl_buffer<Iter> create_buffers(Iter first, std::size_t count)
		{
    		return gpu_sycl_buffer<Iter>(first, count);
		}

    	template<typename Iter>
    	static std::shared_ptr<gpu_sycl_buffer<Iter>> create_buffers_shared(Iter first, std::size_t count)
		{
    		return std::make_shared<gpu_sycl_buffer<Iter>>(first, count);
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

		template <typename F, typename Shape, typename ... Ts>
        std::vector<hpx::future<
            typename detail::bulk_async_execute_result<F, Shape, Ts...>::type
        > >
        bulk_async_execute(F && f, Shape const& shape, Ts const &... ts)
		{
            typedef typename
                    detail::bulk_async_execute_result<F, Shape, Ts...>::type
                result_type;
			//typedef typename GPUBuffer::buffer_view_type buffer_view_type;
			//using kernel_name = typename hpx::parallel::get_kernel_name<F, Parameters>::kernel_name;
            static const int LOCAL_SIZE = 128;
			std::vector<hpx::future<result_type> > results;

			try {
				for (auto const& elem: shape) {
                                
                    typedef typename boost::range_const_iterator<Shape>::type tuple_iterator_type;
                    //tuple type
                    typedef typename std::iterator_traits<tuple_iterator_type>::value_type tuple_type;
                    //iterator stored in tuple
                    typedef typename std::decay<decltype(hpx::util::get<1>(elem))>::type iterator_type;
                    typedef typename std::iterator_traits<iterator_type>::value_type value_type;
                    

                    int offset = hpx::util::get<0>(elem);
                    int data_count = hpx::util::get<2>(elem);
                    int local_size = std::min(LOCAL_SIZE, data_count);
                    int chunk_size = 1;
                    auto begin = hpx::util::get<1>(elem);
                    //auto end = begin;
                    //std::advance(end, data_count);
                    //std::cout << data_count << std::endl;
                    //iterator_type begin = hpx::util::get<1>(elem);
                    //sycl_buffer<value_type> buffer(begin, end);
                    auto buffer = detail::get_buffer(begin, data_count);
                    auto & queue_ = queue;
                    F _f = std::move(f);
                    
                    
                    //auto data_ = buffer.get_access();
                    //std::cout << data_[0] << " " << *data_.get_pointer() << " " << data_count << " " << offset << " " << chunk_size << std::endl;
                    //auto buffer_acc = buffer.get_access();
                    //auto t = hpx::util::make_tuple(offset, buffer_acc.get_pointer(), chunk_size);
                    //hpx::util::invoke(_f, t);

                    // Mutable is necessary to invoke the functor on a tuple of arguments
                    // Otherwise we get a failed substitution
                    auto command_group = [_f, buffer, begin, data_count, queue_, offset, chunk_size, local_size]() mutable {
                        queue_->submit( [_f, buffer, begin, data_count, offset, chunk_size, local_size](cl::sycl::handler & cgh) mutable {

                            //auto buffer_acc = buffer.get_access(cgh);
                            auto buffer_acc = detail::get_device_acc(cgh, buffer, begin);
                            auto kernel = [=] (cl::sycl::nd_item<1> idx) mutable {
                                detail::device_iterator<value_type> it = detail::get_device_it<value_type>(buffer_acc, idx.get_global_linear_id());
                                auto t = hpx::util::make_tuple(offset, it, chunk_size);
                                hpx::util::invoke(_f, t);
                            };

                            cgh.parallel_for<class kernel_name>(
                                cl::sycl::nd_range<1>(cl::sycl::range<1>(data_count), cl::sycl::range<1>(local_size)),
                                kernel
                            );
                        });
                        queue_->wait();
                        auto data = buffer.get_access();
                        std::copy(data.get_pointer(), data.get_pointer() + data_count, begin);
                    };
                    //command_group();
                    //auto data = buffer.get_access();
                    //std::cout << *data.get_pointer() << " " << data_count << std::endl;
                    //std::copy(data.get_pointer(), data.get_pointer() + data_count, begin);
					results.push_back(hpx::async(launch::async, command_group, ts...));
/*
                    std::size_t data_count = std::get<1>(elem);
					std::size_t chunk_size = std::get<2>(elem);
					std::size_t threads_to_run = data_count / chunk_size;
					std::size_t last_thread_chunk = data_count - (threads_to_run - 1)*chunk_size;

					F _f( std::move(f) );

					auto kernelSubmit = [_f, &sycl_buffer, data_count, chunk_size, threads_to_run, last_thread_chunk]() {
						sycl_buffer.queue.submit( [_f, &sycl_buffer, data_count, chunk_size, threads_to_run, last_thread_chunk](cl::sycl::handler & cgh) {
							
							buffer_view_type buffer_view = 
								(*sycl_buffer.buffer.get()).template get_access<cl::sycl::access::mode::read_write>(cgh);
							auto syclKernel = [=] (cl::sycl::id<1> index) {	
								if (true) {
									// This works with all tests. Type of tuple: <const buffer_view_type *, std::size_t, std::size_t>
									// Test 3 shows that hardcoded '1' is passed correctly.
									auto _x = std::make_tuple(&buffer_view, index[0], 1);

									// This doesn't. Obviously, x = 0 means that no work is done.
									// Together with test 3 it proves that the value of last element in tuple is passed incorrectly (same random value on each thread).
									// auto _x = std::make_tuple(&buffer_view, index[0], x);

									// This is what I want to obtain. Test 1 ends with a segfault, because the address is very incorrect - test 2 proves that
									// auto _x = std::make_tuple(&buffer_view, index[0] + x, 1);
									//auto _x = std::make_tuple(&buffer_view, index[0] * chunk_size, index[0] != static_cast<int>(threads_to_run - 1) ? chunk_size : last_thread_chunk);

									_f(_x);
									//for(int i = 0; i < 10; ++i)
									//buffer_view[ 0 ] = 1;//index[0];
								} else {

									// This would show that x has a inproper value here! 1 is written correctly
									//buffer_view[ index[0] ] = 1;
									//buffer_view[ index[0] ] = x;
								}
							};
							cgh.parallel_for<kernel_name>(cl::sycl::range<1>(data_count), syclKernel);

						});
					};

					results.push_back(hpx::async(launch::async, kernelSubmit));*/
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


		template <typename F, typename Parameters,typename Shape, typename GPUBuffer>
		static std::vector<hpx::future<
			typename detail::bulk_async_execute_result<F, Shape>::type
		> >
		bulk_async_execute(Parameters & params, F && f, Shape const& shape, GPUBuffer & sycl_buffer)
		{
			typedef typename detail::bulk_async_execute_result<F, Shape>::type result_type;
			typedef typename GPUBuffer::buffer_view_type buffer_view_type;
			using kernel_name = typename hpx::parallel::get_kernel_name<F, Parameters>::kernel_name;

			std::vector<hpx::future<result_type> > results;
			try {
				for (auto const& elem: shape) {
					std::size_t data_count = std::get<1>(elem);
					std::size_t chunk_size = std::get<2>(elem);
					std::size_t threads_to_run = data_count / chunk_size;
					std::size_t last_thread_chunk = data_count - (threads_to_run - 1)*chunk_size;

					F _f( std::move(f) );

					auto kernelSubmit = [_f, &sycl_buffer, data_count, chunk_size, threads_to_run, last_thread_chunk]() {
						sycl_buffer.queue.submit( [_f, &sycl_buffer, data_count, chunk_size, threads_to_run, last_thread_chunk](cl::sycl::handler & cgh) {
							
							buffer_view_type buffer_view = 
								(*sycl_buffer.buffer.get()).template get_access<cl::sycl::access::mode::read_write>(cgh);
							auto syclKernel = [=] (cl::sycl::id<1> index) {	
								if (true) {
									// This works with all tests. Type of tuple: <const buffer_view_type *, std::size_t, std::size_t>
									// Test 3 shows that hardcoded '1' is passed correctly.
									auto _x = std::make_tuple(&buffer_view, index[0], 1);

									// This doesn't. Obviously, x = 0 means that no work is done.
									// Together with test 3 it proves that the value of last element in tuple is passed incorrectly (same random value on each thread).
									// auto _x = std::make_tuple(&buffer_view, index[0], x);

									// This is what I want to obtain. Test 1 ends with a segfault, because the address is very incorrect - test 2 proves that
									// auto _x = std::make_tuple(&buffer_view, index[0] + x, 1);
									//auto _x = std::make_tuple(&buffer_view, index[0] * chunk_size, index[0] != static_cast<int>(threads_to_run - 1) ? chunk_size : last_thread_chunk);

									_f(_x);
									//for(int i = 0; i < 10; ++i)
									//buffer_view[ 0 ] = 1;//index[0];
								} else {

									// This would show that x has a inproper value here! 1 is written correctly
									//buffer_view[ index[0] ] = 1;
									//buffer_view[ index[0] ] = x;
								}
							};
							cgh.parallel_for<kernel_name>(cl::sycl::range<1>(data_count), syclKernel);

						});
					};

					results.push_back(hpx::async(launch::async, kernelSubmit));
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

		template <typename F, typename Parameters, typename Shape, typename GPUBuffer>
		static typename detail::bulk_execute_result<F, Shape>::type
		bulk_execute(Parameters & params, F && f, Shape const& shape, GPUBuffer & sycl_buffer)
		{
			typedef typename GPUBuffer::buffer_view_type buffer_view_type;
			using kernel_name = typename hpx::parallel::get_kernel_name<F, Parameters>::kernel_name;

			/**
			 * The elements of pair are:
			 * begin at array, # of elements to process
			 */
			for(auto const & elem : shape) {
				F _f( std::move(f) );

				std::size_t data_count = std::get<1>(elem);
				std::size_t chunk_size = std::get<2>(elem);
				
				std::size_t threads_to_run = data_count / chunk_size;
				std::size_t last_thread_chunk = data_count - (threads_to_run - 1)*chunk_size;

				sycl_buffer.queue.submit( [_f, &sycl_buffer, threads_to_run, last_thread_chunk, data_count, chunk_size](cl::sycl::handler & cgh) {

					buffer_view_type buffer_view = 
						(*sycl_buffer.buffer.get()).template get_access<cl::sycl::access::mode::read_write>(cgh);

					cgh.parallel_for<kernel_name>(cl::sycl::range<1>(data_count),
						[=] (cl::sycl::id<1> index)
						{	
							if (true) {
								// This works with all tests. Type of tuple: <const buffer_view_type *, std::size_t, std::size_t>
								// Test 3 shows that hardcoded '1' is passed correctly.
								auto _x = std::make_tuple(&buffer_view, index[0], 1);

								// This doesn't. Obviously, x = 0 means that no work is done.
								// Together with test 3 it proves that the value of last element in tuple is passed incorrectly (same random value on each thread).
								// auto _x = std::make_tuple(&buffer_view, index[0], x);

								// This is what I want to obtain. Test 1 ends with a segfault, because the address is very incorrect - test 2 proves that
								//auto _x = std::make_tuple(&buffer_view, index[0] * chunk_size, index[0] != static_cast<int>(threads_to_run - 1) ? chunk_size : last_thread_chunk);

								_f(_x);

							} else {

								// This would show that x has a inproper value here! 1 is written correctly
								//buffer_view[ index[0] ] = 1;
								//buffer_view[ index[0] ] = data_count;
							}
							
						});
				});
                sycl_buffer.queue.wait();
             }
           }

            template <typename F, typename Parameters, typename GPUBuffer, typename GPUBuffer2, typename GPUBuffer3>
	        static void bulk_execute(Parameters & params, F && f, std::size_t data_count, std::size_t chunk_size, GPUBuffer & sycl_buffer,GPUBuffer2 & sycl_buffer2,GPUBuffer3 & sycl_buffer3)
	        {
		        typedef typename GPUBuffer::buffer_view_type buffer_view_type;
		        using kernel_name = typename hpx::parallel::get_kernel_name<F, Parameters>::kernel_name;

		        /**
		         * The elements of pair are:
		         * begin at array, # of elements to process
		         */
		        //for(auto const & elem : shape) {
			        F _f( std::move(f) );

			
			        std::size_t threads_to_run = data_count / chunk_size;
			        std::size_t last_thread_chunk = data_count - (threads_to_run - 1)*chunk_size;
                    //std::cout << "Runs: " << data_count << std::endl;

			        sycl_buffer.queue.submit( [_f, &sycl_buffer,&sycl_buffer2,&sycl_buffer3, threads_to_run, last_thread_chunk, data_count, chunk_size](cl::sycl::handler & cgh) {

				        buffer_view_type buffer_view = 
					        (*sycl_buffer.buffer.get()).template get_access<cl::sycl::access::mode::read_write>(cgh);
				        buffer_view_type buffer_view2 = 
					        (*sycl_buffer2.buffer.get()).template get_access<cl::sycl::access::mode::read_write>(cgh);
				        buffer_view_type buffer_view3 = 
					        (*sycl_buffer3.buffer.get()).template get_access<cl::sycl::access::mode::read_write>(cgh);

				        cgh.parallel_for<kernel_name>(cl::sycl::range<1>(data_count),
					        [=] (cl::sycl::id<1> index)
					        {	
						        if (true) {
							        // This works with all tests. Type of tuple: <const buffer_view_type *, std::size_t, std::size_t>
							        // Test 3 shows that hardcoded '1' is passed correctly.
							        //auto _x = std::make_tuple();

							        // This doesn't. Obviously, x = 0 means that no work is done.
							        // Together with test 3 it proves that the value of last element in tuple is passed incorrectly (same random value on each thread).
							        // auto _x = std::make_tuple(&buffer_view, index[0], x);

							        // This is what I want to obtain. Test 1 ends with a segfault, because the address is very incorrect - test 2 proves that
							        //auto _x = std::make_tuple(&buffer_view, index[0] * chunk_size, index[0] != static_cast<int>(threads_to_run - 1) ? chunk_size : last_thread_chunk);

							        _f(index[0], 1,&buffer_view,&buffer_view2,&buffer_view3);

						        } else {

							        // This would show that x has a inproper value here! 1 is written correctly
							        //buffer_view[ index[0] ] = 1;
							        //buffer_view[ index[0] ] = data_count;
						        }
						
					        });
			        });
                    sycl_buffer.queue.wait();
		        }

            template <typename F, typename Parameters, typename GPUBuffer, typename GPUBuffer2>
	        static void bulk_execute(Parameters & params, F && f, std::size_t data_count, std::size_t chunk_size, GPUBuffer & sycl_buffer,GPUBuffer2 & sycl_buffer2)
	        {
		        typedef typename GPUBuffer::buffer_view_type buffer_view_type;
		        using kernel_name = typename hpx::parallel::get_kernel_name<F, Parameters>::kernel_name;

		        /**
		         * The elements of pair are:
		         * begin at array, # of elements to process
		         */
		        //for(auto const & elem : shape) {
			        F _f( std::move(f) );

			
			        std::size_t threads_to_run = data_count / chunk_size;
			        std::size_t last_thread_chunk = data_count - (threads_to_run - 1)*chunk_size;
                    //std::cout << "Runs: " << data_count << std::endl;
			        sycl_buffer.queue.submit( [_f, &sycl_buffer,&sycl_buffer2, threads_to_run, last_thread_chunk, data_count, chunk_size](cl::sycl::handler & cgh) {

				        buffer_view_type buffer_view = 
					        (*sycl_buffer.buffer.get()).template get_access<cl::sycl::access::mode::read_write>(cgh);
				        buffer_view_type buffer_view2 = 
					        (*sycl_buffer2.buffer.get()).template get_access<cl::sycl::access::mode::read_write>(cgh);

				        cgh.parallel_for<kernel_name>(cl::sycl::range<1>(data_count),
					        [=] (cl::sycl::id<1> index)
					        {	
						        if (true) {
							        // This works with all tests. Type of tuple: <const buffer_view_type *, std::size_t, std::size_t>
							        // Test 3 shows that hardcoded '1' is passed correctly.
							        //auto _x = std::make_tuple();

							        // This doesn't. Obviously, x = 0 means that no work is done.
							        // Together with test 3 it proves that the value of last element in tuple is passed incorrectly (same random value on each thread).
							        // auto _x = std::make_tuple(&buffer_view, index[0], x);

							        // This is what I want to obtain. Test 1 ends with a segfault, because the address is very incorrect - test 2 proves that
							        //auto _x = std::make_tuple(&buffer_view, index[0] * chunk_size, index[0] != static_cast<int>(threads_to_run - 1) ? chunk_size : last_thread_chunk);

							        _f(index[0], 1,&buffer_view,&buffer_view2);

						        } else {

							        // This would show that x has a inproper value here! 1 is written correctly
							        //buffer_view[ index[0] ] = 1;
							        //buffer_view[ index[0] ] = data_count;
						        }
						
					        });
			        });
                    sycl_buffer.queue.wait();
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
        struct is_executor<gpu_sycl_executor>
          : std::true_type
        {};
        /// \endcond
    }
}}}

#endif
