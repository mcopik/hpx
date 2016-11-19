//  Copyright (c) 2007-2016 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_FOREACH_PARTITIONER_OCT_03_2014_0112PM)
#define HPX_PARALLEL_UTIL_FOREACH_PARTITIONER_OCT_03_2014_0112PM

#include <hpx/config.hpp>
#include <hpx/dataflow.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/tuple.hpp>

#include <hpx/parallel/algorithms/detail/predicates.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/executors/executor_parameter_traits.hpp>
#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/traits/extract_partitioner.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/util/detail/partitioner_iteration.hpp>
#include <hpx/parallel/util/detail/scoped_executor_parameters.hpp>

#include <boost/exception_ptr.hpp>

#include <algorithm>
#include <cstddef>
#include <list>
#include <memory>
#include <utility>
#include <vector>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // The static partitioner simply spawns one chunk of iterations for
        // each available core.
        template <typename ExPolicy_, typename Result = void>
        struct foreach_static_partitioner
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2>
            static FwdIter call(ExPolicy && policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2)
            {
                typedef typename hpx::util::decay<ExPolicy>::type::executor_type
                    executor_type;
                typedef hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                typedef typename hpx::util::tuple<FwdIter, std::size_t> tuple;

                typedef typename
                    hpx::util::decay<ExPolicy>::type::executor_parameters_type
                    parameters_type;
                typedef executor_parameter_traits<parameters_type>
                    parameters_traits;

                // inform parameter traits
                scoped_executor_parameters<parameters_type> scoped_param(
                    policy.parameters());

                FwdIter last = parallel::v1::detail::next(first, count);

                std::vector<hpx::future<Result> > inititems, workitems;
                std::list<boost::exception_ptr> errors;

                try {
                    // estimates a chunk size based on number of cores used
                    typedef typename parameters_traits::has_variable_chunk_size
                        has_variable_chunk_size;

                    auto shapes =
                        get_bulk_iteration_shape_idx(
                            policy, inititems, f1, first, count, 1,
                            has_variable_chunk_size());

                    workitems = executor_traits::bulk_async_execute(
                        policy.executor(), policy.parameters(),
                        partitioner_iteration<Result, F1>{std::forward<F1>(f1)},
                        std::move(shapes));
                }
                catch (...) {
                    handle_local_exceptions<ExPolicy>::call(
                        boost::current_exception(), errors);
                }

                // wait for all tasks to finish
                hpx::wait_all(inititems, workitems);

                // handle exceptions
                handle_local_exceptions<ExPolicy>::call(inititems, errors);
                handle_local_exceptions<ExPolicy>::call(workitems, errors);

                try {
                    return f2(std::move(last));
                }
                catch (...) {
                    // rethrow either bad_alloc or exception_list
                    handle_local_exceptions<ExPolicy>::call(
                        boost::current_exception());
                }
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct foreach_static_partitioner<parallel_task_execution_policy, Result>
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2>
            static hpx::future<FwdIter> call(ExPolicy && policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2)
            {
                typedef typename hpx::util::decay<ExPolicy>::type::executor_type
                    executor_type;
                typedef hpx::parallel::executor_traits<executor_type>
                    executor_traits;

                typedef typename
                    hpx::util::decay<ExPolicy>::type::executor_parameters_type
                    parameters_type;
                typedef executor_parameter_traits<parameters_type>
                    parameters_traits;

                typedef scoped_executor_parameters<parameters_type>
                    scoped_executor_parameters;

                // inform parameter traits
                std::shared_ptr<scoped_executor_parameters>
                    scoped_param(std::make_shared<
                            scoped_executor_parameters
                        >(policy.parameters()));

                FwdIter last = parallel::v1::detail::next(first, count);

                std::vector<hpx::future<Result> > inititems, workitems;
                std::list<boost::exception_ptr> errors;

                try {
                    // estimates a chunk size based on number of cores used
                    typedef typename parameters_traits::has_variable_chunk_size
                        has_variable_chunk_size;

                    auto shapes =
                        get_bulk_iteration_shape_idx(
                            policy, inititems, f1, first, count, 1,
                            has_variable_chunk_size());

                    workitems = executor_traits::bulk_async_execute(
                        policy.executor(), policy.parameters(),
                        partitioner_iteration<Result, F1>{std::forward<F1>(f1)},
                        std::move(shapes));
                }
                catch (std::bad_alloc const&) {
                    return hpx::make_exceptional_future<FwdIter>(
                        boost::current_exception());
                }
                catch (...) {
                    errors.push_back(boost::current_exception());
                }

                // wait for all tasks to finish
                return hpx::dataflow(
                    [last, errors, scoped_param, f2](
                            std::vector<hpx::future<Result> > && r1,
                            std::vector<hpx::future<Result> > && r2) mutable
                    ->  FwdIter
                    {
                        handle_local_exceptions<ExPolicy>::call(r1, errors);
                        handle_local_exceptions<ExPolicy>::call(r2, errors);

                        return f2(std::move(last));
                    },
                    std::move(inititems), std::move(workitems));
            }
        };

#if defined(HPX_WITH_AMP) || defined(HPX_WITH_SYCL)
        ///////////////////////////////////////////////////////////////////////
/*		template <typename Result>
		struct foreach_n_static_partitioner<gpu_execution_policy, Result>
		{
			template <typename ExPolicy, typename FwdIter, typename F1, typename GPUBuffer>
			static FwdIter call(ExPolicy policy,
				FwdIter first, std::size_t count, F1 && f1, GPUBuffer & buffer,
				std::size_t chunk_size)
			{
				typedef typename ExPolicy::executor_type executor_type;
				typedef typename hpx::parallel::executor_traits<executor_type>
					executor_traits;
				typedef typename GPUBuffer::buffer_view_type buffer_view;
                typedef typename ExPolicy::executor_parameters_type parameters_type;
                typedef executor_parameter_traits<parameters_type> traits;

				FwdIter last = first;
				std::advance(last, count);

				std::vector<hpx::future<Result> > inititems, workitems;
				std::list<boost::exception_ptr> errors;

				try {

					std::size_t chunk_size = traits::get_chunk_size(policy.parameters(), policy.executor(),
                        [](){ return 0; }, count);
					// 0 when no parameter is specified
					chunk_size = chunk_size == 0 ? 1 : chunk_size;
                    chunk_size = std::min(chunk_size, count);
					std::vector< std::tuple<const buffer_view *, std::size_t, std::size_t> > shape{ std::make_tuple(nullptr, count, chunk_size) };

					/**
					 * Wrap the GPU lambda - the new functor will take a pair of two ints as an argument,
					 * one of them will point to the starting index and the second one will give the size.
					 *
					 * The wrapping is necessary, otherwise the code in executor_traits will not be able to
					 * correcly detect the return type of this lambda
					 */
/*					F1 _f1 = std::move(f1);
					auto f = hpx::parallel::wrap_kernel(f1, [_f1](std::tuple<const buffer_view *, std::size_t, std::size_t> const& elem)
					{
						/**
						 *	Test 1 : Run function defined in parallel/algorithms/for_each
						 *	HPX - doesn't work, segfault
						 *	No HPX - doesn't work, segfault
						 */
	//					_f1(std::get<0>(elem), std::get<1>(elem), std::get<2>(elem) );

						/**
						 *	Test 2 : Each thread tries to write its ID into each element of the buffer (Writer After Write).
						 *  Notice that the length of an array is hard-coded.
						 */
						//for(std::size_t i = 0;i < 10;++i)
						//	(*std::get<0>(elem))[ i ] = std::get<1>(elem);

						/**
						 *	Test 3 : Each thread tries to write the array position into the element of array specified by index.
						 *	HPX - instead of array position shows the same random value
						 *	No HPX - instead of array position shows the same random value
						 */
						//(*std::get<0>(elem))[ std::get<1>(elem) ] = std::get<2>(elem);
		//			});

					//workitems.reserve(shape.size());
					//workitems = executor_traits::async_execute(
					//	policy.executor(), f, shape);

			/*		executor_traits::execute(policy.executor(),
							//std::forward<decltype(f)>(f),
							policy.parameters(),
							std::move(f),
							shape, buffer);
				}
                catch (...) {
                	detail::handle_local_exceptions<ExPolicy>::call(
                        boost::current_exception(), errors);
                }

                // wait for all tasks to finish
                hpx::wait_all(workitems);
                hpx::wait_all(inititems);

                // handle exceptions
                detail::handle_local_exceptions<ExPolicy>::call(
                    workitems, errors);
                detail::handle_local_exceptions<ExPolicy>::call(
                    inititems, errors);

                return last;
			}
		};

		template <typename Executor, typename Parameters, typename Result>
        struct foreach_n_static_partitioner<
                gpu_execution_policy_shim<Executor, Parameters>,
                Result>
          : foreach_n_static_partitioner<gpu_execution_policy, Result>
        {};

		template <typename Result>
		struct foreach_n_static_partitioner<gpu_task_execution_policy, Result>
		{
			template <typename ExPolicy, typename FwdIter, typename F1, typename GPUBuffer>
			static hpx::future<FwdIter> call(ExPolicy policy,
				FwdIter first, std::size_t count, F1 && f1, GPUBuffer & buffer,
				std::size_t chunk_size)
			{
				typedef typename ExPolicy::executor_type executor_type;
				typedef typename hpx::parallel::executor_traits<executor_type>
					executor_traits;
				typedef typename GPUBuffer::buffer_view_type buffer_view;
				typedef typename ExPolicy::executor_parameters_type parameters_type;
				typedef executor_parameter_traits<parameters_type> traits;

				FwdIter last = first;
				std::advance(last, count);

				std::vector<hpx::future<Result> > inititems, workitems;
				std::list<boost::exception_ptr> errors;

				try {
					std::size_t chunk_size = traits::get_chunk_size(policy.parameters(), policy.executor(),
						[](){ return 0; }, count);
                    chunk_size = std::min(chunk_size, count);
					std::vector< std::tuple<const buffer_view *, std::size_t, std::size_t> > shape{ std::make_tuple(nullptr, count, chunk_size) };

					/**
					 * Wrap the GPU lambda - the new functor will take a pair of two ints as an argument,
					 * one of them will point to the starting index and the second one will give the size.
					 */
				/*	F1 _f1 = std::move(f1);
					auto f = hpx::parallel::wrap_kernel(f1, [_f1](std::tuple<const buffer_view *, std::size_t, std::size_t> const& elem)
					{
						return _f1(std::get<0>(elem), std::get<1>(elem), std::get<2>(elem));
					});

					workitems.reserve(shape.size());

					workitems = executor_traits::async_execute(
						policy.executor(), policy.parameters(),
						std::forward<decltype(f)>(f),
						shape, buffer);
				}
				catch (std::bad_alloc const&) {
					return hpx::make_exceptional_future<FwdIter>(
						boost::current_exception());
				}
				catch (...) {
					errors.push_back(boost::current_exception());
				}

				// wait for all tasks to finish
				return hpx::lcos::local::dataflow(
					[last, errors](std::vector<hpx::future<Result> > && r1,
							std::vector<hpx::future<Result> > && r2)
						mutable -> FwdIter
					{
						detail::handle_local_exceptions<ExPolicy>::call(r1, errors);
						detail::handle_local_exceptions<ExPolicy>::call(r2, errors);
						return last;
					},
					std::move(inititems), std::move(workitems));
			}
		};

		template <typename Executor, typename Parameters, typename Result>
        struct foreach_n_static_partitioner<
                gpu_task_execution_policy_shim<Executor, Parameters>,
                Result>
          : foreach_n_static_partitioner<gpu_task_execution_policy, Result>
        {};*/
#endif

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_static_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                Result>
          : foreach_static_partitioner<parallel_task_execution_policy, Result>
        {};

        ///////////////////////////////////////////////////////////////////////
        // ExPolicy: execution policy
        // Result:   intermediate result type of first step (default: void)
        // PartTag:  select appropriate partitioner
        template <typename ExPolicy, typename Result, typename PartTag>
        struct foreach_partitioner;

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy_, typename Result>
        struct foreach_partitioner<ExPolicy_, Result,
            parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2>
            static FwdIter call(ExPolicy && policy, FwdIter first,
                std::size_t count, F1 && f1, F2 && f2)
            {
                return foreach_static_partitioner<
                        typename hpx::util::decay<ExPolicy>::type, Result
                    >::call(
                        std::forward<ExPolicy>(policy), first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2));
            }
        };

        template <typename Result>
        struct foreach_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2>
            static hpx::future<FwdIter> call(ExPolicy && policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2)
            {
                return foreach_static_partitioner<
                        typename hpx::util::decay<ExPolicy>::type, Result
                    >::call(
                        std::forward<ExPolicy>(policy), first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2));
            }
        };

#if defined(HPX_WITH_GPU_EXECUTOR)
/*		template <typename Result>
		struct foreach_n_partitioner<gpu_execution_policy, Result,
				parallel::traits::static_partitioner_tag>
		{
			template <typename ExPolicy, typename FwdIter, typename F1, typename GPUBuffer>
			static FwdIter call(ExPolicy policy,
				FwdIter first, std::size_t count, F1 && f1, GPUBuffer & buffer,
				std::size_t chunk_size = 0)
			{
				return foreach_n_static_partitioner<gpu_execution_policy, Result>::call(
					policy, first, count, std::forward<F1>(f1), buffer, chunk_size);
			}
		};

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_n_partitioner<
                gpu_execution_policy_shim<Executor, Parameters>,
                Result, parallel::traits::static_partitioner_tag>
          : foreach_n_partitioner<gpu_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_n_partitioner<
                gpu_execution_policy_shim<Executor, Parameters>,
                Result, parallel::traits::auto_partitioner_tag>
          : foreach_n_partitioner<gpu_execution_policy, Result,
                parallel::traits::auto_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_n_partitioner<
                gpu_execution_policy_shim<Executor, Parameters>,
                Result, parallel::traits::default_partitioner_tag>
          : foreach_n_partitioner<gpu_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {};

		template <typename Result>
		struct foreach_n_partitioner<gpu_task_execution_policy, Result,
				parallel::traits::static_partitioner_tag>
		{
			template <typename ExPolicy, typename FwdIter, typename F1, typename GPUBuffer>
			static hpx::future<FwdIter> call(ExPolicy policy,
				FwdIter first, std::size_t count, F1 && f1, GPUBuffer & buffer,
				std::size_t chunk_size = 0)
			{
				return foreach_n_static_partitioner<ExPolicy, Result>::call(
					policy, first, count, std::forward<F1>(f1), buffer, chunk_size);
			}
		};

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_n_partitioner<
                gpu_task_execution_policy_shim<Executor, Parameters>,
                Result, parallel::traits::static_partitioner_tag>
          : foreach_n_partitioner<gpu_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_n_partitioner<
                gpu_task_execution_policy_shim<Executor, Parameters>,
                Result, parallel::traits::auto_partitioner_tag>
          : foreach_n_partitioner<gpu_task_execution_policy, Result,
                parallel::traits::auto_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_n_partitioner<
                gpu_task_execution_policy_shim<Executor, Parameters>,
                Result, parallel::traits::default_partitioner_tag>
          : foreach_n_partitioner<gpu_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {};*/
#endif
#if defined(HPX_HAVE_DATAPAR)
        template <typename Result>
        struct foreach_partitioner<datapar_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename F1,
                typename F2>
            static hpx::future<FwdIter> call(ExPolicy && policy,
                FwdIter first, std::size_t count, F1 && f1, F2 && f2)
            {
                return foreach_static_partitioner<
                        parallel_task_execution_policy, Result
                    >::call(
                        std::forward<ExPolicy>(policy), first, count,
                        std::forward<F1>(f1), std::forward<F2>(f2));
            }
        };
#endif

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                Result, parallel::traits::static_partitioner_tag>
          : foreach_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                Result, parallel::traits::auto_partitioner_tag>
          : foreach_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::auto_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                Result, parallel::traits::default_partitioner_tag>
          : foreach_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {};

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Result>
        struct foreach_partitioner<ExPolicy, Result,
                parallel::traits::default_partitioner_tag>
          : foreach_partitioner<ExPolicy, Result,
                parallel::traits::static_partitioner_tag>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Result = void,
        typename PartTag = typename parallel::traits::extract_partitioner<
            typename hpx::util::decay<ExPolicy>::type
        >::type>
    struct foreach_partitioner
      : detail::foreach_partitioner<
            typename hpx::util::decay<ExPolicy>::type, Result, PartTag>
    {};
}}}

#endif
