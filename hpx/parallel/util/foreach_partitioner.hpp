//  Copyright (c) 2007-2015 Hartmut Kaiser
//  Copyright (c) 2015 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_PARALLEL_UTIL_FOREACH_PARTITIONER_OCT_03_2014_0112PM)
#define HPX_PARALLEL_UTIL_FOREACH_PARTITIONER_OCT_03_2014_0112PM

#include <hpx/hpx_fwd.hpp>
#include <hpx/async.hpp>
#include <hpx/exception_list.hpp>
#include <hpx/lcos/wait_all.hpp>
#include <hpx/lcos/local/dataflow.hpp>
#include <hpx/util/bind.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/deferred_call.hpp>
#include <hpx/util/invoke_fused.hpp>
#include <hpx/util/tuple.hpp>

#include <hpx/parallel/executors/executor_traits.hpp>
#include <hpx/parallel/executors/executor_parameter_traits.hpp>
#include <hpx/parallel/execution_policy.hpp>
#include <hpx/parallel/util/detail/chunk_size.hpp>
#include <hpx/parallel/util/detail/handle_local_exceptions.hpp>
#include <hpx/parallel/traits/extract_partitioner.hpp>

#include <algorithm>


#include <hpx/parallel/util/loop.hpp>

///////////////////////////////////////////////////////////////////////////////
namespace hpx { namespace parallel { namespace util
{
    namespace detail
    {
        ///////////////////////////////////////////////////////////////////////
        // The static partitioner simply spawns one chunk of iterations for
        // each available core.
        template <typename ExPolicy, typename Result = void>
        struct foreach_n_static_partitioner
        {
            template <typename FwdIter, typename F1>
            static FwdIter call(ExPolicy policy, FwdIter first,
                std::size_t count, F1 && f1, std::size_t chunk_size)
            {
                typedef typename ExPolicy::executor_type executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                typedef typename hpx::util::tuple<FwdIter, std::size_t> tuple;

                FwdIter last = first;
                std::advance(last, count);

                std::vector<hpx::future<Result> > inititems, workitems;
                std::list<boost::exception_ptr> errors;
                std::vector<tuple> shape;

                try {
                    // estimates a chunk size based on number of cores used
                    shape = get_bulk_iteration_shape(policy, inititems, f1,
                        first, count, chunk_size);

                    workitems.reserve(shape.size());

                    using hpx::util::bind;
                    using hpx::util::functional::invoke_fused;
                    using hpx::util::placeholders::_1;
                    workitems = executor_traits::async_execute(
                        policy.executor(),
                        bind(invoke_fused(), std::forward<F1>(f1), _1),
                        shape);
                }
                catch (...) {
                    detail::handle_local_exceptions<ExPolicy>::call(
                        boost::current_exception(), errors);
                }

                // wait for all tasks to finish
                hpx::wait_all(inititems);
                hpx::wait_all(workitems);

                // handle exceptions
                detail::handle_local_exceptions<ExPolicy>::call(
                    inititems, errors);
                detail::handle_local_exceptions<ExPolicy>::call(
                    workitems, errors);

                return last;
            }
        };

        ///////////////////////////////////////////////////////////////////////
        template <typename Result>
        struct foreach_n_static_partitioner<parallel_task_execution_policy, Result>
        {
            template <typename ExPolicy, typename FwdIter, typename F1>
            static hpx::future<FwdIter> call(ExPolicy policy,
                FwdIter first, std::size_t count, F1 && f1,
                std::size_t chunk_size)
            {
                typedef typename ExPolicy::executor_type executor_type;
                typedef typename hpx::parallel::executor_traits<executor_type>
                    executor_traits;
                typedef typename hpx::util::tuple<FwdIter, std::size_t> tuple;

                FwdIter last = first;
                std::advance(last, count);

                std::vector<hpx::future<Result> > inititems, workitems;
                std::list<boost::exception_ptr> errors;
                std::vector<tuple> shape;

                try {
                    // estimates a chunk size based on number of cores used
                    shape = get_bulk_iteration_shape(policy, inititems, f1,
                        first, count, chunk_size);

                    workitems.reserve(shape.size());

                    using hpx::util::bind;
                    using hpx::util::functional::invoke_fused;
                    using hpx::util::placeholders::_1;
                    workitems = executor_traits::async_execute(
                        policy.executor(),
                        bind(invoke_fused(), std::forward<F1>(f1), _1),
                        shape);
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
                parallel_task_execution_policy_shim<Executor, Parameters>,
                Result>
          : foreach_n_static_partitioner<parallel_task_execution_policy, Result>
        {};

#if defined(HPX_WITH_GPU_EXECUTOR)
        ///////////////////////////////////////////////////////////////////////
		template <typename Result>
		struct foreach_n_static_partitioner<gpu_execution_policy, Result>
		{
			template <typename ExPolicy, typename FwdIter, typename F1>
			static FwdIter call(ExPolicy policy,
				FwdIter first, std::size_t count, F1 && f1,
				std::size_t chunk_size)
			{
				typedef typename ExPolicy::executor_type executor_type;
				typedef typename hpx::parallel::executor_traits<executor_type>
					executor_traits;
                typedef typename ExPolicy::executor_parameters_type parameters_type;
                typedef executor_parameter_traits<parameters_type> traits;
                typedef typename hpx::util::tuple<FwdIter, std::size_t> tuple;

				FwdIter last = first;
				std::advance(last, count);

				std::vector<hpx::future<Result> > inititems, workitems;
				std::list<boost::exception_ptr> errors;

				try {
					std::size_t chunk_size = traits::get_chunk_size(policy.parameters(), policy.executor(), 
                        [](){ return 0; }, count);
                    chunk_size = std::min(chunk_size, count);
					// Tuple: accelerator number, position to start, data count, chunk size for thread
					// std::vector< std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> > shape{ {0, 0, count, chunk_size} };
					//std::vector< std::pair<std::size_t, std::size_t> > shape{ {count, chunk_size} };
                    std::vector< tuple > shape;
                    shape.emplace_back(first, chunk_size);

					/**
					 * Wrap the GPU lambda - the new functor will take a pair of two ints as an argument,
					 * one of them will point to the starting index and the second one will give the size.
					 *
					 * The wrapping is necessary, otherwise the code in executor_traits will not be able to
					 * correcly detect the return type of this lambda
					 */
					F1 _f1 = std::move(f1);				
					auto f = [_f1](hpx::util::tuple<FwdIter, std::size_t> const& elem)
					{
						_f1(hpx::util::get<0>(elem), hpx::util::get<1>(elem));
					};

					//workitems.reserve(shape.size());
					//workitems = executor_traits::async_execute(
					//	policy.executor(), f, shape);

					executor_traits::execute(policy.executor(),
							//std::forward<decltype(f)>(f),
							std::move(f),
							shape);
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
			template <typename ExPolicy, typename FwdIter, typename F1>
			static hpx::future<FwdIter> call(ExPolicy policy,
				FwdIter first, std::size_t count, F1 && f1,
				std::size_t chunk_size)
			{
				typedef typename ExPolicy::executor_type executor_type;
				typedef typename hpx::parallel::executor_traits<executor_type>
					executor_traits;
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
					std::vector< std::pair<std::size_t, std::size_t> > shape{ {count, chunk_size} };

					/**
					 * Wrap the GPU lambda - the new functor will take a pair of two ints as an argument,
					 * one of them will point to the starting index and the second one will give the size.
					 */
					auto f = [f1](std::pair<std::size_t, std::size_t> const& elem)
					{
						return f1(elem.first, elem.second);
					};

					workitems.reserve(shape.size());

					workitems = executor_traits::async_execute(
						policy.executor(),
						std::forward<decltype(f)>(f),
						shape);
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
        {};
#endif

        ///////////////////////////////////////////////////////////////////////
        // ExPolicy: execution policy
        // Result:   intermediate result type of first step (default: void)
        // PartTag:  select appropriate partitioner
        template <typename ExPolicy, typename Result, typename PartTag>
        struct foreach_n_partitioner;

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Result>
        struct foreach_n_partitioner<ExPolicy, Result,
            parallel::traits::static_partitioner_tag>
        {
            template <typename FwdIter, typename F1>
            static FwdIter call(ExPolicy policy, FwdIter first,
                std::size_t count, F1 && f1, std::size_t chunk_size = 0)
            {
                return foreach_n_static_partitioner<ExPolicy, Result>::call(
                    policy, first, count, std::forward<F1>(f1), chunk_size);
            }
        };

        template <typename Result>
        struct foreach_n_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {
            template <typename ExPolicy, typename FwdIter, typename F1>
            static hpx::future<FwdIter> call(ExPolicy policy,
                FwdIter first, std::size_t count, F1 && f1,
                std::size_t chunk_size = 0)
            {
                return foreach_n_static_partitioner<ExPolicy, Result>::call(
                    policy, first, count, std::forward<F1>(f1), chunk_size);
            }
        };

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_n_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                Result, parallel::traits::static_partitioner_tag>
          : foreach_n_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_n_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                Result, parallel::traits::auto_partitioner_tag>
          : foreach_n_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::auto_partitioner_tag>
        {};

        template <typename Executor, typename Parameters, typename Result>
        struct foreach_n_partitioner<
                parallel_task_execution_policy_shim<Executor, Parameters>,
                Result, parallel::traits::default_partitioner_tag>
          : foreach_n_partitioner<parallel_task_execution_policy, Result,
                parallel::traits::static_partitioner_tag>
        {};

#if defined(HPX_WITH_GPU_EXECUTOR)
		template <typename Result>
		struct foreach_n_partitioner<gpu_execution_policy, Result,
				parallel::traits::static_partitioner_tag>
		{
			template <typename ExPolicy, typename FwdIter, typename F1>
			static FwdIter call(ExPolicy policy,
				FwdIter first, std::size_t count, F1 && f1,
				std::size_t chunk_size = 0)
			{
				return foreach_n_static_partitioner<ExPolicy, Result>::call(
					policy, first, count, std::forward<F1>(f1), chunk_size);
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
			template <typename ExPolicy, typename FwdIter, typename F1>
			static hpx::future<FwdIter> call(ExPolicy policy,
				FwdIter first, std::size_t count, F1 && f1,
				std::size_t chunk_size = 0)
			{
				return foreach_n_static_partitioner<ExPolicy, Result>::call(
					policy, first, count, std::forward<F1>(f1), chunk_size);
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
        {};
#endif

        ///////////////////////////////////////////////////////////////////////
        template <typename ExPolicy, typename Result>
        struct foreach_n_partitioner<ExPolicy, Result,
                parallel::traits::default_partitioner_tag>
          : foreach_n_partitioner<ExPolicy, Result,
                parallel::traits::static_partitioner_tag>
        {};
    }

    ///////////////////////////////////////////////////////////////////////////
    template <typename ExPolicy, typename Result = void,
        typename PartTag = typename parallel::traits::extract_partitioner<
            typename hpx::util::decay<ExPolicy>::type
        >::type>
    struct foreach_n_partitioner
      : detail::foreach_n_partitioner<
            typename hpx::util::decay<ExPolicy>::type, Result, PartTag>
    {};
}}}

#endif
