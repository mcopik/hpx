//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_HC_DEFAULT_EXECUTOR_HPP
#define HPX_COMPUTE_HC_DEFAULT_EXECUTOR_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_HC)

#include <hpx/traits/is_executor.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>

#include <hpx/compute/vector.hpp>
#include <hpx/compute/hc/allocator.hpp>
#include <hpx/compute/hc/target.hpp>
#include <hpx/compute/hc/detail/launch.hpp>

#include <algorithm>
#include <utility>
#include <string>
#include <vector>

#include <boost/range/functions.hpp>

namespace hpx { namespace compute { namespace hc
{
    struct default_executor : hpx::parallel::executor_tag
    {
        default_executor(hc::target& target)
          : target_(target)
        {}

        template <typename F, typename ... Ts>
        void apply_execute(F && f, Ts &&... ts)
        {
            detail::launch(target_, 1, 1,
                std::forward<F>(f), std::forward<Ts>(ts)...);
        }

        template <typename F, typename ... Ts>
        hpx::future<void> async_execute(F && f, Ts &&... ts)
        {
            apply_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
            return target_.get_future();
        }

        template <typename F, typename ... Ts>
        void execute(F && f, Ts &&... ts)
        {
            apply_execute(std::forward<F>(f), std::forward<Ts>(ts)...);
            target_.synchronize();
        }

        std::size_t processing_units_count()
        {
            // well... std::rand()?
            return 4;
        }

        template <typename F, typename Shape, typename ... Ts>
        void bulk_launch(F && f, Shape const& shape, Ts &&... ts) const
        {
            typedef typename boost::range_const_iterator<Shape>::type
                iterator_type;
            typedef typename std::iterator_traits<iterator_type>::value_type
                value_type;
            for (auto const& s: shape)
            {
                //iterator to GPU data
                auto begin = hpx::util::get<0>(s);
                std::size_t chunk_size = hpx::util::get<1>(s);
                std::size_t base_idx = hpx::util::get<2>(s);

                typedef typename
                    std::iterator_traits<decltype(begin)>::value_type
                    data_type;

                // FIXME: make the 1024 to be configurable...
                int threads_per_block =
                    (std::min)(1024, static_cast<int>(chunk_size));
                int num_blocks = static_cast<int>(
                    (chunk_size + threads_per_block - 1) / threads_per_block);

                detail::launch(
                    target_, num_blocks, threads_per_block,
                    [chunk_size, f]
                    HPX_DEVICE_LAMBDA(int idx, const target_ptr<data_type> & ptr,
                        Ts&... ts)
                    {
                        hpx::util::invoke(f, value_type(ptr + idx, 1, idx),
                            ts...);
                    },
                    begin.device_ptr(), std::forward<Ts>(ts)...
                );
            }
        }

        template <typename F, typename Shape, typename ... Ts>
        std::vector<hpx::future<void> >
        bulk_async_execute(F && f, Shape const& shape, Ts &&... ts)
        {
            bulk_launch(std::forward<F>(f), shape, std::forward<Ts>(ts)...);

            std::vector<hpx::future<void>> result;
            result.push_back(target_.get_future());
            return result;
        }

        template <typename F, typename Shape, typename ... Ts>
        void bulk_execute(F && f, Shape const& shape, Ts &&... ts)
        {
            bulk_launch(std::forward<F>(f), shape, std::forward<Ts>(ts)...);
            target_.synchronize();
        }

    private:
        hc::target& target_;
    };
}}}

#endif
#endif
