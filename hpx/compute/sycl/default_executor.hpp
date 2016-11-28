//  Copyright (c) 2016 Marcin Copik
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_SYCL_DEFAULT_EXECUTOR_HPP
#define HPX_COMPUTE_SYCL_DEFAULT_EXECUTOR_HPP

#include <hpx/config.hpp>

#include <hpx/compute/sycl/detail/launch.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/traits/is_executor.hpp>
#include <hpx/traits/is_iterator.hpp>
#include <hpx/util/decay.hpp>
#include <hpx/util/invoke.hpp>
#include <hpx/util/tuple.hpp>

#include <algorithm>
#include <cstddef>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>


namespace hpx { namespace compute { namespace sycl
{

    namespace detail
    {
        template<typename Iter, typename Name>
        struct launcher_helper
        {
            template<typename F, typename ... Args>
            static void execute(target const& t, int local_size, F && f, Iter begin, int chunk_size, int offset, Args &&... args)
            {
                typedef typename std::iterator_traits<Iter>::value_type value_type;

                F _f = std::move(f);
                auto f1 = [=](size_t idx, cl::sycl::global_ptr<value_type> ptr) mutable {
                                //ptr += pos + idx.get_global_linear_id();
                                auto t = hpx::util::make_tuple(ptr + idx, 1, offset);
                                hpx::util::invoke(_f, t);
                            };
                detail::launch<Name>(t, chunk_size, local_size, f1, begin.device_ptr(), std::forward<Args>(args)...);
            }
        };
    }


    struct default_executor : hpx::parallel::executor_tag
    {
        // By default, this executor relies on a special executor parameters
        // implementation which knows about the specifics of creating the
        // bulk-shape ranges for the accelerator.
        //typedef default_executor_parameters executor_parameters_type;
        default_executor(sycl::target const& target)
          : target_(target)
        {}

        template <typename F, typename ... Args>
        void apply_execute(F && f, Args &&... args) const
        {
            detail::launch(target_, 1, std::forward<F>(f), std::forward<Args>(args)...);
        }

        template <typename F, typename ... Args>
        hpx::future<void> async_execute(F && f, Args &&... args) const
        {
            apply_execute(std::forward<F>(f), std::forward<Args>(args)...);
            return target_.get_future();
        }

        template <typename F, typename ... Args>
        void execute(F && f, Args &&... args) const
        {
            apply_execute(std::forward<F>(f), std::forward<Args>(args)...);
            target_.synchronize();
        }

        template <typename F, typename Shape, typename ... Args>
        void bulk_launch(F && f, Shape const& shape, Args &&... args) const
        {
            for (auto const& elem: shape)
            {
                auto begin = hpx::util::get<0>(elem);
                std::size_t chunk_size = hpx::util::get<1>(elem);
                std::size_t offset = hpx::util::get<2>(elem);
                detail::launcher_helper<decltype(begin), class Name>::execute(target_, 32,
                                                                              std::forward<F>(f), begin, chunk_size, offset, std::forward<Args>(args)...);
            }
        }

        template <typename F, typename Shape, typename ... Args>
        std::vector<hpx::future<void> >
        bulk_async_execute(F && f, Shape const& shape, Args &&... args) const
        {
            std::cout << "Launch!" << std::endl;
            bulk_launch(std::forward<F>(f), shape, std::forward<Args>(args)...);
            std::vector<hpx::future<void> > result;
            result.push_back(target_.get_future());
            return result;
        }

        template <typename F, typename Shape, typename ... Args>
        void bulk_execute(F && f, Shape const& shape, Args &&... args) const
        {
            std::cout << "Launch!" << std::endl;
            bulk_launch(std::forward<F>(f), shape, std::forward<Args>(args)...);
            target_.synchronize();
        }

        sycl::target& target()
        {
            return target_;
        }

        sycl::target const& target() const
        {
            return target_;
        }

    private:
        sycl::target target_;
    };
}}}

#endif
