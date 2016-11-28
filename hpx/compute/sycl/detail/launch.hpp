//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_SYCL_DETAIL_LAUNCH_HPP
#define HPX_COMPUTE_SYCL_DETAIL_LAUNCH_HPP


#include <hpx/config.hpp>

#if defined(HPX_HAVE_SYCL)
#include <hpx/compute/sycl/target.hpp>
#include <hpx/util/invoke_fused.hpp>

#include <SYCL/sycl.hpp>
#include <SYCL/accessor.h>

#include <string>
#include <type_traits>
#include <utility>

namespace std
{
    template<typename T>
    struct iterator_traits< cl::sycl::global_ptr<T> >
    {
        typedef T value_type;
        typedef typename cl::sycl::global_ptr<T>::pointer_t pointer;
        typedef typename cl::sycl::global_ptr<T>::reference_t reference;
        typedef ptrdiff_t difference_type;
        typedef std::random_access_iterator_tag iterator_category;
    };
}


namespace hpx { namespace compute { namespace sycl { namespace detail
{

    /*template<typename T, typename Name>
    struct executor_helper<
            hpx::util::zip_iterator<detail::host_iterator<T>, detail::host_iterator<T>>,
            Name
            >
    {
        typedef hpx::util::zip_iterator<detail::host_iterator<T>, detail::host_iterator<T>> Iter;
        template<typename F>
        static void execute(F && f, const Iter & begin, int data_count, int offset, int chunk_size, int global_size, int local_size, cl::sycl::queue & queue)
        {
            F _f = std::move(f);
            auto iter_tuple = begin.get_iterator_tuple();
            auto it1 = hpx::util::get<0>(iter_tuple);
            auto it2 = hpx::util::get<1>(iter_tuple);
            auto buffer_first = detail::get_buffer( it1, data_count);
            auto buffer_second = detail::get_buffer( it2, data_count);
            std::cout << "Run: " << it1.pos() << " " << it2.pos() << " " << global_size << std::endl;
            auto command_group = [=]() mutable {
                queue.submit( [=](cl::sycl::handler & cgh) mutable {
                    auto buffer_acc_first = detail::get_device_acc(cgh, buffer_first, it1);
                    auto buffer_acc_second = detail::get_device_acc(cgh, buffer_second, it2);
                    auto kernel = [=] (cl::sycl::nd_item<1> idx) mutable {

                        if(idx.get_global_linear_id() >= data_count)
                            return;
                        //detail::device_iterator<value_type> it = detail::get_device_it<value_type>(buffer_acc, idx.get_global_linear_id());
                        cl::sycl::global_ptr<unsigned long> dev_it1 = buffer_acc_first.first.get_pointer() + buffer_acc_first.second + idx.get_global_linear_id();
                        cl::sycl::global_ptr<unsigned long> dev_it2 = buffer_acc_second.first.get_pointer() + buffer_acc_second.second + idx.get_global_linear_id();

                        auto t = hpx::util::make_tuple(hpx::util::make_zip_iterator(dev_it1, dev_it2), chunk_size, offset);
                        hpx::util::invoke(_f, t);
                    };

                    cgh.parallel_for<Name>(
                        cl::sycl::nd_range<1>(cl::sycl::range<1>(global_size), cl::sycl::range<1>(local_size)),
                        kernel
                    );
                });
            };
            command_group();
        }
    };*

    //FIXME: read-write access
    /*template<typename Iter>
    std::pair<
        cl::sycl::accessor<
            typename Iter::value_type,
            1,
            cl::sycl::access::mode::read_write,
            cl::sycl::access::target::global_buffer
        >,
        size_t
    > get_accessor(const Iter & begin, cl::sycl::handler & cgh)
    {
        return std::make_pair(
                (*begin).device_data().template get_access<cl::sycl::access::mode::read_write>(cgh),
                (*begin).pos()
        );
    }

    template<typename T, typename Name>
    struct launch_helper<detail::host_iterator<T>, Name>
    {
        // Implement 1D ND-Range for now
        // We don't have algorithms requiring 2D or 3D access
        template <typename F, typename... Args>
        HPX_HOST_DEVICE
        static void call(F && f, const detail::host_iterator<T> & begin, int data_count, int offset,
                         int chunk_size, int global_size, int local_size, cl::sycl::queue & queue)
        {
            auto cmd_group_launcher = [=](cl::sycl::handler & cgh) mutable {
                auto buffer_accessor = get_accessor(begin, cgh);
                auto kernel = [=] (cl::sycl::nd_item<1> idx) mutable {

                    if(idx.get_global_linear_id() >= data_count)
                        return;
                    cl::sycl::global_ptr<T> dev_it = buffer_accessor.first.get_pointer()
                                                      + buffer_accessor.second + idx.get_global_linear_id();

                    auto t = hpx::util::make_tuple(dev_it, chunk_size, offset);
                    hpx::util::invoke(_f, t);
                };
                cgh.parallel_for<Name>(
                    cl::sycl::nd_range<1>(cl::sycl::range<1>(global_size), cl::sycl::range<1>(local_size)),
                    kernel
                );
            };
        }
    };

    template<typename Name, typename F, typename... Args>
    void launch(target const& t, int global_size, int local_size, F && f, Args &&... vs)
    {
        typedef closure<F, Ts...> closure_type;
        launch_helper<closure_type>::call(t, gridDim, blockDim, std::forward<F>(f),
                                          util::forward_as_tuple(std::forward<Ts>(vs)...));
    }*/

    // FIXME: change it to allow variadic args - parse buffers to a tuple
    template<typename Name, typename F, typename T, typename... Args>
    void launch(target const& t, std::size_t global_size, std::size_t local_size, F && f, const target_ptr<T> & ptr, Args &&...)
    {
        // First element of shape object is either an iterator or a tuple of iterators
        //typedef typename hpx::util::tuple_element<0, decltShape>::type iterator_type;

        //f can't be caught as an rvalue
        auto _f = std::move(f);
        // Iter has to be a SYCL device iterator
        cl::sycl::buffer<T> * buffer = ptr.device_data();
        uint64_t pos = ptr.pos();
        std::size_t original_global_size = global_size;
        if (global_size % local_size)
        {
            global_size += local_size - (global_size % local_size);
        }
        auto cmd_group_launcher = [=](cl::sycl::handler & cgh) mutable {
            auto buffer_acc = buffer->template get_access<cl::sycl::access::mode::read_write>(cgh);
            auto kernel = [=] (cl::sycl::nd_item<1> idx) mutable {
                size_t id = idx.get_global_linear_id();
                if(id >= original_global_size)
                    return;
                hpx::util::invoke(_f, id, buffer_acc.get_pointer() + pos);
            };
            cgh.parallel_for<Name>(
                    cl::sycl::nd_range<1>(cl::sycl::range<1>(global_size), cl::sycl::range<1>(local_size)),
                    kernel
            );
        };
        std::cout << "Launch: " << global_size << " " << local_size << " " << pos << std::endl;
        t.native_handle().get_queue().submit(cmd_group_launcher);
    }

    // FIXME: change it to allow variadic args - parse buffers to a tuple
    //template<typename Name, typename F, typename T, typename... Args>
    //void launch(target const& t, int global_size, int local_size, F && f, const target_ptr<T> & ptr, Args &&... args)
    //{
    //}

    // FIXME: change it to allow variadic args - parse buffers to a tuple
    // FIXME: allow &&... and transfer as tuple
    template<typename Name, typename F, typename T, typename... Args>
    void launch(target const& t, int global_size, F && f, const target_ptr<T> & buffer, Args &&...)
    {
        //F can't be caught by a reference
        F _f = std::move(f);
        auto cmd_group_launcher = [=](cl::sycl::handler & cgh) mutable {
            auto buffer_accessor = buffer.device_data()->template get_access<cl::sycl::access::mode::read_write>(cgh);
            auto kernel = [=] (cl::sycl::item<1> idx) mutable {
                hpx::util::invoke(_f, idx.get_linear_id(), buffer_accessor.get_pointer());
            };
            cgh.parallel_for<Name>(
                    cl::sycl::range<1>(global_size),
                    kernel
            );
        };
        t.native_handle().get_queue().submit(cmd_group_launcher);
    }
}}}}

#endif
#endif
