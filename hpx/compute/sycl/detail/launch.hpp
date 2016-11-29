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
#include <hpx/util/zip_iterator.hpp>

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
    namespace detail
    {
        // FIXME: change it to allow variadic args - parse buffers to a tuple
        template<typename... Args>
        struct launch;

        template<typename T, typename... Args>
        struct launch<target_ptr<T>, Args...>
        {
            template<typename Name, typename F>
            static void call(target const& t, std::size_t global_size, std::size_t local_size, F && f, const target_ptr<T> & ptr, Args &&...)
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
        };

        template<typename T, typename U, typename... Args>
        struct launch<target_ptr<T>, target_ptr<U>, Args...>
        {
            // FIXME: change it to allow variadic args - parse buffers to a tuple
            template<typename Name, typename F>
            static void call(target const& t, std::size_t global_size, std::size_t local_size, F && f,
                const target_ptr<T> & ptr1, const target_ptr<U> & ptr2, Args &&...)
            {
                // First element of shape object is either an iterator or a tuple of iterators
                //typedef typename hpx::util::tuple_element<0, decltShape>::type iterator_type;

                //f can't be caught as an rvalue
                auto _f = std::move(f);
                uint64_t pos1 = ptr1.pos(), pos2 = ptr2.pos();
                // Iter has to be a SYCL device iterator
                std::size_t original_global_size = global_size;
                if (global_size % local_size)
                {
                    global_size += local_size - (global_size % local_size);
                }
                auto cmd_group_launcher = [=](cl::sycl::handler & cgh) mutable {
                    auto buffer_acc = ptr1.device_data()->template get_access<cl::sycl::access::mode::read_write>(cgh);
                    auto buffer_acc2 = ptr2.device_data()->template get_access<cl::sycl::access::mode::read_write>(cgh);
                    auto kernel = [=] (cl::sycl::nd_item<1> idx) mutable {
                        size_t id = idx.get_global_linear_id();
                        if(id >= original_global_size)
                            return;
                        cl::sycl::global_ptr<T> dev_ptr1 = buffer_acc.get_pointer() + pos1;
                        cl::sycl::global_ptr<U> dev_ptr2 = buffer_acc2.get_pointer() + pos2;
                        hpx::util::invoke(_f, id, hpx::util::make_zip_iterator(dev_ptr1, dev_ptr2));
                    };
                    cgh.parallel_for<Name>(
                            cl::sycl::nd_range<1>(cl::sycl::range<1>(global_size), cl::sycl::range<1>(local_size)),
                            kernel
                    );
                };
                //std::cout << "Launch: " << global_size << " " << local_size << " " << pos << std::endl;
                t.native_handle().get_queue().submit(cmd_group_launcher);
            }
        };
    }

    template<typename Name, typename F, typename... Args>
    void launch(target const& t, std::size_t global_size, std::size_t local_size, F && f, Args &&... args)
    {
        detail::launch<Args...>::template call<Name>(t, global_size, local_size, std::forward<F>(f), std::forward<Args>(args)...);
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
