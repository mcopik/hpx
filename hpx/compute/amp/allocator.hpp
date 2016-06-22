//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_AMP_ALLOCATOR_HPP
#define HPX_COMPUTE_AMP_ALLOCATOR_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_AMP)
#include <hpx/compute/amp/detail/buffer_proxy.hpp>
#include <hpx/compute/amp/target_ptr.hpp>

#include <hpx/exception.hpp>
#include <hpx/util/unused.hpp>

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <memory>
#include <string>
#include <type_traits>

namespace hpx { namespace compute { namespace amp
{
    template <typename T>
    class allocator
    {
    public:
        typedef detail::buffer_proxy<T> value_type;
        typedef target_ptr<value_type> pointer;
        typedef target_ptr<value_type const> const_pointer;
#if defined(__COMPUTE__ACCELERATOR__)
        /// Define direct access to allocated data in device code
        typedef value_type& reference;
        typedef value_type const& const_reference;
#else
        /// On host code, use a proxy handling access to device memory
        typedef value_proxy<value_type> & reference;
        typedef value_proxy<value_type const> & const_reference;
#endif
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        template <typename U>
        struct rebind
        {
            typedef allocator<U> other;
        };

        typedef std::true_type is_always_equal;
        typedef std::true_type propagate_on_container_move_assignment;

        typedef amp::target target_type;
        typedef Concurrency::array<T, 1> buffer_type;

        allocator()
          : target_(&amp::get_default_target())
        {}

        allocator(target_type& tgt)
          : target_(&tgt)
        {}

        template <typename U>
        allocator(allocator<U>& alloc)
          : target_(alloc.target_)
        {}

        // Returns the actual address of x even in presence of overloaded
        // operator&
        pointer address(reference x) const HPX_NOEXCEPT
        {
#if defined(__HCC_ACCELERATOR__)
             return &x;
#else
            return pointer(x.device_ptr(), *target_);
#endif
        }

        const_pointer address(const_reference x) const HPX_NOEXCEPT
        {
#if defined(__HCC_ACCELERATOR__)
            return &x;
#else
            return pointer(x.device_ptr(), *target_);
#endif
        }

        // Allocates n * sizeof(T) bytes of uninitialized storage by calling
        // cudaMalloc, but it is unspecified when and how this function is
        // called. The pointer hint may be used to provide locality of
        // reference: the allocator, if supported by the implementation, will
        // attempt to allocate the new memory block as close as possible to hint.
        pointer allocate(size_type n, const_pointer hint = nullptr)
            restrict(amp, cpu)
        {
#if defined(__HCC_ACCELERATOR__)
            /// No memory allocation on device side
            pointer result;
            return result;
#else
            value_type *p = 0;
            try {
                buffer_type * buffer = new buffer_type(
                    Concurrency::extent<1>(n),
                    target_->native_handle().get_device());
                p = new value_type(buffer);
                pointer result(p, *target_);
                return result;
            } catch (Concurrency::runtime_exception & exc) {

                HPX_THROW_EXCEPTION(out_of_memory,
                    "amp::allocator<T>::allocate()",
                    std::string("Construction of AMP array failed: ") +
                        std::string(exc.what()));

            } catch (std::exception & exc) {

                HPX_THROW_EXCEPTION(no_success,
                    "amp::allocator<T>::allocate()",
                    std::string("Construction of AMP array failed: ") +
                        std::string(exc.what()));

            }
#endif
        }

        // Deallocates the storage referenced by the pointer p, which must be a
        // pointer obtained by an earlier call to allocate(). The argument n
        // must be equal to the first argument of the call to allocate() that
        // originally produced p; otherwise, the behavior is undefined.
        void deallocate(pointer p, size_type n)
        {
#if !defined(__HCC_ACCELERATOR__)
            try {
                delete p.device_ptr();
            } catch (Concurrency::runtime_exception & exc) {

                HPX_THROW_EXCEPTION(out_of_memory,
                    "amp::allocator<T>::deallocate()",
                    std::string("Deallocation of AMP array failed: ") +
                        std::string(exc.what()));

            } catch (std::exception & exc) {

                HPX_THROW_EXCEPTION(no_success,
                    "amp::allocator<T>::allocate()",
                    std::string("Deallocation of AMP array failed: ") +
                        std::string(exc.what()));

            }
#endif
        }

        // Returns the maximum theoretically possible value of n, for which the
        // call allocate(n, 0) could succeed.
        // Implementation uses dedicated_memory() to obtain amount of memory
        // on accelerator.
        size_type max_size() const HPX_NOEXCEPT
        {
            try {
                auto device = target_->native_handle().get_device();
                return device.get_accelerator().get_dedicated_memory() /
                    sizeof(value_type);
            } catch (Concurrency::runtime_exception & exc) {

                HPX_THROW_EXCEPTION(no_success,
                    "amp::allocator<T>::max_size(()",
                    std::string("Calling accellerator_view.dedicated_memory()"
                    " failed: ") +  std::string(exc.what()));

            }
        }

    public:
//        // Constructs count objects of type T in allocated uninitialized
//        // storage pointed to by p, using placement-new
//        template <typename U, typename ... Args>
//        void bulk_construct(U* p, std::size_t count, Args &&... args)
//        {
//            int threads_per_block = (std::min)(1024, int(count));
//            int num_blocks =
//                int((count + threads_per_block - 1) / threads_per_block);
//
//            detail::launch(
//                *target_, num_blocks, threads_per_block,
//                [] __device__ (U* p, std::size_t count, Args const&... args)
//                {
//                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//                    if (idx < count)
//                    {
//                        ::new (p + idx) U (std::forward<Args>(args)...);
//                    }
//                },
//                p, count, std::forward<Args>(args)...);
//            target_->synchronize();
//        }
//
//        // Constructs an object of type T in allocated uninitialized storage
//        // pointed to by p, using placement-new
//        template <typename U, typename ... Args>
//        void construct(U* p, Args &&... args)
//        {
//            detail::launch(
//                *target_, 1, 1,
//                [] __device__ (U* p, Args const&... args)
//                {
//                    ::new (p) U (std::forward<Args>(args)...);
//                },
//                p, std::forward<Args>(args)...);
//            target_->synchronize();
//        }
//
//        // Calls the destructor of count objects pointed to by p
        void bulk_destroy(pointer p, std::size_t count)
        {
//            int threads_per_block = (std::min)(1024, int(count));
//            int num_blocks =
//                int((count + threads_per_block) / threads_per_block) - 1;
//
//            detail::launch(
//                *target_, num_blocks, threads_per_block,
//                [] __device__ (U* p, std::size_t count)
//                {
//                    int idx = blockIdx.x * blockDim.x + threadIdx.x;
//                    if (idx < count)
//                    {
//                        (p + idx)->~U();
//                    }
//                },
//                p, count);
//            target_->synchronize();
        }
//
        // Calls the destructor of the object pointed to by p
        void destroy(pointer p)
        {
            bulk_destroy(p, 1);
        }

        // Access the underlying target (device)
        target_type& target() const HPX_NOEXCEPT
        {
            return *target_;
        }

    private:
        target_type* target_;
    };
}}}

#endif
#endif

