//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_COMPUTE_SYCL_ALLOCATOR_HPP
#define HPX_COMPUTE_SYCL_ALLOCATOR_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SYCL)

#include <hpx/exception.hpp>

#include <hpx/compute/sycl/config.hpp>
#include <hpx/compute/sycl/target.hpp>
#include <hpx/compute/sycl/target_ptr.hpp>
#include <hpx/compute/sycl/value_proxy.hpp>

#include <string>

namespace hpx { namespace compute { namespace sycl
{
    template <typename T>
    class allocator
    {
    public:
        typedef T value_type;
        typedef target_ptr<T> pointer;
        typedef target_ptr<T const> const_pointer;
//#if defined(__CUDA_ARCH__)
//        typedef T& reference;
//        typedef T const& const_reference;
//#else
        typedef value_proxy<T> reference;
        typedef value_proxy<T const> const_reference;
//#endif
        typedef std::size_t size_type;
        typedef std::ptrdiff_t difference_type;

        template <typename U>
        struct rebind
        {
            typedef allocator<U> other;
        };

        typedef std::true_type is_always_equal;
        typedef std::true_type propagate_on_container_move_assignment;

        typedef sycl::target target_type;

        allocator()
          : target_(sycl::get_default_target())
        {}

        allocator(target_type const& tgt)
          : target_(tgt)
        {}

        allocator(target_type && tgt)
          : target_(std::move(tgt))
        {}

        template <typename U>
        allocator(allocator<U> const& alloc)
          : target_(alloc.target_)
        {}

        // Returns the actual address of x even in presence of overloaded
        // operator&
        pointer address(reference x) const HPX_NOEXCEPT
        {
            return pointer(x.device_data(), target_);
        }

        const_pointer address(const_reference x) const HPX_NOEXCEPT
        {
            return pointer(x.device_data(), target_);
        }

        // Allocates n * sizeof(T) bytes of uninitialized storage by creating
        // a new sycl::buffer. Exact location of the memory is unspecified
        // but it is logically tied in HPX to a specific target.
        // The responsilibity for efficient memory transfer and allocation relies
        // on SYCL runtime because its API does not give us any possibility to control
        // it or provide hints.
        pointer allocate(size_type n)
        {
            try {
                std::cout << "Allocate" << std::endl;
                return pointer(new buffer_t<T>(n), 0);
            } catch (exception_t & exc) {
                HPX_THROW_EXCEPTION(out_of_memory,
                    "sycl::allocator<T>::allocate()",
                    std::string("buffer creation failed: ") +
                        exc.what());
            }
        }

        // Deallocates the storage referenced by the pointer p, which must be a
        // pointer obtained by an earlier call to allocate(). The argument n
        // must be equal to the first argument of the call to allocate() that
        // originally produced p; otherwise, the behavior is undefined.
        //
        // sycl::target_ptr<T> saves only a pointer to given SYCL buffer
        // and never attempts to deallocate this memory.
        // Freeing this buffer happens *only* here - that's why we use plain
        // pointer instead of shared_ptr
        void deallocate(pointer p, size_type n)
        {
            try {
                std::cout << "Delete " << std::endl;
                delete p.device_data();
            } catch (exception_t & exc) {
                HPX_THROW_EXCEPTION(out_of_memory,
                    "sycl::allocator<T>::deallocate()",
                    std::string("buffer deletion failed: ") +
                        exc.what());
            }
        }

        // Returns the maximal possible value of n in allocation
        // process. Returns the max_mem_alloc_size from device info
        // divided by size of type T
        size_type max_size() const HPX_NOEXCEPT
        {
            try {
                typedef cl::sycl::info::device info;
                int max_size = target_.native_handle().get_device().template
                    get_info<info::max_mem_alloc_size>();

                return max_size / sizeof(T);
            } catch(exception_t & exc) {
                HPX_THROW_EXCEPTION(kernel_error,
                    "sycl::allocator<T>::max_size()",
                    std::string("sycl::device::get_info<info::device::"
                        "max_mem_alloc_size> failed: ") + exc.what()
                    );
            }
        }

    public:
        // Constructs count objects of type T in allocated uninitialized
        // storage pointed to by p, using placement-new
        template <typename ... Args>
        void bulk_construct(pointer p, std::size_t count, Args &&... args)
        {
        }

        // Constructs an object of type T in allocated uninitialized storage
        // pointed to by p, using placement-new
        template <typename ... Args>
        void construct(pointer p, Args &&... args)
        {
        }

        // Calls the destructor of count objects pointed to by p
        void bulk_destroy(pointer p, std::size_t count)
        {
        }

        // Calls the destructor of the object pointed to by p
        void destroy(pointer p)
        {
            bulk_destroy(p, 1);
        }

        // Access the underlying target (device)
        target_type& target() HPX_NOEXCEPT
        {
            return target_;
        }
        target_type const& target() const HPX_NOEXCEPT
        {
            return target_;
        }

    private:
        target_type target_;
    };
}}}

#endif
#endif

