///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_SYCL_TARGET_PTR_HPP
#define HPX_COMPUTE_SYCL_TARGET_PTR_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SYCL)

#include <hpx/util/assert.hpp>
#include <hpx/util/iterator_facade.hpp>

#include <hpx/compute/sycl/target.hpp>
#include <hpx/compute/sycl/value_proxy.hpp>
#include <hpx/compute/detail/get_proxy_type.hpp>

#include <cstddef>

namespace hpx { namespace compute { namespace sycl
{
    // A pointer is not offered hence we fake it
    //
    template <typename T>
    class target_ptr :
        public hpx::util::iterator_facade<
                target_ptr<T>,
                value_proxy<T>,
                std::random_access_iterator_tag,
                value_proxy<T>,
                std::ptrdiff_t,
                T*
            >
    {
        typedef hpx::util::iterator_facade<
                target_ptr<T>,
                value_proxy<T>,
                std::random_access_iterator_tag,
                value_proxy<T>,
                std::ptrdiff_t,
                T*
            > base_type;
    public:

        // Necessary for concepts in algorithms - verification of callable
        typename compute::detail::get_proxy_type<T>::type * proxy_type;

        typedef target_ptr<T> pointer;
        typedef value_proxy<T> reference;
        typedef std::ptrdiff_t difference_type;

        target_ptr()
          : pos_(0),
            buffer_(nullptr)
        {}

        explicit target_ptr(std::nullptr_t)
          : pos_(0),
            buffer_(nullptr)
        {}

        target_ptr(buffer_t<T> * buffer, uint64_t pos)
          : pos_(pos),
            buffer_(buffer)
        {}

        target_ptr(target_ptr const& rhs)
          : pos_(rhs.pos_),
            buffer_(rhs.buffer_)
        {}

        target_ptr& operator=(target_ptr const& rhs)
        {
            pos_ = rhs.pos_;
            buffer_ = rhs.buffer_;
            return *this;
        }

        target_ptr& operator=(std::nullptr_t)
        {
            pos_ = 0;
            buffer_ = nullptr;
            return *this;
        }

        explicit operator bool() const
        {
            return buffer_;
        }

        friend bool operator==(target_ptr const& lhs, std::nullptr_t)
        {
            return !lhs.buffer_;
        }

        friend bool operator!=(target_ptr const& lhs, std::nullptr_t)
        {
            return lhs.buffer_;
        }

        friend bool operator==(std::nullptr_t, target_ptr const& rhs)
        {
            return !rhs.buffer_;
        }

        friend bool operator!=(std::nullptr_t, target_ptr const& rhs)
        {
            return rhs.buffer_;
        }

        buffer_t<T> * device_data() const
        {
            return buffer_;
        }

        uint64_t pos() const
        {
            return pos_;
        }

        reference dereference() const
        {
            return value_proxy<T>(buffer_, pos_);
        }

        bool equal(const target_ptr<T> & oth) const
        {
            return buffer_ == oth.buffer_ && pos_ == oth.pos_;
        }

        void increment()
        {
            ++pos_;
        }

        void decrement()
        {
            --pos_;
        }

        void advance(std::ptrdiff_t n)
        {
            pos_ += n;
        }

        std::ptrdiff_t distance_to(const target_ptr<T> & oth) const
        {
            return oth.pos_ - pos_;
        }

        explicit operator T*() const
        {
            return nullptr;
        }

    private:

        uint64_t pos_;
        buffer_t<T> * buffer_;
    };
}}}

#endif
#endif
