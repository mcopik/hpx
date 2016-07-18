///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_HC_TARGET_PTR_HPP
#define HPX_COMPUTE_HC_TARGET_PTR_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_HC)

#include <hpx/util/assert.hpp>

#include <hpx/compute/hc/config.hpp>
#include <hpx/compute/hc/target.hpp>
#include <hpx/compute/hc/value_proxy.hpp>
#include <hpx/compute/detail/get_proxy_type.hpp>
#include <hpx/compute/hc/detail/buffer_proxy.hpp>

namespace hpx { namespace compute { namespace hc
{

	template<typename T>
	struct get_internal_type
	{
		typedef T type;
	};

	template<typename T>
	struct get_internal_type<detail::buffer_proxy<T>>
	{
		typedef get_internal_type<T> type;
	};

    template<typename T>
    struct target_ptr;

    template<typename T>
    struct target_ptr< buffer_t<T> >
    {
    public:
        //typedef typename compute::detail::get_proxy_type<T>::type *
        //    proxy_type;
        typedef buffer_t<T> proxy_type;
        //typedef typename get_internal_type<T>::type internal_type;
        typedef std::random_access_iterator_tag iterator_category;
#if defined(__COMPUTE__ACCELERATOR__)
        typedef T value_type;
        typedef T * pointer;
        typedef T & reference;
#else
        typedef value_proxy<proxy_type> value_type;
        typedef T *pointer;
        typedef value_proxy<proxy_type> reference;
        typedef value_proxy<const proxy_type> const_reference;
#endif
        typedef std::ptrdiff_t difference_type;

        [[hc,cpu]] target_ptr()
            : p_(nullptr), buffer_(nullptr) {}//, tgt_(nullptr) { }

        // Necessary for nullability of pointer:
        //
        [[hc,cpu]] target_ptr(std::nullptr_t)
            : p_(nullptr), buffer_(nullptr) {}//, tgt_(nullptr) { }

        [[hc,cpu]] target_ptr(  proxy_type * buffer,
                                T * p = nullptr)
            : p_(p), buffer_(buffer)
        {
            p_ = p ? p : buffer->accelerator_pointer();
        }

        [[hc,cpu]] target_ptr(proxy_type && p)
            :  p_(p->accelerator_pointer()), buffer_(*p) {}//, buffer(&p){}//, pos_(pos), tgt_(tgt) { }

        target_ptr const &operator++() {
            HPX_ASSERT(p_);
            ++p_;
            return *this;
        }

        target_ptr const &operator--() {
            HPX_ASSERT(p_);
            --p_;
            return *this;
        }

        target_ptr operator++(int) {
            target_ptr tmp(*this);
            HPX_ASSERT(p_);
            ++p_;
            return tmp;
        }

        target_ptr operator--(int) {
            target_ptr tmp(*this);
            HPX_ASSERT(p_);
            --p_;
            return tmp;
        }

        explicit operator bool() const {
            return p_ != nullptr;
        }

        bool operator==(std::nullptr_t) const {
            return p_ == nullptr;
        }

        bool operator!=(std::nullptr_t) const {
            return p_ != nullptr;
        }

        friend bool operator==(std::nullptr_t, target_ptr const &rhs) {
            return nullptr == rhs.p_;
        }

        friend bool operator!=(std::nullptr_t, target_ptr const &rhs) {
            return nullptr != rhs.p_;
        }

        friend bool operator==(target_ptr const &lhs,
                               target_ptr const &rhs) {
            return lhs.p_ == rhs.p_;
        }

        friend bool operator!=(target_ptr const &lhs,
                               target_ptr const &rhs) {
            return lhs.p_ != rhs.p_;
        }

        friend bool operator<(target_ptr const &lhs,
                              target_ptr const &rhs) {
            return lhs.p_ < rhs.p_;
        }

        friend bool operator>(target_ptr const &lhs,
                              target_ptr const &rhs) {
            return lhs.p_ > rhs.p_;
        }

        friend bool operator<=(target_ptr const &lhs,
                               target_ptr const &rhs) {
            return lhs.p_ <= rhs.p_;
        }

        friend bool operator>=(target_ptr const &lhs,
                               target_ptr const &rhs) {
            return lhs.p_ >= rhs.p_;
        }

        target_ptr &operator+=(std::ptrdiff_t offset) {
            HPX_ASSERT(p_);
            p_ += offset;
            return *this;
        }

        target_ptr &operator-=(std::ptrdiff_t offset) {
            HPX_ASSERT(p_);
            p_ -= offset;
            return *this;
        }

        std::ptrdiff_t operator-(target_ptr const &other) const {
            return p_ - other.p_;
        }

        target_ptr operator-(std::ptrdiff_t offset) const {
            return target_ptr(buffer_, p_ - offset);
        }

        target_ptr operator+(std::ptrdiff_t offset) const {
            return target_ptr(buffer_, p_ + offset);
        }

        T *device_ptr() const {
            return p_;
        }

#if defined(__COMPUTE__ACCELERATOR__)

        T & operator*()
        {
            return *p_;
        }

        //T const& operator[](std::ptrdiff_t offset) const
        //{
        //    return (*p_)[pos_ + offset];
        //}

        T & operator[](std::ptrdiff_t offset) const
        {
            return p_[offset];
        }

        operator T*() const
        {
            return p_;
        }

        T* operator->() const
        {
            return p_;
        }
#else

        reference operator*() const {
            return value_proxy<proxy_type>(buffer_, p_);
        }

//        const_reference operator*() const {
//            return value_proxy<const proxy_type>(p_, *tgt_);
//        }

        reference operator[](std::ptrdiff_t offset) {
            return value_proxy<proxy_type>(buffer_, p_ + offset);
        }

        const_reference operator[](std::ptrdiff_t offset) const {
            return value_proxy<const proxy_type>(buffer_, p_ + offset);
        }

        // Dirrect access to GPU pointer, for compilation compability.
        // Will NOT work correctly!
        explicit operator T*() const {
            return p_;
        }

        // Dirrect access to GPU pointer, for compilation compability.
        // Will NOT work correctly!
        T *operator->() const {
            return p_;
        }

#endif
        //std::ptrdiff_t pos() const {
        //    return pos_;
        //}
        //target * target_() {
        //    return tgt_;
        //}
        buffer_t<T> * device_buffer() {
            return buffer_;
        }
    protected:
        T * p_;
        buffer_t<T> * buffer_;
        //std::ptrdiff_t pos_;
        //target * tgt_;
    };

    template<typename T>
    class target_ptr< const buffer_t<T> >
    {
    public:
        //typedef typename compute::detail::get_proxy_type<T>::type *
        //    proxy_type;
        typedef const buffer_t<T> proxy_type;
        //typedef typename get_internal_type<T>::type internal_type;
        typedef std::random_access_iterator_tag iterator_category;
#if defined(__COMPUTE__ACCELERATOR__)
        typedef proxy_type value_type;
        typedef proxy_type * pointer;
        typedef proxy_type & reference;
#else
        typedef value_proxy<proxy_type> value_type;
        typedef T *pointer;
        typedef value_proxy<proxy_type> reference;
        typedef value_proxy<const proxy_type> const_reference;
#endif
        typedef std::ptrdiff_t difference_type;

        target_ptr()
            : p_(nullptr), tgt_(nullptr) { }

        // Necessary for nullability of pointer:
        //
        target_ptr(std::nullptr_t)
            : p_(nullptr), tgt_(nullptr) { }

        target_ptr(proxy_type * p, hc::target * tgt,
                std::ptrdiff_t pos = 0)
            : buffer(p), p_(p->accelerator_pointer()), pos_(pos), tgt_(tgt) { }

        target_ptr(proxy_type && p, hc::target * tgt,
                std::ptrdiff_t pos = 0)
            : buffer(&p), p_(p->accelerator_pointer()), pos_(pos), tgt_(tgt) { }

        target_ptr const &operator++() {
            HPX_ASSERT(p_);
            ++pos_;
            return *this;
        }

        target_ptr const &operator--() {
            HPX_ASSERT(p_);
            --pos_;
            return *this;
        }

        target_ptr operator++(int) {
            target_ptr tmp(*this);
            HPX_ASSERT(p_);
            ++pos_;
            return tmp;
        }

        target_ptr operator--(int) {
            target_ptr tmp(*this);
            HPX_ASSERT(p_);
            --pos_;
            return tmp;
        }

        explicit operator bool() const {
            return p_ != nullptr;
        }

        bool operator==(std::nullptr_t) const {
            return p_ == nullptr;
        }

        bool operator!=(std::nullptr_t) const {
            return p_ != nullptr;
        }

        friend bool operator==(std::nullptr_t, target_ptr const &rhs) {
            return nullptr == rhs.p_;
        }

        friend bool operator!=(std::nullptr_t, target_ptr const &rhs) {
            return nullptr != rhs.p_;
        }

        friend bool operator==(target_ptr const &lhs,
                               target_ptr const &rhs) {
            return lhs.pos_ == rhs.pos_;
        }

        friend bool operator!=(target_ptr const &lhs,
                               target_ptr const &rhs) {
            return lhs.pos_ != rhs.pos_;
        }

        friend bool operator<(target_ptr const &lhs,
                              target_ptr const &rhs) {
            return lhs.pos_ < rhs.pos_;
        }

        friend bool operator>(target_ptr const &lhs,
                              target_ptr const &rhs) {
            return lhs.pos_ > rhs.pos_;
        }

        friend bool operator<=(target_ptr const &lhs,
                               target_ptr const &rhs) {
            return lhs.pos_ <= rhs.pos_;
        }

        friend bool operator>=(target_ptr const &lhs,
                               target_ptr const &rhs) {
            return lhs.pos_ >= rhs.pos_;
        }

        target_ptr &operator+=(std::ptrdiff_t offset) {
            HPX_ASSERT(p_);
            pos_ += offset;
            return *this;
        }

        target_ptr &operator-=(std::ptrdiff_t offset) {
            HPX_ASSERT(p_);
            pos_ -= offset;
            return *this;
        }

        std::ptrdiff_t operator-(target_ptr const &other) const {
            return p_ - other.pos_;
        }

        target_ptr operator-(std::ptrdiff_t offset) const {
            return target_ptr(p_, tgt_, offset);
        }

        target_ptr operator+(std::ptrdiff_t offset) const {
            return target_ptr(p_, tgt_, offset);
        }

        proxy_type *device_ptr() const {
            return buffer;
        }

#if defined(__COMPUTE__ACCELERATOR__)

        /// DOESN'T WORK!
        T & operator*()
        {
            return *p_;
        }

        //T const& operator[](std::ptrdiff_t offset) const
        //{
        //    return (*p_)[pos_ + offset];
        //}

        T & operator[](std::ptrdiff_t offset) const
        {
            return p_[offset];
        }

        operator T*() const
        {
            return p_;
        }

        T* operator->() const
        {
            return p_;
        }
#else

        reference operator*() const {
            return value_proxy<proxy_type>(p_, pos_, tgt_);
        }

//        const_reference operator*() const {
//            return value_proxy<const proxy_type>(p_, *tgt_);
//        }

        reference operator[](std::ptrdiff_t offset) {
            return value_proxy<proxy_type>(p_, pos_ + offset, tgt_);
        }

        const_reference operator[](std::ptrdiff_t offset) const {
            return value_proxy<const proxy_type>(p_, pos_ + offset,
                tgt_);
        }

        // Dirrect access to GPU pointer, for compilation compability.
        // Will NOT work correctly!
        explicit operator T*() const {
            return p_;
        }

        // Dirrect access to GPU pointer, for compilation compability.
        // Will NOT work correctly!
        T *operator->() const {
            return p_;
        }

#endif
        std::ptrdiff_t pos() const {
            return pos_;
        }
    protected:
        buffer_t<T> * buffer;
        T * p_;
        std::ptrdiff_t pos_;
        target * tgt_;
    };


}}}

#endif
#endif
