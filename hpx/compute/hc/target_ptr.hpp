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
    class target_ptr;

    template<typename T>
    class target_ptr< detail::buffer_proxy<T> >
    {
    public:
        //typedef typename compute::detail::get_proxy_type<T>::type *
        //    proxy_type;
        typedef detail::buffer_proxy<T> proxy_type;
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

        target_ptr(proxy_type *p, hc::target &tgt)
            : p_(p), tgt_(&tgt) { }

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
            return target_ptr(p_ - offset, *tgt_);
        }

        target_ptr operator+(std::ptrdiff_t offset) const {
            return target_ptr(p_ + offset, *tgt_);
        }

        T *device_ptr() const {
            return p_->device_ptr();
        }

#if defined(__COMPUTE__ACCELERATOR__)

        proxy_type & operator*()
        {
            return *p_;
        }

        proxy_type const& operator[](std::ptrdiff_t offset) const
        {
            return *(p_ + offset);
        }

        proxy_type& operator[](std::ptrdiff_t offset)
        {
            return *(p_ + offset);
        }

        operator proxy_type*() const
        {
            return p_;
        }

        proxy_type* operator->() const
        {
            return p_;
        }
#else

        reference operator*() const {
            return value_proxy<proxy_type>(p_, *tgt_);
        }

//        const_reference operator*() const {
//            return value_proxy<const proxy_type>(p_, *tgt_);
//        }

        reference operator[](std::ptrdiff_t offset) {
            return value_proxy<proxy_type>(p_ + offset, *tgt_);
        }

        const_reference operator[](std::ptrdiff_t offset) const {
            return value_proxy<const proxy_type>(p_ + offset, *tgt_);
        }

        explicit operator T*() const {
            return p_;
        }

        T *operator->() const {
            return p_;
        }

#endif

    protected:
        detail::buffer_proxy<T> *p_;
        target *tgt_;
    };

    template<typename T>
    class target_ptr< const detail::buffer_proxy<T> >
    {
    public:
        //typedef typename compute::detail::get_proxy_type<T>::type *
        //    proxy_type;
        typedef const detail::buffer_proxy<T> proxy_type;
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

        target_ptr(proxy_type *p, hc::target &tgt)
            : p_(p), tgt_(&tgt) { }

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
            return target_ptr(p_ - offset, *tgt_);
        }

        target_ptr operator+(std::ptrdiff_t offset) const {
            return target_ptr(p_ + offset, *tgt_);
        }

        T *device_ptr() const {
            return p_->device_ptr();
        }

#if defined(__COMPUTE__ACCELERATOR__)

        proxy_type & operator*()
        {
            return *p_;
        }

        proxy_type const& operator[](std::ptrdiff_t offset) const
        {
            return *(p_ + offset);
        }

        proxy_type& operator[](std::ptrdiff_t offset)
        {
            return *(p_ + offset);
        }

        operator proxy_type*() const
        {
            return p_;
        }

        proxy_type* operator->() const
        {
            return p_;
        }
#else

        reference operator*() const {
            return value_proxy<proxy_type>(p_, *tgt_);
        }

        //        const_reference operator*() const {
        //            return value_proxy<const proxy_type>(p_, *tgt_);
        //        }

        reference operator[](std::ptrdiff_t offset) {
            return value_proxy<proxy_type>(p_ + offset, *tgt_);
        }

        const_reference operator[](std::ptrdiff_t offset) const {
            return value_proxy<const proxy_type>(p_ + offset, *tgt_);
        }

        explicit operator T *() const {
            return p_;
        }

        T *operator->() const {
            return p_;
        }

#endif

    protected:
        detail::buffer_proxy<T> *p_;
        target *tgt_;
    };


}}}

#endif
#endif
