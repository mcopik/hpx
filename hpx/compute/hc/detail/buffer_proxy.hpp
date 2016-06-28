//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt

#ifndef HPX_COMPUTE_HC_DETAIL_BUFFER_PROXY_HPP
#define HPX_COMPUTE_HC_DETAIL_BUFFER_PROXY_HPP

#if defined(HPX_HAVE_HC)

#include <hpx/compute/hc/config.hpp>
#include <hpx/compute/hc/traits/access_target.hpp>

namespace hpx { namespace compute { namespace hc
{
    namespace detail {

        template<typename T>
        class buffer_proxy {
            typedef std::size_t size_type;
        public:
            /// Initialize pointer with begin position of device data
            buffer_proxy(buffer_t<T> *device_buffer) HPX_NOEXCEPT :
                device_buffer_(*device_buffer),
                device_buffer_view(device_buffer_),
                p_(device_buffer->data())
            {}

            buffer_proxy(buffer_t<T> *device_buffer, T *p) HPX_NOEXCEPT :
                device_buffer_(*device_buffer),
                device_buffer_view(device_buffer_),
                p_(p)
            {}

            buffer_proxy(buffer_proxy const &other) :
                device_buffer_(other.device_buffer_),
                device_buffer_view(other.device_buffer_view),
                p_(other.p_)
            {}

            ~buffer_proxy() {
                delete &device_buffer_;
            }

            buffer_proxy& operator=(T const& t)
            {
                *p_ = t;
                //access_target::write(*target_, p_, &t);
                return *this;
            }

            buffer_proxy& operator=(buffer_proxy const& t)
            {
                *p_ = *t;
                return *this;
            }

            //operator T() const
            //{
                //TODO: which version is more efficient?
                //return access_target::read(*target_, p_);
              //  return device_buffer_view[0];
            //}

            operator T&() const
            {
                return *t;
                //return device_buffer_view[0];
            }

            T &operator*() const {
                return *p_;
            }

            operator T*() const
            {
                return p_;
            }

            buffer_proxy<T> operator+(size_type pos) {
                return buffer_proxy(device_buffer_, p_ + pos);
            }

            T &operator++() {
                ++p_;
                return *this;
            }

            T &operator--() {
                --p_;
                return *this;
            }

            T* operator->() const
            {
                return p_;
            }

        private:
            buffer_t<T> & device_buffer_;
            buffer_acc_t<T> device_buffer_view;
            T *p_;
        };

        // Const specialization is required for situations
        template<typename T>
        class buffer_proxy<const T>
        {
            typedef std::size_t size_type;
        public:
            /// Initialize pointer with begin position of device data
            buffer_proxy(buffer_t<T> *device_buffer) HPX_NOEXCEPT :
                device_buffer_(*device_buffer),
                device_buffer_view(device_buffer_),
                p_(device_buffer->data())
            {}

            buffer_proxy(buffer_t<T> *device_buffer, T *p) HPX_NOEXCEPT :
                device_buffer_(*device_buffer),
                device_buffer_view(device_buffer_),
                p_(p)
            {}

            buffer_proxy(buffer_proxy const &other) :
                device_buffer_(other.device_buffer_),
                device_buffer_view(other.device_buffer_view),
                p_(other.p_)
            {}

            ~buffer_proxy() {
                delete device_buffer_;
            }

            operator T&() const
            {
                //TODO: which version is more efficient?
                //return access_target::read(*target_, p_);
                return device_buffer_view[0];
            }

            T &operator*() const {
                return *p_;
            }

            operator T*() const
            {
                return p_;
            }

            buffer_proxy<T> operator+(size_type pos) {
                return buffer_proxy(device_buffer_, p_ + pos);
            }

            T &operator++() {
                ++p_;
                return *this;
            }

            T &operator--() {
                --p_;
                return *this;
            }

        private:
            buffer_t<T> & device_buffer_;
            buffer_acc_t<T> device_buffer_view;
            T *p_;
        };
    }
}}}

#endif

#endif //HPX_BUFFER_PROXY_HPP
