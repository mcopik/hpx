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
        public:
            typedef std::size_t size_type;
            typedef T value_type;
            /// Initialize pointer with begin position of device data
            buffer_proxy(buffer_t<T> *device_buffer) HPX_NOEXCEPT :
                device_buffer_(*device_buffer),
                device_buffer_view(device_buffer_),
                pos_(0)
            {std::cout << "construct " << std::endl;}

            buffer_proxy(buffer_t<T> *device_buffer, uint64_t pos) HPX_NOEXCEPT :
                device_buffer_(*device_buffer),
                device_buffer_view(device_buffer_),
                pos_(pos)
            {std::cout << "construct " << std::endl;}

            buffer_proxy(buffer_proxy const &other) :
                device_buffer_(other.device_buffer_),
                device_buffer_view(other.device_buffer_view),
                pos_(other.pos_)
            {std::cout << "construct " << std::endl;}

            ~buffer_proxy() {
                std::cout << "Destruct" << std::endl;
                delete &device_buffer_;
            }

            buffer_proxy& operator=(T const& t)
            {
#if defined(__COMPUTE__ACCELERATOR__)
                device_buffer_[pos_] = t;
#else
                device_buffer_view[pos_] = t;
#endif
                //access_target::write(*target_, p_, &t);
                return *this;
            }

            buffer_proxy& operator=(buffer_proxy const& t)
            {
#if defined(__COMPUTE__ACCELERATOR__)
                device_buffer_[pos_] = *t;
#else
                device_buffer_view[pos_] = *t;
#endif
                return *this;
            }

            operator T() const
            {
#if defined(__COMPUTE__ACCELERATOR__)
                return device_buffer_[pos_];
#else
                return device_buffer_view[pos_];
#endif
            }

//            operator T&() const
//            {
//                return *p_;
//                //return device_buffer_view[0];
//            }

            T &operator*() const {
                return device_buffer_[pos_];
            }

//            operator T*() const
//            {
//                return p_;
//            }

            buffer_proxy<T> operator+(size_type pos) {
                return buffer_proxy(device_buffer_, pos_ + pos);
            }

            T &operator++() {
                ++pos_;
                return *this;
            }

            T &operator--() {
                --pos_;
                return *this;
            }

            T* operator->() const
            {
#if defined(__COMPUTE__ACCELERATOR__)
                return device_buffer_.data() + pos_;
#else
                // todo : throw exception
                return nullptr;
#endif
            }

            //T * device_ptr() const
            //{
            //    return p_;
            //}

            buffer_t<T> & get_buffer() const HPX_NOEXCEPT
            {
                return device_buffer_;
            }

            buffer_acc_t<T> get_buffer_acc() const HPX_NOEXCEPT
            {
                return device_buffer_view;
            }

        private:
            buffer_t<T> & device_buffer_;
            buffer_acc_t<T> device_buffer_view;
            // We can't operate directly on pointers, because
            // amp::array<T>.data() evalues to nullptr on host
            // We have to count positions to behave like a pointer
            uint64_t pos_;
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

            buffer_proxy(buffer_proxy<T> const &other) :
                device_buffer_(other.get_buffer()),
                device_buffer_view(other.get_buffer_acc()),
                p_(other.device_ptr())
            {}

            ~buffer_proxy() {
                delete &device_buffer_;
            }

            buffer_proxy& operator=(T const& t)
            {
#if defined(__COMPUTE__ACCELERATOR__)
                *p_ = t;
#else
                size_t idx = p_ - device_buffer_.data();
                device_buffer_view[idx] = t;
#endif
                //access_target::write(*target_, p_, &t);
                return *this;
            }

            buffer_proxy& operator=(buffer_proxy const& t)
            {
                *p_ = *t;
                return *this;
            }

            operator T() const
            {
#if defined(__COMPUTE__ACCELERATOR__)
                return *p_;
#else
                size_t idx = p_ - device_buffer_.data();
                return device_buffer_view[idx];
#endif

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
