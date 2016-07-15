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
            typedef std::ptrdiff_t difference_type;
            typedef T value_type;
            /// Initialize pointer with begin position of device data


            buffer_proxy(buffer_t<T> * device_buffer) HPX_NOEXCEPT :
                device_buffer_(device_buffer),
                device_buffer_view(*device_buffer_)
            {std::cout << "construct " << std::endl;}

            buffer_proxy(const buffer_acc_t<T> & device_buffer) HPX_NOEXCEPT :
                device_buffer_(nullptr),
                device_buffer_view(device_buffer)
            {std::cout << "construct " << std::endl;}

            buffer_proxy(buffer_proxy const &other) :
                device_buffer_(other.device_buffer_),
                device_buffer_view(other.device_buffer_view)
            {std::cout << "construct " << std::endl;}

            ~buffer_proxy() {
                std::cout << "Destruct" << std::endl;
                //delete &device_buffer_;
            }

//            buffer_proxy& operator=(T const& t)
//            {
//#if defined(__COMPUTE__ACCELERATOR__)
//                device_buffer_view[pos_] = t;
//#else
//                std::cout << "Write: " << t << std::endl;
//                device_buffer_view[pos_] = t;
//#endif
//                //access_target::write(*target_, p_, &t);
//                return *this;
//            }
//
//            buffer_proxy& operator=(buffer_proxy const& t)
//            {
//#if defined(__COMPUTE__ACCELERATOR__)
//                device_buffer_view[pos_] = *t;
//#else
//                device_buffer_view[pos_] = *t;
//#endif
//                return *this;
//            }

            buffer_proxy& operator=(T const& t)
            {
#if defined(__COMPUTE__ACCELERATOR__)
                device_buffer_view[0] = t;
#else
                std::cout << "Write: " << t << std::endl;
                device_buffer_view[0] = t;
#endif
                //access_target::write(*target_, p_, &t);
                return *this;
            }

            buffer_proxy& operator=(buffer_proxy const& t)
            {
#if defined(__COMPUTE__ACCELERATOR__)
                device_buffer_view[0] = *t;
#else
                device_buffer_view[0] = *t;
#endif
                return *this;
            }

            T & operator[](std::ptrdiff_t pos)
            {
#if defined(__COMPUTE__ACCELERATOR__)
                return device_buffer_view[pos];
#else
                std::cout << "Write: "<< std::endl;
                return device_buffer_view[pos];
#endif
                //access_target::write(*target_, p_, &t);
            }

            operator T() const
            {
#if defined(__COMPUTE__ACCELERATOR__)
                return device_buffer_view[0];
#else
                return device_buffer_view[0];
#endif
            }

            //operator T&()
            //{
            //    return device_buffer_view[0];
            //}

            T &operator*() const {
                return device_buffer_view[0];
            }

//            operator T*() const
//            {
//                return p_;
//            }

            //buffer_proxy<T> operator+(difference_type pos) {
            //    return buffer_proxy(&device_buffer_, pos_ + pos);
            //}

            //buffer_proxy<T> & operator+=(difference_type pos) {
            //    pos_ += pos;
            //    return *this;
            //}

            //buffer_proxy<T> & operator++() {
            //    std::cout << "Increment " << pos_ << " " << pos_ + 1 << std::endl;
            //    ++pos_;
            //    return *this;
            //}

            //buffer_proxy<T> & operator--() {
            //    --pos_;
            //    return *this;
            //}

//            T* operator->() const
//            {
//#if defined(__COMPUTE__ACCELERATOR__)
//                return device_buffer_.data();
//#else
//                // todo : throw exception
//                return nullptr;
//#endif
//            }

            //T * device_ptr() const
            //{
            //    return p_;
            //}

            buffer_t<T> * get_buffer() const HPX_NOEXCEPT
            {
                return device_buffer_;
            }

            const buffer_acc_t<T> & get_buffer_acc() const HPX_NOEXCEPT
            {
                return device_buffer_view;
            }

            buffer_acc_t<T> section(std::ptrdiff_t pos,
                    std::ptrdiff_t size) const HPX_NOEXCEPT
            {
                if (pos > 0 && size > 0) {
                    return device_buffer_->section(index<1>(pos),
                            global_size<1>(size));
                } else if (size > 0) {
                    return device_buffer_->section(global_size<1>(size));
                } else if (pos > 0) {
                    return device_buffer_->section(index<1>(pos));
                } else {
                    return device_buffer_view;
                }
            }
        private:
            buffer_t<T> * device_buffer_;
            buffer_acc_t<T> device_buffer_view;
            // We can't operate directly on pointers, because
            // amp::array<T>.data() evalues to nullptr on host
            // We have to count positions to behave like a pointer
            //uint64_t pos_;
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
