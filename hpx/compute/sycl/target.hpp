///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_SYCL_TARGET_HPP
#define HPX_COMPUTE_SYCL_TARGET_HPP

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SYCL)

#include <hpx/runtime/runtime_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/spinlock.hpp>
#include <hpx/config/warnings_prefix.hpp>
#include <hpx/compute/sycl/config.hpp>
#include <hpx/compute/sycl/get_targets.hpp>

#include <mutex>
#include <string>
#include <utility>


namespace hpx { namespace compute { namespace sycl
{
    namespace detail
    {
        struct HPX_EXPORT runtime_registration_wrapper
        {
            runtime_registration_wrapper(hpx::runtime* rt);
            ~runtime_registration_wrapper();

            hpx::runtime* rt_;
        };
    }

    ///////////////////////////////////////////////////////////////////////////
    struct HPX_EXPORT target
    {
    public:
        struct HPX_EXPORT native_handle_type
        {
            typedef hpx::lcos::local::spinlock mutex_type;

            native_handle_type(int device = -1);

            ~native_handle_type();

            native_handle_type(const native_handle_type & rhs) HPX_NOEXCEPT;
            native_handle_type(native_handle_type && rhs) HPX_NOEXCEPT;

            native_handle_type& operator=(const native_handle_type & rhs) HPX_NOEXCEPT;
            native_handle_type& operator=(native_handle_type && rhs) HPX_NOEXCEPT;

            int get_device_idx() const HPX_NOEXCEPT
            {
                return device_idx_;
            }

            device_t & get_device() const HPX_NOEXCEPT
            {
                return device_;
            }

            queue_t & get_queue() const HPX_NOEXCEPT
            {
                return queue_;
            }

        private:
            friend struct target;

            mutable mutex_type mtx_;
            int device_idx_;
            mutable device_t device_;
            mutable queue_t queue_;
        };

        // Constructs default target
        target() HPX_NOEXCEPT
          : handle_(),
            locality_(hpx::find_here())
        {}

        // Constructs target from a given device ID
        explicit target(int device)
          : handle_(device),
            locality_(hpx::find_here())
        {}

        target(const target & rhs) HPX_NOEXCEPT
          : handle_(rhs.handle_),
            locality_(rhs.locality_)
        {}

        target(target && rhs) HPX_NOEXCEPT
          : handle_(std::move(rhs.handle_)),
            locality_(rhs.locality_)
        {
            rhs.locality_ = hpx::invalid_id;
        }

        target& operator=(const target & rhs) HPX_NOEXCEPT
        {
            if(&rhs != this)
            {
                handle_ = rhs.handle_;
            }
            return *this;
        }

        target& operator=(target && rhs) HPX_NOEXCEPT
        {
            if(&rhs != this)
            {
                handle_ = std::move(rhs.handle_);
                rhs.locality_ = hpx::invalid_id;
            }
            return *this;
        }

        native_handle_type const& native_handle() const
        {
            return handle_;
        }

        hpx::id_type const& get_locality() const HPX_NOEXCEPT
        {
            return locality_;
        }

        static std::vector<target> get_local_targets()
        {
            return sycl::get_local_targets();
        }

        static hpx::future<std::vector<target> >
            get_targets(hpx::id_type const& locality)
        {
            //FIXME: infinite recursion, call the true function
            return sycl::get_targets(locality);
        }

        friend bool operator==(target const& lhs, target const& rhs)
        {
            return lhs.handle_.get_device_idx() == rhs.handle_.get_device_idx() &&
                lhs.locality_ == rhs.locality_;
        }

        void synchronize() const;

        hpx::future<void> get_future() const;

        static void async_handler(cl::sycl::exception_list el);

    private:
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar & handle_.device_idx_ & locality_;
        }

        native_handle_type handle_;
        hpx::id_type locality_;
    };

    HPX_API_EXPORT target& get_default_target();
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
#endif
