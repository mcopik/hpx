///////////////////////////////////////////////////////////////////////////////
//  Copyright (c) 2016 Thomas Heller
//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
///////////////////////////////////////////////////////////////////////////////

#ifndef HPX_COMPUTE_AMP_TARGET_HPP
#define HPX_COMPUTE_AMP_TARGET_HPP

#include <hpx/config.hpp>
#include <hpx/compute/amp/config.hpp>

#if defined(HPX_HAVE_AMP)
#include <hpx/runtime/runtime_fwd.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/lcos/local/spinlock.hpp>

#if !defined(__COMPUTE__ACCELERATOR__)
#include <hpx/runtime/serialization/serialization_fwd.hpp>
#endif

#include <mutex>
#include <string>
#include <utility>

#include <hpx/config/warnings_prefix.hpp>

namespace hpx { namespace compute { namespace amp
{

    ///////////////////////////////////////////////////////////////////////////
    struct HPX_EXPORT target
    {
    private:
        HPX_MOVABLE_ONLY(target);

    public:
        struct HPX_EXPORT native_handle_type
        {
            typedef hpx::lcos::local::spinlock mutex_type;

            HPX_MOVABLE_ONLY(native_handle_type);

            native_handle_type(int device = 0);

            ~native_handle_type();

            native_handle_type(native_handle_type && rhs) HPX_NOEXCEPT;

            native_handle_type& operator=(native_handle_type && rhs) HPX_NOEXCEPT;

            accelerator_view get_device() const HPX_NOEXCEPT
            {
                return device_;
            }

            hpx::id_type const& get_locality() const HPX_NOEXCEPT
            {
                return locality_;
            }

        private:
            friend struct target;

            mutable mutex_type mtx_;
            int device_;
            accelerator_view device_view;
            hpx::id_type locality_;
        };

        // Constructs default target
        target() HPX_NOEXCEPT {}

        // Constructs target from a given device ID
        explicit target(int device)
          : handle_(device)
        {}

        target(target && rhs) HPX_NOEXCEPT
          : handle_(std::move(rhs.handle_))
        {}

        target& operator=(target && rhs) HPX_NOEXCEPT
        {
            handle_ = std::move(rhs.handle_);
            return *this;
        }

        native_handle_type const& native_handle() const
        {
            return handle_;
        }

        void synchronize() const;

        hpx::future<void> get_future() const;

    private:
#if !defined(__COMPUTE__ACCELERATOR__)
        friend class hpx::serialization::access;

        template <typename Archive>
        void serialize(Archive& ar, const unsigned int version)
        {
            ar & handle_.device_ & handle_.locality_;
        }
#endif
        native_handle_type handle_;
    };

    HPX_API_EXPORT target& get_default_target();
}}}

#include <hpx/config/warnings_suffix.hpp>

#endif
#endif
