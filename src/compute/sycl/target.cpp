//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <hpx/config.hpp>

#if defined(HPX_HAVE_SYCL)

#include <hpx/exception.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/assert.hpp>

#include <hpx/compute/sycl/config.hpp>
#include <hpx/compute/sycl/target.hpp>

#include <SYCL/sycl.hpp>

#include <string>
#include <exception>

namespace hpx { namespace compute { namespace sycl
{
    namespace detail
    {
        runtime_registration_wrapper::runtime_registration_wrapper(
                hpx::runtime* rt)
          : rt_(rt)
        {
            HPX_ASSERT(rt);

            // Register this thread with HPX, this should be done once for
            // each external OS-thread intended to invoke HPX functionality.
            // Calling this function more than once on the same thread will
            // report an error.
            hpx::error_code ec(hpx::lightweight);       // ignore errors
            hpx::register_thread(rt_, "hc", ec);
        }

        runtime_registration_wrapper::~runtime_registration_wrapper()
        {
            // Unregister the thread from HPX, this should be done once in
            // the end before the external thread exists.
            hpx::unregister_thread(rt_);
        }

        ///////////////////////////////////////////////////////////////////////
        /// Shared future state responsible for notifying
        struct future_data : lcos::detail::future_data<void>
        {
        private:
            static void marker_callback(future_data * data,
                boost::exception_ptr exc_ptr = boost::exception_ptr());
        public:
            future_data();
            ~future_data()
            {
                std::cout << "Destruct future data " << std::endl;
            }
            void initialize(queue_t & queue);
        private:
            hpx::runtime* rt_;
        };

        struct release_on_exit
        {
            release_on_exit(future_data* data)
              : data_(data)
            {}

            ~release_on_exit()
            {
                // release the shared state
                lcos::detail::intrusive_ptr_release(data_);
            }

            future_data* data_;
        };

        ///////////////////////////////////////////////////////////////////////
        void future_data::marker_callback(future_data * this_,
                boost::exception_ptr exc_ptr)
        {
            // Notify HPX runtime about executing from a "foreign" thread -
            // asynchronous result from execution supported by HC runtime,
            // not HPX.
            runtime_registration_wrapper wrap(this_->rt_);

            // We need to run this as an HPX thread ...
            hpx::applier::register_thread_nullary(
                [this_, exc_ptr] ()
                {
                    // Delete the shared state after setting future value
                    // Reference count for the shared state has been increased
                    // in marker callback creation
                    release_on_exit on_exit(this_);

                    if(exc_ptr) {
                        this_->set_exception(exc_ptr);
                    }

                    // future<void>, hence we do not transfer any data
                    this_->set_data(hpx::util::unused);
                },
                "hpx::compute::sycl::future_data::marker_callback"
            );
        }
        /*struct marker_functor
        {
            future_data * ptr;
            void operator()()
            {

            }
        }*/
        void future_data::initialize(queue_t & queue)
        {
                future_data * ptr = this;
                cl_event marker;
                //std::cout << queue.
                auto err = clEnqueueMarkerWithWaitList(
                    queue.get(),
                    0,
                    nullptr,
                    &marker);
                if(err == CL_SUCCESS)
                {
                    err = clSetEventCallback(
                        marker,
                        CL_COMPLETE,
                        [](cl_event, cl_int, void * ptr) { marker_callback(static_cast<future_data*>(ptr)); },
                        ptr
                     );
                }

                if(err != CL_SUCCESS)
                {
                    lcos::detail::intrusive_ptr_release(this);
                    // report error
                    HPX_THROW_EXCEPTION(kernel_error,
                        "sycl::detail::future_data::future_data()",
                        std::string("OpenCL error code") + std::to_string(err)
                    );
                }
        }

        future_data::future_data()
           : rt_(hpx::get_runtime_ptr())
        {
            // Hold on to the shared state on behalf of the cuda runtime
            // right away as the callback could be called immediately.
            lcos::detail::intrusive_ptr_add_ref(this);
        }
    }

    target::native_handle_type::native_handle_type(int device) :
        device_idx_(device),
        locality_(hpx::find_here())
    {
        HPX_ASSERT(device_idx_ >= -1);
        // TODO: recheck if this implementation is what we want
        auto devices = device_t::get_devices();
        //TODO: use first device when -1 has been passed. use selectors in future
        device_idx_ = device_idx_ >= 0 ? device_idx_ : 0;
        HPX_ASSERT(devices.size() > static_cast<std::size_t>(device_idx_));
        device_ = devices[device_idx_];
        queue_ = queue_t(device_);
    }

    target::native_handle_type::~native_handle_type()
    {
        queue_.wait();
    }

    target::native_handle_type::native_handle_type(
            target::native_handle_type && rhs) HPX_NOEXCEPT
      : device_idx_(rhs.device_idx_),
        device_(std::move(rhs.device_)),
        queue_(std::move(rhs.queue_)),
        locality_(rhs.locality_)
    {
        //rhs.device_view_ = nullptr;
        rhs.locality_ = hpx::invalid_id;
    }

    target::native_handle_type::native_handle_type(
            const target::native_handle_type & rhs) HPX_NOEXCEPT
      : device_idx_(rhs.device_idx_),
        device_(rhs.device_),
        queue_(rhs.queue_),
        locality_(rhs.locality_)
    {
        //rhs.device_view_ = nullptr;
        //rhs.locality_ = hpx::invalid_id;
    }

    target::native_handle_type& target::native_handle_type::operator=(
        target::native_handle_type && rhs) HPX_NOEXCEPT
    {
        if (this == &rhs)
            return *this;

        device_idx_ = rhs.device_idx_;
        device_= std::move(rhs.device_);
        locality_ = rhs.locality_;
        queue_ = std::move(rhs.queue_);
        rhs.locality_ = hpx::invalid_id;
        return *this;
    }

    target::native_handle_type& target::native_handle_type::operator=(
        const target::native_handle_type & rhs) HPX_NOEXCEPT
    {
        if (this == &rhs)
            return *this;

        device_idx_ = rhs.device_idx_;
        device_= rhs.device_;
        locality_ = rhs.locality_;
        queue_ = rhs.queue_;
        return *this;
    }
    ///////////////////////////////////////////////////////////////////////////
    void target::synchronize() const
    {
        //if (handle_.device_view_)
        //{
        //    HPX_THROW_EXCEPTION(invalid_status,
        //        "hc::target::synchronize",
        //        "no view of accelerator available");
        //}

        try {
            handle_.get_queue().wait_and_throw();
        } catch (exception_t & exc) {

            HPX_THROW_EXCEPTION(kernel_error,
                "sycl::target::synchronize",
                std::string("wait_and_throw on sycl::queue failed: ") +
                std::string(exc.what()));
        }
    }

    hpx::future<void> target::get_future() const
    {
        typedef detail::future_data shared_state_type;
        boost::intrusive_ptr<shared_state_type> p(new shared_state_type());
        p.get()->initialize(handle_.get_queue());
        return hpx::traits::future_access<hpx::future<void>>::create(p);
    }

    ///////////////////////////////////////////////////////////////////////////
    target& get_default_target()
    {
        static target target_;
        return target_;
    }
}}}

#endif


