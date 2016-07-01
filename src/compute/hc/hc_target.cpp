//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_HC)

#include <hpx/exception.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/assert.hpp>

#include <hpx/compute/hc/config.hpp>
#include <hpx/compute/hc/target.hpp>

#include <hc.hpp>

#include <string>
#include <exception>

namespace hpx { namespace compute { namespace hc
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
            void initialize(device_t & device);
        private:
            std::function<void ()> callback_lambda;
            hpx::runtime* rt_;
            ::hc::completion_future hc_marker;
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
                "hpx::compute::hc::future_data::marker_callback"
            );
        }
        /*struct marker_functor
        {
            future_data * ptr;
            void operator()()
            {

            }
        }*/
        void future_data::initialize(device_t & device)
        {
            try {
                hc_marker = device.create_marker();
                future_data * ptr = this;
                hpx::runtime * rt_ptr = this->rt_;
                //hc_marker.get();
                //boost::intrusive_ptr<future_data> keep_alive(this);
                printf("ptr: %p\n", (void*)ptr);
                //printf("ptr: %p\n", (void*)functor.ptr);
                callback_lambda =
                    [ptr]() {
                        // propagate exception
                        try {

                            printf("ptr: %p\n", (void*)ptr);
                            auto dummy = ptr;
                            printf("ptr: %p\n", (void*)dummy);
                            ptr->hc_marker.get();

                            marker_callback(ptr);

                        } catch(...) {
                            marker_callback(ptr,
                                boost::current_exception());
                        }
                    };
                hc_marker.then(callback_lambda
                    /*[ptr]() {
                        // propagate exception
                        try {

                            printf("ptr: %p\n", (void*)ptr);
                            auto dummy = ptr;
                            printf("ptr: %p\n", (void*)dummy);
                            ptr->hc_marker.get();

                            marker_callback(ptr);

                        } catch(...) {
                            marker_callback(ptr,
                                boost::current_exception());
                        }
                    }*/
                );
            } catch(exception_t & exc) {
                // callback was not called, release object
                lcos::detail::intrusive_ptr_release(this);

                // report error
                HPX_THROW_EXCEPTION(kernel_error,
                    "hc::detail::future_data::future_data()",
                    std::string("accelerator_view::create_marker() andfailed") +
                        std::string(exc.what())
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
        device_(device),
        device_view_(nullptr),
        locality_(hpx::find_here())
    {
        HPX_ASSERT(device_ >= -1);
        // TODO: recheck if this implementation is what we want
        if(device_ != -1) {
            auto devices = ::hc::accelerator::get_all();
            HPX_ASSERT(devices.size() > static_cast<std::size_t>(device_));
            device_view_ = new device_t(devices[device_].create_view());
        } else {
            // Use default accelerator
            device_view_ = new device_t(::hc::accelerator().create_view());
        }
    }

    target::native_handle_type::~native_handle_type()
    {
        // TODO: do we need it here?
        device_view_->flush();
        delete device_view_;
    }

    target::native_handle_type::native_handle_type(
            target::native_handle_type && rhs) HPX_NOEXCEPT
      : device_(rhs.device_),
        device_view_(new device_t(*rhs.device_view_)),
        locality_(rhs.locality_)
    {
        //rhs.device_view_ = nullptr;
        rhs.locality_ = hpx::invalid_id;
    }

    target::native_handle_type& target::native_handle_type::operator=(
        target::native_handle_type && rhs) HPX_NOEXCEPT
    {
        if (this == &rhs)
            return *this;

        device_ = rhs.device_;
        device_view_ = rhs.device_view_;
        locality_ = rhs.locality_;
        device_view_ = new device_t(*rhs.device_view_);
        rhs.locality_ = hpx::invalid_id;
        return *this;
    }

    ///////////////////////////////////////////////////////////////////////////
    void target::synchronize() const
    {
        if (!handle_.device_view_)
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "hc::target::synchronize",
                "no view of accelerator available");
        }

        try {
            // TODO: is synchronized correctly implemented?
            handle_.device_view_->wait();
        } catch (exception_t & exc) {

            HPX_THROW_EXCEPTION(kernel_error,
                "hc::target::synchronize",
                std::string("wait() on accelerator_view failed: ") +
                std::string(exc.what()));
        }
    }

    hpx::future<void> target::get_future() const
    {
        typedef detail::future_data shared_state_type;
        boost::intrusive_ptr<shared_state_type> p(new shared_state_type());
        p.get()->initialize(*handle_.device_view_);
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

