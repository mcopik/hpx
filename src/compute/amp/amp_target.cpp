//  Copyright (c) 2016 Hartmut Kaiser
//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_AMP)

#include <hpx/exception.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>
#include <hpx/util/assert.hpp>

#include <hpx/compute/amp/target.hpp>

#include <string>

#include <amp.h>

namespace hpx { namespace compute { namespace amp
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
            hpx::register_thread(rt_, "amp", ec);
        }
        runtime_registration_wrapper::~runtime_registration_wrapper()
        {
            // Unregister the thread from HPX, this should be done once in
            // the end before the external thread exists.
            hpx::unregister_thread(rt_);
        }

//        ///////////////////////////////////////////////////////////////////////
//        struct future_data : lcos::detail::future_data<void>
//        {
//        private:
//            static void CUDART_CB stream_callback(cudaStream_t stream,
//                cudaError_t error, void* user_data);
//
//        public:
//            future_data(cudaStream_t stream);
//
//        private:
//            hpx::runtime* rt_;
//        };
//
//        struct release_on_exit
//        {
//            release_on_exit(future_data* data)
//              : data_(data)
//            {}
//
//            ~release_on_exit()
//            {
//                // release the shared state
//                lcos::detail::intrusive_ptr_release(data_);
//            }
//
//            future_data* data_;
//        };
//
//        ///////////////////////////////////////////////////////////////////////
//        void CUDART_CB future_data::stream_callback(cudaStream_t stream,
//            cudaError_t error, void* user_data)
//        {
//            future_data* this_ = static_cast<future_data*>(user_data);
//
//            runtime_registration_wrapper wrap(this_->rt_);
//
//            // We need to run this as an HPX thread ...
//            hpx::applier::register_thread_nullary(
//                [this_, error] ()
//                {
//                    release_on_exit on_exit(this_);
//
//                    if (error != cudaSuccess)
//                    {
//                        this_->set_exception(
//                            HPX_GET_EXCEPTION(kernel_error,
//                                "cuda::detail::future_data::stream_callback()",
//                                std::string("cudaStreamAddCallback failed: ") +
//                                    cudaGetErrorString(error))
//                        );
//                        return;
//                    }
//
//                    this_->set_data(hpx::util::unused);
//                },
//                "hpx::compute::cuda::future_data::stream_callback"
//            );
//        }
//
//        future_data::future_data(cudaStream_t stream)
//           : rt_(hpx::get_runtime_ptr())
//        {
//            // Hold on to the shared state on behalf of the cuda runtime
//            // right away as the callback could be called immediately.
//            lcos::detail::intrusive_ptr_add_ref(this);
//
//            cudaError_t error = cudaStreamAddCallback(
//                stream, stream_callback, this, 0);
//            if (error != cudaSuccess)
//            {
//                // callback was not called, release object
//                lcos::detail::intrusive_ptr_release(this);
//
//                // report error
//                HPX_THROW_EXCEPTION(kernel_error,
//                    "cuda::detail::future_data::future_data()",
//                    std::string("cudaStreamAddCallback failed: ") +
//                        cudaGetErrorString(error));
//            }
//        }
    }

    target::native_handle_type::native_handle_type(int device) :
        device_(device),
	    device_view_(nullptr),
	    locality_(hpx::find_here())
    {
        HPX_ASSERT(device_ >= -1);
        // TODO: recheck if this implementation is what we want
        if(device_ != -1) {
            auto devices = Concurrency::accelerator::get_all();
            HPX_ASSERT(devices.size() > static_cast<std::size_t>(device_));
            device_view_ = new device_t(
                devices[device_].create_view()
            );
        } else {
            // Use default accelerator
            device_view_ = new device_t(
                Concurrency::accelerator().create_view()
            );
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
        // TODO: is it supposed to be incorrect after copy construct?
        // If yes, then pointer to device_t should be nullptr
        rhs.locality_ = hpx::invalid_id;
    }

    target::native_handle_type& target::native_handle_type::operator=(
        target::native_handle_type && rhs) HPX_NOEXCEPT
    {
        // TODO: same as above
        if (this == &rhs)
            return *this;

        device_ = rhs.device_;
        device_view_ = rhs.device_view_;
        locality_ = rhs.locality_;
        rhs.device_view_ = new device_t(*rhs.device_view_);
        rhs.locality_ = hpx::invalid_id;
        return *this;
    }

    ///////////////////////////////////////////////////////////////////////////
    void target::synchronize() const
    {
        if (handle_.device_view_)
        {
            HPX_THROW_EXCEPTION(invalid_status,
                "amp::target::synchronize",
                "no view of accelerator available");
        }

        try {
            // TODO: is synchronized correctly implemented?
            handle_.device_view_->wait();
        } catch (Concurrency::runtime_exception & exc) {

            HPX_THROW_EXCEPTION(kernel_error,
                "amp::target::synchronize",
                std::string("wait() on accelerator_view failed: ") +
                std::string(exc.what()));
        }
    }

    // TODO: implement
//    hpx::future<void> target::get_future() const
//    {
////        typedef detail::future_data shared_state_type;
////        shared_state_type* p = new shared_state_type(handle_.stream_);
//        return hpx::traits::future_access<hpx::future<void> >::create<(nullptr);
//    }

    ///////////////////////////////////////////////////////////////////////////
    target& get_default_target()
    {
        static target target_;
        return target_;
    }
}}}

#endif

