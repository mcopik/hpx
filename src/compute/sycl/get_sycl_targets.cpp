//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <hpx/config.hpp>

#if defined(HPX_HAVE_SYCL)

#include <hpx/exception.hpp>
#include <hpx/async.hpp>
#include <hpx/lcos/future.hpp>
#include <hpx/runtime_fwd.hpp>
#include <hpx/runtime/actions/plain_action.hpp>
#include <hpx/runtime/find_here.hpp>
#include <hpx/runtime/naming/id_type.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/runtime/serialization/vector.hpp>
#include <hpx/runtime/threads/thread_helpers.hpp>

#include <hpx/compute/sycl/target.hpp>

#include <string>
#include <vector>

#include <SYCL/sycl.hpp>

namespace hpx { namespace compute { namespace sycl
{
    std::vector<target> get_local_targets()
    {
        auto devices = cl::sycl::device::get_devices();
        std::vector<target> targets;
        targets.reserve(devices.size());

        for(size_t i = 0; i < devices.size(); ++i)
        {
            targets.emplace_back(target(i));
        }

        return targets;
    }
}}}

HPX_PLAIN_ACTION(hpx::compute::sycl::get_local_targets,
    compute_sycl_get_targets_action);

namespace hpx { namespace compute { namespace sycl
{
    hpx::future<std::vector<target> > get_targets(hpx::id_type const& locality)
    {
        if (locality == hpx::find_here())
            return hpx::make_ready_future(get_local_targets());

        return hpx::async(compute_sycl_get_targets_action(), locality);
    }
}}}

#endif

