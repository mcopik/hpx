//  Copyright (c) 2014 Thomas Heller
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#ifndef HPX_EXAMPLES_MINI_STEPPER_HPP
#define HPX_EXAMPLES_MINI_STEPPER_HPP

#include <examples/mini_ghost/params.hpp>
#include <examples/mini_ghost/global_sum.hpp>
#include <examples/mini_ghost/partition.hpp>
#include <examples/mini_ghost/send_buffer.hpp>
#include <examples/mini_ghost/spikes.hpp>
#include <examples/mini_ghost/recv_buffer.hpp>

#include <hpx/include/components.hpp>

#include <random>

namespace mini_ghost {
    template <typename Real>
    struct stepper
      : hpx::components::managed_component_base<stepper<Real> >
    {
        static const std::size_t max_num_neighbors = 6;

        typedef
            hpx::util::serialize_buffer<Real>
            buffer_type;

        stepper();

        void init(params<Real> & p);

        void run(std::size_t num_spikes, std::size_t num_tsteps);

        void set_global_sum(std::size_t generation, std::size_t which, Real value, std::size_t idx);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_global_sum, set_global_sum_action);

        void set_north_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_north_zone, set_north_zone_action);

        void set_south_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_south_zone, set_south_zone_action);

        void set_east_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_east_zone, set_east_zone_action);

        void set_west_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_west_zone, set_west_zone_action);

        void set_front_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_front_zone, set_front_zone_action);

        void set_back_zone(buffer_type buffer, std::size_t step, std::size_t var);
        HPX_DEFINE_COMPONENT_ACTION_TPL(stepper<Real>, set_back_zone, set_back_zone_action);

    private:
        std::mt19937 gen;
        std::uniform_real_distribution<Real> random;

        std::size_t rank;
        std::vector<hpx::id_type> stepper_ids;

        std::vector<spikes<Real> > spikes_;

        typedef
            partition<
                Real
              , set_global_sum_action
              , set_south_zone_action
              , set_north_zone_action
              , set_west_zone_action
              , set_east_zone_action
              , set_front_zone_action
              , set_back_zone_action
            >
            partition_type;
        std::vector<partition_type> partitions_;
    };
}

#endif
