//  Copyright (c) 2016 Marcin Copik
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

/// \file parallel/executors/kernel.hpp

#if !defined(HPX_PARALLEL_KERNEL)
#define HPX_PARALLEL_KERNEL

#include <hpx/config.hpp>
#include <hpx/runtime/serialization/serialize.hpp>
#include <hpx/parallel/config/inline_namespace.hpp>

namespace hpx { namespace parallel {

	struct DefaultKernelName;
	struct GenericKernelName;
}}

namespace hpx { namespace parallel { 

	HPX_INLINE_NAMESPACE(v3) { namespace detail
	{

		template<typename F, typename KernelName>
		struct kernel
		{
			using functor_type = F;
			using kernel_name = KernelName;
		
			kernel(const F & f) : f(f) {}

			template<typename ...Args>
			typename std::result_of<F(Args...)>::type operator()(Args &&... args) const
			{
				return f( std::forward<Args>(args)... );
			}		

		private:
			F f;
		};

		template<typename Kernel, typename Enable = void>
		struct get_kernel_name
		{
			typedef GenericKernelName kernel_name;
		};

		template<typename F, typename KernelName>
		struct get_kernel_name<kernel<F, KernelName>, typename std::enable_if<std::is_same<KernelName, DefaultKernelName>::value>::type>
		{
			typedef typename kernel<F, KernelName>::functor_type kernel_name;
		};

		template<typename F, typename KernelName>
		struct get_kernel_name<kernel<F, KernelName>, typename std::enable_if<!std::is_same<KernelName, DefaultKernelName>::value>::type>
		{
			typedef typename kernel<F, KernelName>::kernel_name kernel_name;
		};
	}}

	template<typename KernelName = DefaultKernelName, typename F>
	v3::detail::kernel<F, KernelName> make_kernel(F && f)
	{
		return v3::detail::kernel<typename hpx::util::decay<F>::type, KernelName>(f);
	}

    template <typename Kernel>
    struct get_kernel_name
      : v3::detail::get_kernel_name<typename hpx::util::decay<Kernel>::type>
    {};
}}

#endif
