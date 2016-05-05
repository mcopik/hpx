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
		struct kernel_extract_name
		{
			typedef GenericKernelName kernel_name;
		};

		template<typename F, typename KernelName>
		struct kernel_extract_name<kernel<F, KernelName>, typename std::enable_if<std::is_same<KernelName, DefaultKernelName>::value>::type>
		{
			typedef typename kernel<F, KernelName>::functor_type kernel_name;
		};

		template<typename F, typename KernelName>
		struct kernel_extract_name<kernel<F, KernelName>, typename std::enable_if<!std::is_same<KernelName, DefaultKernelName>::value>::type>
		{
			typedef typename kernel<F, KernelName>::kernel_name kernel_name;
		};
	}}

	template<typename KernelName = DefaultKernelName, typename F, typename F_decay = typename hpx::util::decay<F>::type>
	v3::detail::kernel<F_decay, KernelName> make_kernel(F && f)
	{
		return v3::detail::kernel<F_decay, KernelName>(f);
	}

	HPX_INLINE_NAMESPACE(v3) { namespace detail
	{
		template<typename F1, typename F2>
		struct wrap_kernel_helper
		{
			template<typename F12, typename F22>
			static auto call(F12 && old_functor, F22 && new_functor) -> decltype(std::forward<F2>(new_functor))
			{
				return std::forward<F2>(new_functor);
			}
		};

		template<typename F, typename KernelName, typename F2>
		struct wrap_kernel_helper<v3::detail::kernel<F, KernelName>, F2>
		{
			template<typename F1, typename F3>
			static auto call(F1 && old_functor, F3 && new_functor) -> decltype(make_kernel<KernelName>(new_functor))
			{
				return make_kernel<KernelName>(new_functor);
			}
		};
	}}

	template<typename F1, typename F2>
	auto wrap_kernel(F1 && old_functor, F2 && new_functor) -> 
		decltype( detail::wrap_kernel_helper<
				typename std::decay<F1>::type, 
				typename std::decay<F2>::type
			>::call( std::forward<F1>(old_functor), std::forward<F2>(new_functor)) )
	{
		return detail::wrap_kernel_helper<
				typename std::decay<F1>::type, 
				typename std::decay<F2>::type
			>::call( std::forward<F1>(old_functor), std::forward<F2>(new_functor));
	}
/*
	//TODO: universal reference
	template<typename F, typename KernelName, typename F2>
	v3::detail::kernel<typename std::decay<F2>::type, KernelName> wrap_kernel(const v3::detail::kernel<typename std::decay<F>::type, KernelName> & old_functor, F2 && new_functor)// -> decltype(make_kernel<KernelName>(new_functor))
	{
		return make_kernel<KernelName>(new_functor);
	}

	template<typename F, typename KernelName, typename F2>
	v3::detail::kernel<typename std::decay<F2>::type, KernelName> wrap_kernel(v3::detail::kernel<typename std::decay<F>::type, KernelName> && old_functor, F2 && new_functor) //-> decltype(make_kernel<KernelName>(new_functor))
	{
		return make_kernel<KernelName>(new_functor);
	}*/

    template <typename Kernel>
    struct kernel_extract_name
      : v3::detail::kernel_extract_name<typename hpx::util::decay<Kernel>::type>
    {};
}}

#endif
