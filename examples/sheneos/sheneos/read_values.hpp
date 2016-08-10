//  Copyright (c) 2007-2012 Hartmut Kaiser
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(HPX_SHENEOS_READ_VALUES_AUG_08_2011_1225PM)
#define HPX_SHENEOS_READ_VALUES_AUG_08_2011_1225PM

#include <H5Cpp.h>
#include <H5pubconf.h>

#if !defined(H5_HAVE_THREADSAFE)
    #error sheneos requires threadsafe HDF5 libraries
#endif

#include <boost/cstdint.hpp>

#include <string>

#include "dimension.hpp"

///////////////////////////////////////////////////////////////////////////////
// Interpolation helper functions related to HDF5.
namespace sheneos
{
    ///////////////////////////////////////////////////////////////////////////
    /// Extract the lower and upper bounds of a data range from \a datafilename.
    boost::uint64_t extract_data_range(std::string const& datafilename,
        char const* name, double& minval, double& maxval, double& delta,
        std::size_t start = 0, std::size_t end = std::size_t(-1));

    ///////////////////////////////////////////////////////////////////////////
    /// Extract a 1D data slice from \a datafilename.
    void extract_data(std::string const& datafilename, char const* name,
        double* values, hsize_t offset, hsize_t count);

    ///////////////////////////////////////////////////////////////////////////
    /// Extract a 3D data slice from \a datafilename.
    void extract_data(std::string const& datafilename, char const* name,
        double* values, dimension const& dimx, dimension const& dimy,
        dimension const& dimz);
}

#endif

