//
//  Created by FMLAB4 on 09.12.17.
//  Copyright © 2017 FMLAB4. All rights reserved.
//

#ifndef UsingGnuplot_h
#define UsingGnuplot_h

#include <fstream>
#include <vector>
#include <map>
#include <limits>
#include <cmath>
#include <cstdio>
#include <boost/tuple/tuple.hpp>
#include <boost/foreach.hpp>

// Warn about use of deprecated functions.
#define GNUPLOT_DEPRECATE_WARN
#include "gnuplot-iostream.h"

#ifndef M_PI
#    define M_PI 3.14159265358979323846
#endif

// http://stackoverflow.com/a/1658429
#ifdef _WIN32
#include <windows.h>
inline void mysleep(unsigned millis) {
    ::Sleep(millis);
}
#else
#include <unistd.h>
inline void mysleep(unsigned millis) {
    ::usleep(millis * 1000);
}
#endif



#endif /* UsingGnuplot_h */
