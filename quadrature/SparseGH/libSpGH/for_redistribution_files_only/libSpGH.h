//
// MATLAB Compiler: 8.1 (R2020b)
// Date: Sat Jan 27 20:38:41 2024
// Arguments:
// "-B""macro_default""-W""cpplib:libSpGH,all""-T""link:lib""-d""/home/hzyu/git/
// GaussianVI/quadrature/SparseGH/libSpGH/for_testing""-v""/home/hzyu/git/Gaussi
// anVI/quadrature/SparseGH/nwspgr.m"
//

#ifndef libSpGH_h
#define libSpGH_h 1

#if defined(__cplusplus) && !defined(mclmcrrt_h) && defined(__linux__)
#  pragma implementation "mclmcrrt.h"
#endif
#include "mclmcrrt.h"
#include "mclcppclass.h"
#ifdef __cplusplus
extern "C" { // sbcheck:ok:extern_c
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_libSpGH_C_API 
#define LIB_libSpGH_C_API /* No special import/export declaration */
#endif

/* GENERAL LIBRARY FUNCTIONS -- START */

extern LIB_libSpGH_C_API 
bool MW_CALL_CONV libSpGHInitializeWithHandlers(
       mclOutputHandlerFcn error_handler, 
       mclOutputHandlerFcn print_handler);

extern LIB_libSpGH_C_API 
bool MW_CALL_CONV libSpGHInitialize(void);

extern LIB_libSpGH_C_API 
void MW_CALL_CONV libSpGHTerminate(void);

extern LIB_libSpGH_C_API 
void MW_CALL_CONV libSpGHPrintStackTrace(void);

/* GENERAL LIBRARY FUNCTIONS -- END */

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

extern LIB_libSpGH_C_API 
bool MW_CALL_CONV mlxNwspgr(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[]);

/* C INTERFACE -- MLX WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */

#ifdef __cplusplus
}
#endif


/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- START */

#ifdef __cplusplus

/* On Windows, use __declspec to control the exported API */
#if defined(_MSC_VER) || defined(__MINGW64__)

#ifdef EXPORTING_libSpGH
#define PUBLIC_libSpGH_CPP_API __declspec(dllexport)
#else
#define PUBLIC_libSpGH_CPP_API __declspec(dllimport)
#endif

#define LIB_libSpGH_CPP_API PUBLIC_libSpGH_CPP_API

#else

#if !defined(LIB_libSpGH_CPP_API)
#if defined(LIB_libSpGH_C_API)
#define LIB_libSpGH_CPP_API LIB_libSpGH_C_API
#else
#define LIB_libSpGH_CPP_API /* empty! */ 
#endif
#endif

#endif

extern LIB_libSpGH_CPP_API void MW_CALL_CONV nwspgr(int nargout, mwArray& nodes, mwArray& weights, const mwArray& type, const mwArray& dim, const mwArray& k, const mwArray& sym);

/* C++ INTERFACE -- WRAPPERS FOR USER-DEFINED MATLAB FUNCTIONS -- END */
#endif

#endif
