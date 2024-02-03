//
// MATLAB Compiler: 8.1 (R2020b)
// Date: Sat Jan 27 20:38:41 2024
// Arguments:
// "-B""macro_default""-W""cpplib:libSpGH,all""-T""link:lib""-d""/home/hzyu/git/
// GaussianVI/quadrature/SparseGH/libSpGH/for_testing""-v""/home/hzyu/git/Gaussi
// anVI/quadrature/SparseGH/nwspgr.m"
//

#define EXPORTING_libSpGH 1
#include "libSpGH.h"

static HMCRINSTANCE _mcr_inst = NULL; /* don't use nullptr; this may be either C or C++ */

#ifdef __cplusplus
extern "C" { // sbcheck:ok:extern_c
#endif

static int mclDefaultPrintHandler(const char *s)
{
    return mclWrite(1 /* stdout */, s, sizeof(char)*strlen(s));
}

#ifdef __cplusplus
} /* End extern C block */
#endif

#ifdef __cplusplus
extern "C" { // sbcheck:ok:extern_c
#endif

static int mclDefaultErrorHandler(const char *s)
{
    int written = 0;
    size_t len = 0;
    len = strlen(s);
    written = mclWrite(2 /* stderr */, s, sizeof(char)*len);
    if (len > 0 && s[ len-1 ] != '\n')
        written += mclWrite(2 /* stderr */, "\n", sizeof(char));
    return written;
}

#ifdef __cplusplus
} /* End extern C block */
#endif

/* This symbol is defined in shared libraries. Define it here
 * (to nothing) in case this isn't a shared library. 
 */
#ifndef LIB_libSpGH_C_API
#define LIB_libSpGH_C_API /* No special import/export declaration */
#endif

LIB_libSpGH_C_API 
bool MW_CALL_CONV libSpGHInitializeWithHandlers(
    mclOutputHandlerFcn error_handler,
    mclOutputHandlerFcn print_handler)
{
    int bResult = 0;
    if (_mcr_inst)
        return true;
    if (!mclmcrInitialize())
        return false;
    {
        mclCtfStream ctfStream = 
            mclGetEmbeddedCtfStream((void *)(libSpGHInitializeWithHandlers));
        if (ctfStream) {
            bResult = mclInitializeComponentInstanceEmbedded(&_mcr_inst,
                                                             error_handler, 
                                                             print_handler,
                                                             ctfStream);
            mclDestroyStream(ctfStream);
        } else {
            bResult = 0;
        }
    }  
    if (!bResult)
    return false;
    return true;
}

LIB_libSpGH_C_API 
bool MW_CALL_CONV libSpGHInitialize(void)
{
    return libSpGHInitializeWithHandlers(mclDefaultErrorHandler, mclDefaultPrintHandler);
}

LIB_libSpGH_C_API 
void MW_CALL_CONV libSpGHTerminate(void)
{
    if (_mcr_inst)
        mclTerminateInstance(&_mcr_inst);
}

LIB_libSpGH_C_API 
void MW_CALL_CONV libSpGHPrintStackTrace(void) 
{
    char** stackTrace;
    int stackDepth = mclGetStackTrace(&stackTrace);
    int i;
    for(i=0; i<stackDepth; i++)
    {
        mclWrite(2 /* stderr */, stackTrace[i], sizeof(char)*strlen(stackTrace[i]));
        mclWrite(2 /* stderr */, "\n", sizeof(char)*strlen("\n"));
    }
    mclFreeStackTrace(&stackTrace, stackDepth);
}


LIB_libSpGH_C_API 
bool MW_CALL_CONV mlxNwspgr(int nlhs, mxArray *plhs[], int nrhs, mxArray *prhs[])
{
    return mclFeval(_mcr_inst, "nwspgr", nlhs, plhs, nrhs, prhs);
}

LIB_libSpGH_CPP_API 
void MW_CALL_CONV nwspgr(int nargout, mwArray& nodes, mwArray& weights, const mwArray& 
                         type, const mwArray& dim, const mwArray& k, const mwArray& sym)
{
    mclcppMlfFeval(_mcr_inst, "nwspgr", nargout, 2, 4, &nodes, &weights, &type, &dim, &k, &sym);
}

