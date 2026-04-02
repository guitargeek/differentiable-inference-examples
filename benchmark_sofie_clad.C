#include "model.hxx"

#include <Math/CladDerivator.h>

using Sess = TMVA_SOFIE_model::Session;

// Wrapper functions for Clad
float my_func(Sess const &session, float *tensor_x)
{
   float out = 0.;
   TMVA_SOFIE_model::doInfer(session, tensor_x, &out);
   return out;
}

float my_func_wrapper(Sess const &session, float *tensor_x)
{
   return my_func(session, tensor_x);
}

#if 1
namespace TMVA::Experimental::SOFIE {

inline void Gemm_Call_reverse_forw(float *output, bool transa, bool transb, int m, int n, int k, float alpha, const float *A, const float *B, float beta, const float *C, float *_d_output, bool _d_transa, bool _d_transb, int _d_m, int _d_n, int _d_k, float _d_alpha, const float *_d_A, const float *_d_B, float _d_beta, const float *_d_C, clad::restore_tracker &_tracker0) {
    char _d_ct = 0;
    char ct = 't';
    char _d_cn = 0;
    char cn = 'n';
    bool _cond0 = transa;
    const int *_d_lda = nullptr;
    const int *lda = _cond0 ? &k : &m;
    bool _cond1 = transb;
    const int *_d_ldb = nullptr;
    const int *ldb = _cond1 ? &n : &k;
    const int *_d_ldc = &_d_m;
    const int *ldc = &m;
    {
        bool _cond2 = C != nullptr;
        if (_cond2) {
            std::copy(C, C + m * n, output);
        }
    }
    bool _cond3 = transa;
    bool _cond4 = transb;
    TMVA::Experimental::SOFIE::BLAS::sgemm_(_cond3 ? &ct : &cn, _cond4 ? &ct : &cn, &m, &n, &k, &alpha, A, lda, B, ldb, &beta, output, ldc);
}


inline void Relu_reverse_forw(float *output, const float *input, int size, float *_d_output, const float *_d_input, int _d_size, clad::restore_tracker &_tracker0) {
    //clad::tape<bool> _cond0 = {};
    unsigned long _t0 = 0;
    int _d_i = 0;
    for (int i = 0; i < size; i++) {
        _t0++;
        //_tracker0.store(output[i]);
        //clad::push(_cond0, (input[i] > 0.F));
        //output[i] = clad::back(_cond0) ? input[i] : 0.F;
        output[i] = input[i] > 0.F ? input[i] : 0.F;
    }
}


inline void Copy_reverse_forw(float *output, const float *input, int size, float *_d_output, const float *_d_input, int _d_size, clad::restore_tracker &_tracker0) {
    unsigned long _t0 = 0;
    int _d_i = 0;
    for (int i = 0; i < size; i++) {
        _t0++;
        //_tracker0.store(output[i]);
        output[i] = input[i];
    }
}

}
namespace TMVA_SOFIE_model {

inline void doInfer_pullback(const Sess &session, const float *tensor_input, float *tensor_output, Sess *_d_session, float *_d_tensor_input, float *_d_tensor_output) {
    size_t _d_j = 0UL;
    size_t j = 0UL;
    size_t _d_y_index = 0UL;
    size_t y_index = 0UL;
    size_t _d_j0 = 0UL;
    size_t j2 = 0UL;
    size_t _d_y_index0 = 0UL;
    size_t y_index0 = 0UL;
    size_t _d_j1 = 0UL;
    size_t j3 = 0UL;
    size_t _d_y_index1 = 0UL;
    size_t y_index1 = 0UL;
    size_t _d_j2 = 0UL;
    size_t j4 = 0UL;
    size_t _d_y_index2 = 0UL;
    size_t y_index2 = 0UL;
    size_t _d_j3 = 0UL;
    size_t j5 = 0UL;
    size_t _d_y_index3 = 0UL;
    size_t y_index3 = 0UL;
    size_t _d_j4 = 0UL;
    size_t j6 = 0UL;
    size_t _d_y_index4 = 0UL;
    size_t y_index4 = 0UL;
    float *&_d_tensor_net0bias = (*_d_session).tensor_net0bias;
    float *const &tensor_net0bias = session.tensor_net0bias;
    float *&_d_tensor_net0weight = (*_d_session).tensor_net0weight;
    float *const &tensor_net0weight = session.tensor_net0weight;
    float *&_d_tensor_net10bias = (*_d_session).tensor_net10bias;
    float *const &tensor_net10bias = session.tensor_net10bias;
    float *&_d_tensor_net10weight = (*_d_session).tensor_net10weight;
    float *const &tensor_net10weight = session.tensor_net10weight;
    float *&_d_tensor_net2bias = (*_d_session).tensor_net2bias;
    float *const &tensor_net2bias = session.tensor_net2bias;
    float *&_d_tensor_net2weight = (*_d_session).tensor_net2weight;
    float *const &tensor_net2weight = session.tensor_net2weight;
    float *&_d_tensor_net4bias = (*_d_session).tensor_net4bias;
    float *const &tensor_net4bias = session.tensor_net4bias;
    float *&_d_tensor_net4weight = (*_d_session).tensor_net4weight;
    float *const &tensor_net4weight = session.tensor_net4weight;
    float *&_d_tensor_net6bias = (*_d_session).tensor_net6bias;
    float *const &tensor_net6bias = session.tensor_net6bias;
    float *&_d_tensor_net6weight = (*_d_session).tensor_net6weight;
    float *const &tensor_net6weight = session.tensor_net6weight;
    float *&_d_tensor_net8bias = (*_d_session).tensor_net8bias;
    float *const &tensor_net8bias = session.tensor_net8bias;
    float *&_d_tensor_net8weight = (*_d_session).tensor_net8weight;
    float *const &tensor_net8weight = session.tensor_net8weight;
    float *&_d_tensor_relu = (*_d_session).tensor_relu;
    float *const &tensor_relu = session.tensor_relu;
    float *&_d_tensor_relu_1 = (*_d_session).tensor_relu_1;
    float *const &tensor_relu_1 = session.tensor_relu_1;
    float *&_d_tensor_relu_2 = (*_d_session).tensor_relu_2;
    float *const &tensor_relu_2 = session.tensor_relu_2;
    float *&_d_tensor_relu_3 = (*_d_session).tensor_relu_3;
    float *const &tensor_relu_3 = session.tensor_relu_3;
    float *&_d_tensor_relu_4 = (*_d_session).tensor_relu_4;
    float *const &tensor_relu_4 = session.tensor_relu_4;
    {
        j = 0;
        y_index = 128 * j;
        TMVA::Experimental::SOFIE::Copy(tensor_relu + y_index, tensor_net0bias, 128);
    }
    clad::restore_tracker _tracker0 = {};
    TMVA::Experimental::SOFIE::Gemm_Call_reverse_forw(tensor_relu, true, false, 128, 1, 20, 1, (float *)tensor_net0weight, (float *)tensor_input, 1, nullptr, _d_tensor_relu, false, false, 0, 0, 0, 0, (float *)_d_tensor_net0weight, (float *)_d_tensor_input, 0, nullptr, _tracker0);
    clad::restore_tracker _tracker1 = {};
    TMVA::Experimental::SOFIE::Relu_reverse_forw(tensor_relu, tensor_relu, 128, _d_tensor_relu, _d_tensor_relu, 0, _tracker1);
    {
        j2 = 0;
        y_index0 = 128 * j2;
        clad::restore_tracker _tracker2 = {};
        TMVA::Experimental::SOFIE::Copy_reverse_forw(tensor_relu_1 + y_index0, tensor_net2bias, 128, _d_tensor_relu_1 + y_index0, _d_tensor_net2bias, 0, _tracker2);
    }
    clad::restore_tracker _tracker3 = {};
    TMVA::Experimental::SOFIE::Gemm_Call_reverse_forw(tensor_relu_1, true, false, 128, 1, 128, 1, (float *)tensor_net2weight, (float *)tensor_relu, 1, nullptr, _d_tensor_relu_1, false, false, 0, 0, 0, 0, (float *)_d_tensor_net2weight, (float *)_d_tensor_relu, 0, nullptr, _tracker3);
    clad::restore_tracker _tracker4 = {};
    TMVA::Experimental::SOFIE::Relu_reverse_forw(tensor_relu_1, tensor_relu_1, 128, _d_tensor_relu_1, _d_tensor_relu_1, 0, _tracker4);
    {
        j3 = 0;
        y_index1 = 128 * j3;
        clad::restore_tracker _tracker5 = {};
        TMVA::Experimental::SOFIE::Copy_reverse_forw(tensor_relu_2 + y_index1, tensor_net4bias, 128, _d_tensor_relu_2 + y_index1, _d_tensor_net4bias, 0, _tracker5);
    }
    clad::restore_tracker _tracker6 = {};
    TMVA::Experimental::SOFIE::Gemm_Call_reverse_forw(tensor_relu_2, true, false, 128, 1, 128, 1, (float *)tensor_net4weight, (float *)tensor_relu_1, 1, nullptr, _d_tensor_relu_2, false, false, 0, 0, 0, 0, (float *)_d_tensor_net4weight, (float *)_d_tensor_relu_1, 0, nullptr, _tracker6);
    clad::restore_tracker _tracker7 = {};
    TMVA::Experimental::SOFIE::Relu_reverse_forw(tensor_relu_2, tensor_relu_2, 128, _d_tensor_relu_2, _d_tensor_relu_2, 0, _tracker7);
    {
        j4 = 0;
        y_index2 = 128 * j4;
        clad::restore_tracker _tracker8 = {};
        TMVA::Experimental::SOFIE::Copy_reverse_forw(tensor_relu_3 + y_index2, tensor_net6bias, 128, _d_tensor_relu_3 + y_index2, _d_tensor_net6bias, 0, _tracker8);
    }
    clad::restore_tracker _tracker9 = {};
    TMVA::Experimental::SOFIE::Gemm_Call_reverse_forw(tensor_relu_3, true, false, 128, 1, 128, 1, (float *)tensor_net6weight, (float *)tensor_relu_2, 1, nullptr, _d_tensor_relu_3, false, false, 0, 0, 0, 0, (float *)_d_tensor_net6weight, (float *)_d_tensor_relu_2, 0, nullptr, _tracker9);
    clad::restore_tracker _tracker10 = {};
    TMVA::Experimental::SOFIE::Relu_reverse_forw(tensor_relu_3, tensor_relu_3, 128, _d_tensor_relu_3, _d_tensor_relu_3, 0, _tracker10);
    {
        j5 = 0;
        y_index3 = 128 * j5;
        clad::restore_tracker _tracker11 = {};
        TMVA::Experimental::SOFIE::Copy_reverse_forw(tensor_relu_4 + y_index3, tensor_net8bias, 128, _d_tensor_relu_4 + y_index3, _d_tensor_net8bias, 0, _tracker11);
    }
    clad::restore_tracker _tracker12 = {};
    TMVA::Experimental::SOFIE::Gemm_Call_reverse_forw(tensor_relu_4, true, false, 128, 1, 128, 1, (float *)tensor_net8weight, (float *)tensor_relu_3, 1, nullptr, _d_tensor_relu_4, false, false, 0, 0, 0, 0, (float *)_d_tensor_net8weight, (float *)_d_tensor_relu_3, 0, nullptr, _tracker12);
    clad::restore_tracker _tracker13 = {};
    TMVA::Experimental::SOFIE::Relu_reverse_forw(tensor_relu_4, tensor_relu_4, 128, _d_tensor_relu_4, _d_tensor_relu_4, 0, _tracker13);
    {
        j6 = 0;
        y_index4 = j6;
        TMVA::Experimental::SOFIE::Copy(tensor_output + y_index4, tensor_net10bias, 1);
    }
    clad::restore_tracker _tracker14 = {};
    TMVA::Experimental::SOFIE::Gemm_Call_reverse_forw(tensor_output, true, false, 1, 1, 128, 1, (float *)tensor_net10weight, (float *)tensor_relu_4, 1, nullptr, _d_tensor_output, false, false, 0, 0, 0, 0, (float *)_d_tensor_net10weight, (float *)_d_tensor_relu_4, 0, nullptr, _tracker14);
    {
        bool _r46 = false;
        bool _r47 = false;
        int _r48 = 0;
        int _r49 = 0;
        int _r50 = 0;
        float _r51 = 0.F;
        float _r52 = 0.F;
        clad::custom_derivatives::TMVA::Experimental::SOFIE::Gemm_Call_pullback(tensor_output, true, false, 1, 1, 128, 1, (float *)tensor_net10weight, (float *)tensor_relu_4, 1, nullptr, _d_tensor_output, &_r46, &_r47, &_r48, &_r49, &_r50, &_r51, (float *)_d_tensor_net10weight, (float *)_d_tensor_relu_4, &_r52, nullptr);
    }
    {
        {
            int _r45 = 0;
            clad::custom_derivatives::TMVA::Experimental::SOFIE::Copy_pullback(tensor_output + y_index4, tensor_net10bias, 1, _d_tensor_output + y_index4, _d_tensor_net10bias, &_r45);
        }
        _d_j4 += _d_y_index4;
    }
    {
        int _r44 = 0;
        clad::custom_derivatives::TMVA::Experimental::SOFIE::Relu_pullback(tensor_relu_4, tensor_relu_4, 128, _d_tensor_relu_4, _d_tensor_relu_4, &_r44);
    }
    {
        bool _r37 = false;
        bool _r38 = false;
        int _r39 = 0;
        int _r40 = 0;
        int _r41 = 0;
        float _r42 = 0.F;
        float _r43 = 0.F;
        clad::custom_derivatives::TMVA::Experimental::SOFIE::Gemm_Call_pullback(tensor_relu_4, true, false, 128, 1, 128, 1, (float *)tensor_net8weight, (float *)tensor_relu_3, 1, nullptr, _d_tensor_relu_4, &_r37, &_r38, &_r39, &_r40, &_r41, &_r42, (float *)_d_tensor_net8weight, (float *)_d_tensor_relu_3, &_r43, nullptr);
    }
    {
        {
            int _r36 = 0;
            clad::custom_derivatives::TMVA::Experimental::SOFIE::Copy_pullback(tensor_relu_4 + y_index3, tensor_net8bias, 128, _d_tensor_relu_4 + y_index3, _d_tensor_net8bias, &_r36);
        }
        _d_j3 += 128 * _d_y_index3;
    }
    {
        int _r35 = 0;
        clad::custom_derivatives::TMVA::Experimental::SOFIE::Relu_pullback(tensor_relu_3, tensor_relu_3, 128, _d_tensor_relu_3, _d_tensor_relu_3, &_r35);
    }
    {
        bool _r28 = false;
        bool _r29 = false;
        int _r30 = 0;
        int _r31 = 0;
        int _r32 = 0;
        float _r33 = 0.F;
        float _r34 = 0.F;
        clad::custom_derivatives::TMVA::Experimental::SOFIE::Gemm_Call_pullback(tensor_relu_3, true, false, 128, 1, 128, 1, (float *)tensor_net6weight, (float *)tensor_relu_2, 1, nullptr, _d_tensor_relu_3, &_r28, &_r29, &_r30, &_r31, &_r32, &_r33, (float *)_d_tensor_net6weight, (float *)_d_tensor_relu_2, &_r34, nullptr);
    }
    {
        {
            int _r27 = 0;
            clad::custom_derivatives::TMVA::Experimental::SOFIE::Copy_pullback(tensor_relu_3 + y_index2, tensor_net6bias, 128, _d_tensor_relu_3 + y_index2, _d_tensor_net6bias, &_r27);
        }
        _d_j2 += 128 * _d_y_index2;
    }
    {
        int _r26 = 0;
        clad::custom_derivatives::TMVA::Experimental::SOFIE::Relu_pullback(tensor_relu_2, tensor_relu_2, 128, _d_tensor_relu_2, _d_tensor_relu_2, &_r26);
    }
    {
        bool _r19 = false;
        bool _r20 = false;
        int _r21 = 0;
        int _r22 = 0;
        int _r23 = 0;
        float _r24 = 0.F;
        float _r25 = 0.F;
        clad::custom_derivatives::TMVA::Experimental::SOFIE::Gemm_Call_pullback(tensor_relu_2, true, false, 128, 1, 128, 1, (float *)tensor_net4weight, (float *)tensor_relu_1, 1, nullptr, _d_tensor_relu_2, &_r19, &_r20, &_r21, &_r22, &_r23, &_r24, (float *)_d_tensor_net4weight, (float *)_d_tensor_relu_1, &_r25, nullptr);
    }
    {
        {
            int _r18 = 0;
            clad::custom_derivatives::TMVA::Experimental::SOFIE::Copy_pullback(tensor_relu_2 + y_index1, tensor_net4bias, 128, _d_tensor_relu_2 + y_index1, _d_tensor_net4bias, &_r18);
        }
        _d_j1 += 128 * _d_y_index1;
    }
    {
        int _r17 = 0;
        clad::custom_derivatives::TMVA::Experimental::SOFIE::Relu_pullback(tensor_relu_1, tensor_relu_1, 128, _d_tensor_relu_1, _d_tensor_relu_1, &_r17);
    }
    {
        bool _r10 = false;
        bool _r11 = false;
        int _r12 = 0;
        int _r13 = 0;
        int _r14 = 0;
        float _r15 = 0.F;
        float _r16 = 0.F;
        clad::custom_derivatives::TMVA::Experimental::SOFIE::Gemm_Call_pullback(tensor_relu_1, true, false, 128, 1, 128, 1, (float *)tensor_net2weight, (float *)tensor_relu, 1, nullptr, _d_tensor_relu_1, &_r10, &_r11, &_r12, &_r13, &_r14, &_r15, (float *)_d_tensor_net2weight, (float *)_d_tensor_relu, &_r16, nullptr);
    }
    {
        {
            int _r9 = 0;
            clad::custom_derivatives::TMVA::Experimental::SOFIE::Copy_pullback(tensor_relu_1 + y_index0, tensor_net2bias, 128, _d_tensor_relu_1 + y_index0, _d_tensor_net2bias, &_r9);
        }
        _d_j0 += 128 * _d_y_index0;
    }
    {
        int _r8 = 0;
        clad::custom_derivatives::TMVA::Experimental::SOFIE::Relu_pullback(tensor_relu, tensor_relu, 128, _d_tensor_relu, _d_tensor_relu, &_r8);
    }
    {
        bool _r1 = false;
        bool _r2 = false;
        int _r3 = 0;
        int _r4 = 0;
        int _r5 = 0;
        float _r6 = 0.F;
        float _r7 = 0.F;
        clad::custom_derivatives::TMVA::Experimental::SOFIE::Gemm_Call_pullback(tensor_relu, true, false, 128, 1, 20, 1, (float *)tensor_net0weight, (float *)tensor_input, 1, nullptr, _d_tensor_relu, &_r1, &_r2, &_r3, &_r4, &_r5, &_r6, (float *)_d_tensor_net0weight, (float *)_d_tensor_input, &_r7, nullptr);
    }
    {
        {
            int _r0 = 0;
            clad::custom_derivatives::TMVA::Experimental::SOFIE::Copy_pullback(tensor_relu + y_index, tensor_net0bias, 128, _d_tensor_relu + y_index, _d_tensor_net0bias, &_r0);
        }
        _d_j += 128 * _d_y_index;
    }
}

inline void doInfer_reverse_forw(const Session &session, const float *tensor_input, float *tensor_output, const Session &_d_session, const float *_d_tensor_input, float *_d_tensor_output, clad::restore_tracker &_tracker0) {
    float *const &_d_tensor_net0bias = _d_session.tensor_net0bias;
    float *const &tensor_net0bias = session.tensor_net0bias;
    float *const &_d_tensor_net0weight = _d_session.tensor_net0weight;
    float *const &tensor_net0weight = session.tensor_net0weight;
    float *const &_d_tensor_net10bias = _d_session.tensor_net10bias;
    float *const &tensor_net10bias = session.tensor_net10bias;
    float *const &_d_tensor_net10weight = _d_session.tensor_net10weight;
    float *const &tensor_net10weight = session.tensor_net10weight;
    float *const &_d_tensor_net2bias = _d_session.tensor_net2bias;
    float *const &tensor_net2bias = session.tensor_net2bias;
    float *const &_d_tensor_net2weight = _d_session.tensor_net2weight;
    float *const &tensor_net2weight = session.tensor_net2weight;
    float *const &_d_tensor_net4bias = _d_session.tensor_net4bias;
    float *const &tensor_net4bias = session.tensor_net4bias;
    float *const &_d_tensor_net4weight = _d_session.tensor_net4weight;
    float *const &tensor_net4weight = session.tensor_net4weight;
    float *const &_d_tensor_net6bias = _d_session.tensor_net6bias;
    float *const &tensor_net6bias = session.tensor_net6bias;
    float *const &_d_tensor_net6weight = _d_session.tensor_net6weight;
    float *const &tensor_net6weight = session.tensor_net6weight;
    float *const &_d_tensor_net8bias = _d_session.tensor_net8bias;
    float *const &tensor_net8bias = session.tensor_net8bias;
    float *const &_d_tensor_net8weight = _d_session.tensor_net8weight;
    float *const &tensor_net8weight = session.tensor_net8weight;
    float *const &_d_tensor_relu = _d_session.tensor_relu;
    float *const &tensor_relu = session.tensor_relu;
    float *const &_d_tensor_relu_1 = _d_session.tensor_relu_1;
    float *const &tensor_relu_1 = session.tensor_relu_1;
    float *const &_d_tensor_relu_2 = _d_session.tensor_relu_2;
    float *const &tensor_relu_2 = session.tensor_relu_2;
    float *const &_d_tensor_relu_3 = _d_session.tensor_relu_3;
    float *const &tensor_relu_3 = session.tensor_relu_3;
    float *const &_d_tensor_relu_4 = _d_session.tensor_relu_4;
    float *const &tensor_relu_4 = session.tensor_relu_4;
    {
        size_t _d_j = 0;
        size_t j = 0;
        size_t _d_y_index = 0UL;
        size_t y_index = 128 * j;
        TMVA::Experimental::SOFIE::Copy_reverse_forw(tensor_relu + y_index, tensor_net0bias, 128, _d_tensor_relu + y_index, _d_tensor_net0bias, 0, _tracker0);
    }
    TMVA::Experimental::SOFIE::Gemm_Call_reverse_forw(tensor_relu, true, false, 128, 1, 20, 1, (float *)tensor_net0weight, (float *)tensor_input, 1, nullptr, _d_tensor_relu, false, false, 0, 0, 0, 0, (float *)_d_tensor_net0weight, (float *)_d_tensor_input, 0, nullptr, _tracker0);
    TMVA::Experimental::SOFIE::Relu_reverse_forw(tensor_relu, tensor_relu, 128, _d_tensor_relu, _d_tensor_relu, 0, _tracker0);
    {
        size_t _d_j0 = 0;
        size_t j2 = 0;
        size_t _d_y_index0 = 0UL;
        size_t y_index0 = 128 * j2;
        TMVA::Experimental::SOFIE::Copy_reverse_forw(tensor_relu_1 + y_index0, tensor_net2bias, 128, _d_tensor_relu_1 + y_index0, _d_tensor_net2bias, 0, _tracker0);
    }
    TMVA::Experimental::SOFIE::Gemm_Call_reverse_forw(tensor_relu_1, true, false, 128, 1, 128, 1, (float *)tensor_net2weight, (float *)tensor_relu, 1, nullptr, _d_tensor_relu_1, false, false, 0, 0, 0, 0, (float *)_d_tensor_net2weight, (float *)_d_tensor_relu, 0, nullptr, _tracker0);
    TMVA::Experimental::SOFIE::Relu_reverse_forw(tensor_relu_1, tensor_relu_1, 128, _d_tensor_relu_1, _d_tensor_relu_1, 0, _tracker0);
    {
        size_t _d_j1 = 0;
        size_t j3 = 0;
        size_t _d_y_index1 = 0UL;
        size_t y_index1 = 128 * j3;
        TMVA::Experimental::SOFIE::Copy_reverse_forw(tensor_relu_2 + y_index1, tensor_net4bias, 128, _d_tensor_relu_2 + y_index1, _d_tensor_net4bias, 0, _tracker0);
    }
    TMVA::Experimental::SOFIE::Gemm_Call_reverse_forw(tensor_relu_2, true, false, 128, 1, 128, 1, (float *)tensor_net4weight, (float *)tensor_relu_1, 1, nullptr, _d_tensor_relu_2, false, false, 0, 0, 0, 0, (float *)_d_tensor_net4weight, (float *)_d_tensor_relu_1, 0, nullptr, _tracker0);
    TMVA::Experimental::SOFIE::Relu_reverse_forw(tensor_relu_2, tensor_relu_2, 128, _d_tensor_relu_2, _d_tensor_relu_2, 0, _tracker0);
    {
        size_t _d_j2 = 0;
        size_t j4 = 0;
        size_t _d_y_index2 = 0UL;
        size_t y_index2 = 128 * j4;
        TMVA::Experimental::SOFIE::Copy_reverse_forw(tensor_relu_3 + y_index2, tensor_net6bias, 128, _d_tensor_relu_3 + y_index2, _d_tensor_net6bias, 0, _tracker0);
    }
    TMVA::Experimental::SOFIE::Gemm_Call_reverse_forw(tensor_relu_3, true, false, 128, 1, 128, 1, (float *)tensor_net6weight, (float *)tensor_relu_2, 1, nullptr, _d_tensor_relu_3, false, false, 0, 0, 0, 0, (float *)_d_tensor_net6weight, (float *)_d_tensor_relu_2, 0, nullptr, _tracker0);
    TMVA::Experimental::SOFIE::Relu_reverse_forw(tensor_relu_3, tensor_relu_3, 128, _d_tensor_relu_3, _d_tensor_relu_3, 0, _tracker0);
    {
        size_t _d_j3 = 0;
        size_t j5 = 0;
        size_t _d_y_index3 = 0UL;
        size_t y_index3 = 128 * j5;
        TMVA::Experimental::SOFIE::Copy_reverse_forw(tensor_relu_4 + y_index3, tensor_net8bias, 128, _d_tensor_relu_4 + y_index3, _d_tensor_net8bias, 0, _tracker0);
    }
    TMVA::Experimental::SOFIE::Gemm_Call_reverse_forw(tensor_relu_4, true, false, 128, 1, 128, 1, (float *)tensor_net8weight, (float *)tensor_relu_3, 1, nullptr, _d_tensor_relu_4, false, false, 0, 0, 0, 0, (float *)_d_tensor_net8weight, (float *)_d_tensor_relu_3, 0, nullptr, _tracker0);
    TMVA::Experimental::SOFIE::Relu_reverse_forw(tensor_relu_4, tensor_relu_4, 128, _d_tensor_relu_4, _d_tensor_relu_4, 0, _tracker0);
    {
        size_t _d_j4 = 0;
        size_t j6 = 0;
        size_t _d_y_index4 = 0UL;
        size_t y_index4 = j6;
        TMVA::Experimental::SOFIE::Copy_reverse_forw(tensor_output + y_index4, tensor_net10bias, 1, _d_tensor_output + y_index4, _d_tensor_net10bias, 0, _tracker0);
    }
    TMVA::Experimental::SOFIE::Gemm_Call_reverse_forw(tensor_output, true, false, 1, 1, 128, 1, (float *)tensor_net10weight, (float *)tensor_relu_4, 1, nullptr, _d_tensor_output, false, false, 0, 0, 0, 0, (float *)_d_tensor_net10weight, (float *)_d_tensor_relu_4, 0, nullptr, _tracker0);
}

}

void my_func_pullback(const Sess &session, float *tensor_x, float _d_y, Sess *_d_session, float *_d_tensor_x) {
    float _d_out = 0.F;
    float out = 0.;
    clad::restore_tracker _tracker0 = {};
    TMVA_SOFIE_model::doInfer_reverse_forw(session, tensor_x, &out, (*_d_session), _d_tensor_x, &_d_out, _tracker0);
    _d_out += _d_y;
    {
        _tracker0.restore();
        TMVA_SOFIE_model::doInfer_pullback(session, tensor_x, &out, _d_session, _d_tensor_x, &_d_out);
    }
}
#endif

void benchmark_sofie_ad()
{
   // Let's go Clad!
   //clad::gradient(my_func_wrapper, "tensor_x");

   // Get a function pointer to the pullback. If you are unsure what the
   // signature is, try to cast the pullback to some function pointer, like
   // static_cast<void (*)(float)>(my_func_pullback) in the interpreter, and
   // the compiler will tell you what the real signature is.
   using Grad_t = void (*)(const Sess &, float *, float, Sess *, float *);

   // Get the functions from the interpreter (remove semicolor to get the code printed)
   //gInterpreter->ProcessLine("TMVA_SOFIE_model::doInfer_pullback");
   //gInterpreter->ProcessLine("my_func_pullback");
   auto grad = reinterpret_cast<Grad_t>(gInterpreter->ProcessLine("my_func_pullback;"));
   //gInterpreter->ProcessLine("TMVA_SOFIE_model::doInfer_reverse_forw");
   //gInterpreter->ProcessLine("TMVA::Experimental::SOFIE::Gemm_Call_reverse_forw");
   //gInterpreter->ProcessLine("TMVA::Experimental::SOFIE::Relu_reverse_forw");
   //gInterpreter->ProcessLine("TMVA::Experimental::SOFIE::Copy_reverse_forw");

   std::vector<float> input(20, 1.0f);

   // A trick: pre-allocate session struct both for the forward pass and
   // backward pass, to that no memory allocation of intermediate tensors has
   // to happend in the gradient.
   Sess s;
   Sess d_s;

   // Calculate numerical gradient
   auto numDiff = [&](int i) {
      const float eps = 1e-4;
      std::vector<float> p{input};
      p[i] = input[i] - eps;
      float funcValDown = my_func(s, p.data());
      p[i] = input[i] + eps;
      float funcValUp = my_func(s, p.data());
      return (funcValUp - funcValDown) / (2 * eps);
   };


   // Calculate gradient with Clad
   float output = 0.0f;
   float grad_output[20]{};
   grad(s, input.data(), 1.0, &d_s, grad_output);

   for (std::size_t i = 0; i < input.size(); ++i) {
      std::cout << i << ":" << std::endl;
      std::cout << "  numr : " << numDiff(i) << std::endl;
      std::cout << "  clad : " << grad_output[i] << std::endl;
   }


   // ----------------------------------
   // Benchmark
   // ----------------------------------

   const int N = 100;

   using Clock = std::chrono::high_resolution_clock;

   // ----------------------------------
   // Warm-up (important!)
   // ----------------------------------
   for (int i = 0; i < 100; ++i) {
      TMVA_SOFIE_model::doInfer(s, input.data(), &output);
      grad(s, input.data(), 1.0, &d_s, grad_output);
   }

   // ----------------------------------
   // Forward benchmark
   // ----------------------------------
   auto t1 = Clock::now();

   for (int i = 0; i < N; ++i) {
      TMVA_SOFIE_model::doInfer(s, input.data(), &output);
   }

   auto t2 = Clock::now();

   double forward_time = std::chrono::duration<double>(t2 - t1).count() / N;

   // ----------------------------------
   // Gradient benchmark
   // ----------------------------------
   auto t3 = Clock::now();

   for (int i = 0; i < N; ++i) {
      std::fill_n(grad_output, 20, 0.);
      grad(s, input.data(), 1.0, &d_s, grad_output);
   }

   auto t4 = Clock::now();

   double grad_time = std::chrono::duration<double>(t4 - t3).count() / N;

   // ----------------------------------
   // Results
   // ----------------------------------
   std::cout << "\n--- SOFIE + Clad Benchmark ---\n";
   std::cout << "Forward time: " << forward_time * 1e6 << " us\n";
   std::cout << "Gradient time: " << grad_time * 1e6 << " us\n";

   std::cout << "Grad / Forward ratio: "
             << grad_time / forward_time << std::endl;
}
