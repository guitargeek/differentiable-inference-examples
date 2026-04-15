#include "model.hxx"

#include "TInterpreter.h"

#include "TMVA/SOFIE_common.hxx"

#include <chrono>

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

void benchmark_sofie_clad()
{
   // Let's go Clad!
   clad::gradient(my_func_wrapper, "tensor_x");

   // Get a function pointer to the pullback. If you are unsure what the
   // signature is, try to cast the pullback to some function pointer, like
   // static_cast<void (*)(float)>(my_func_pullback) in the interpreter, and
   // the compiler will tell you what the real signature is.
   using Grad_t = void (*)(const Sess &, float *, float, Sess *, float *);

   // Get the functions from the interpreter (remove semicolor to get the code printed)
   // gInterpreter->ProcessLine("TMVA_SOFIE_model::doInfer_pullback");
   // gInterpreter->ProcessLine("my_func_pullback");
   auto grad = reinterpret_cast<Grad_t>(gInterpreter->ProcessLine("my_func_pullback;"));
   // gInterpreter->ProcessLine("TMVA_SOFIE_model::doInfer_reverse_forw");
   // gInterpreter->ProcessLine("TMVA_SOFIE_model::doInfer_pullback");
   // gInterpreter->ProcessLine("TMVA::Experimental::SOFIE::Relu_pullback");
   // gInterpreter->ProcessLine("TMVA::Experimental::SOFIE::Copy_pullback");
   // gInterpreter->ProcessLine("TMVA::Experimental::SOFIE::Fill_pullback");

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

   const int N = 10000;

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

   std::cout << "Grad / Forward ratio: " << grad_time / forward_time << std::endl;
}
