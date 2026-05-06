#include "model.hxx"

#include "TInterpreter.h"

#include "TMVA/SOFIE_common.hxx"

#include <algorithm>
#include <chrono>
#include <fstream>
#include <string>
#include <vector>

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

// Median of a vector of doubles. Sorts the input.
static double median(std::vector<double> v)
{
   std::sort(v.begin(), v.end());
   const std::size_t n = v.size();
   if (n == 0) return 0.0;
   if (n % 2 == 1) return v[n / 2];
   return 0.5 * (v[n / 2 - 1] + v[n / 2]);
}

void benchmark_sofie_clad(std::string const &out_path = "results_sofie_clad.json", int n_repeats = 5)
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

   std::vector<double> forward_repeats_us;
   std::vector<double> grad_repeats_us;
   forward_repeats_us.reserve(n_repeats);
   grad_repeats_us.reserve(n_repeats);

   for (int r = 0; r < n_repeats; ++r) {
      // Forward
      auto t1 = Clock::now();
      for (int i = 0; i < N; ++i) {
         TMVA_SOFIE_model::doInfer(s, input.data(), &output);
      }
      auto t2 = Clock::now();
      forward_repeats_us.push_back(std::chrono::duration<double>(t2 - t1).count() / N * 1e6);

      // Gradient
      auto t3 = Clock::now();
      for (int i = 0; i < N; ++i) {
         std::fill_n(grad_output, 20, 0.);
         grad(s, input.data(), 1.0, &d_s, grad_output);
      }
      auto t4 = Clock::now();
      grad_repeats_us.push_back(std::chrono::duration<double>(t4 - t3).count() / N * 1e6);
   }

   const double forward_med = median(forward_repeats_us);
   const double grad_med = median(grad_repeats_us);
   const double forward_min = *std::min_element(forward_repeats_us.begin(), forward_repeats_us.end());
   const double forward_max = *std::max_element(forward_repeats_us.begin(), forward_repeats_us.end());
   const double grad_min = *std::min_element(grad_repeats_us.begin(), grad_repeats_us.end());
   const double grad_max = *std::max_element(grad_repeats_us.begin(), grad_repeats_us.end());

   // ----------------------------------
   // Results
   // ----------------------------------
   std::cout << "\n--- SOFIE + Clad Benchmark ---\n";
   std::cout << "Forward per sample: median " << forward_med << " us "
             << "(min " << forward_min << ", max " << forward_max << ") over " << n_repeats << " repeats\n";
   std::cout << "Grad    per sample: median " << grad_med << " us "
             << "(min " << grad_min << ", max " << grad_max << ") over " << n_repeats << " repeats\n";
   std::cout << "Grad / Forward ratio: " << grad_med / forward_med << std::endl;

   // ----------------------------------
   // Write JSON results so the plotting script can pick them up
   // without manual copy-paste.
   // ----------------------------------
   auto write_array = [](std::ofstream &o, std::vector<double> const &v) {
      o << "[";
      for (std::size_t i = 0; i < v.size(); ++i) {
         if (i) o << ", ";
         o << v[i];
      }
      o << "]";
   };

   std::ofstream out(out_path);
   out << "{\n";
   out << "  \"entries\": [\n";
   out << "    {\n";
   out << "      \"label\": \"SOFIE+Clad (1 thread)\",\n";
   out << "      \"forward_us\": " << forward_med << ",\n";
   out << "      \"grad_us\": " << grad_med << ",\n";
   out << "      \"forward_us_repeats\": ";
   write_array(out, forward_repeats_us);
   out << ",\n";
   out << "      \"grad_us_repeats\": ";
   write_array(out, grad_repeats_us);
   out << "\n";
   out << "    }\n";
   out << "  ]\n";
   out << "}\n";
   std::cout << "\nResults written to " << out_path << std::endl;
}
