void onnx_to_cpp()
{
   using namespace TMVA::Experimental;
   SOFIE::RModelParser_ONNX parser;
   SOFIE::RModel model = parser.Parse("./model.onnx");
   model.SetOptimizationLevel(SOFIE::OptimizationLevel::kBasic);
   model.Generate(SOFIE::Options::kNoWeightFile);
   model.PrintRequiredInputTensors();

   model.OutputGenerated("./model.hxx");
}
