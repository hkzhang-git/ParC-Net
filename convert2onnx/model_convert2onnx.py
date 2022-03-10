import torch
from cvnets import get_model
from options.opts import get_training_arguments

# initialization
opts = get_training_arguments()
model = get_model(opts)
model.eval()

input = torch.randn(1, 3, 224, 224)

# torch.onnx.export(model, input, "./converted_models/edgeformer.onnx", opset_version=12)

torch.onnx.export(model, input, "./converted_models/mobilevit.onnx", opset_version=12)