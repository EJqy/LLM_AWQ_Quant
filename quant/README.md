mainly for quantization only, output a quantized model, then input the quantized model into runtime to run with custom low-bit gemm kernel.

* pre-requirements by pip install: transformers==4.46.0(如果是qwen3这种比较新的模型，那么去掉指定的版本，安装默认版本), zstandard, datasets==3.6.0, accelerate
