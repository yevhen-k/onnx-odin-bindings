# ONNX Odin SqueezeNet Inference Demo

This repo is a word for word (with slight enhancements) translation of ONNX C API [example](https://github.com/microsoft/onnxruntime/blob/v1.4.0/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp)

1. Download ONNX library

```bash
export ONNXRUNTIME_VERSION=1.17.3
curl https://github.com/microsoft/onnxruntime/releases/download/v${ONNXRUNTIME_VERSION}/onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz -Lso onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz
tar -xvf onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION}.tgz && \
    mv onnxruntime-linux-x64-gpu-${ONNXRUNTIME_VERSION} /thirdparty/onnxruntime
```

2. Get ONNX-Odin bindings
```bash
export ONNXRUNTIME_VERSION=1.17.3
curl https://raw.githubusercontent.com/yevhen-k/onnx-odin-bindings/ONNX-${ONNXRUNTIME_VERSION}_OdinLinux-0.0.2/onnxbinding.odin -Lso onnxbinding.odin
```

3. Edit [onnxbinding.odin](onnxbinding.odin) if necessary to adjust package name or foreign import of `libonnxruntime.so`
```golang
// ...
package onnx_bindings
// ...
when ODIN_OS == .Linux do foreign import onnx "/thirdparty/onnxruntime/lib/libonnxruntime.so"
// ...
```

4. Get SqueezeNet model (using squeezenet version 1.3)
```bash
curl https://github.com/onnx/models/raw/main/validated/vision/classification/squeezenet/model/squeezenet1.0-8.onnx -Lso squeezenet1.0-8.onnx
```

5. Build and run

With the use of `rpath`:
```bash
cd ..
odin build SqueezeNet -extra-linker-flags:"-Wl,-rpath=/thirdparty/onnxruntime/lib/" -out:SqueezeNet/odin_onnx_example
cd SqueezeNet && ./odin_onnx_example
```

Or with the use of `LD_LIBRARY_PATH`:
```bash
cd ..
odin build SqueezeNet -out:SqueezeNet/odin_onnx_example
cd SqueezeNet && LD_LIBRARY_PATH=/thirdparty/onnxruntime/lib/ ./odin_onnx_example
```

## References
- Model squeezenet.onnx: https://github.com/onnx/models/tree/main/validated/vision/classification/squeezenet
- ONNX C API example: https://github.com/microsoft/onnxruntime/blob/v1.4.0/csharp/test/Microsoft.ML.OnnxRuntime.EndToEndTests.Capi/C_Api_Sample.cpp
- ONNX Odin bindings: https://github.com/yevhen-k/onnx-odin-bindings