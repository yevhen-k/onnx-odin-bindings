# Tool for translation C's ONNX header into Odin lang

Tool to translate C ONNX headers into Odin.

## Disclamers

1. The tool was made for self-educational purposes only. Use it at your own risk.
2. Linux only! At the moment translation for linux only supported. Feel free fork and add any required funcitonality to add Window and Darwin support.
3. I'm not proud of it from the software perspective, but it was fun to make.

## Prerequisites

To translate header into Oding `pycparser` is used.

Requirements:
- `pycparser`

## Quick start
```
pip install -r requirements.txt
just parse-ast
```

See [Justfile](Justfile) for more details.

## Step-by-step
1. Install `pycparser`
```bash
pip install -r requirements.txt
```
2. Get ONNX C header of required version (for example, `1.17.3`)
```bash
wget -O onnxruntime_c_api.h https://raw.githubusercontent.com/microsoft/onnxruntime/v1.17.3/include/onnxruntime/core/session/onnxruntime_c_api.h
```
3. Preprocess `onnxruntime_c_api.h`

```bash
gcc -E -std=c99 onnxruntime_c_api.h > onnxruntime_c_api_prep.h
# OR
# cpp onnxruntime_c_api.h > onnxruntime_c_api_prep.h
# OR
# clang -E onnxruntime_c_api.h > onnxruntime_c_api_prep.h
```

4. Clean-up preprocessed header file:

```bash
perl -pi -e 's/onnxruntime_c_api.h/onnxruntime_c_api_prep.h/g' onnxruntime_c_api_prep.h
perl -i -nle 'print if !/^\#/' onnxruntime_c_api_prep.h 
perl -pi -e 's/__\w+__//g' onnxruntime_c_api_prep.h
perl -pi -e 's/\(\(.+\)\)//g' onnxruntime_c_api_prep.h
perl -pi -e 's/__restrict//g' onnxruntime_c_api_prep.h
```

5. Dump and parse AST

5.1 Get ONNX Version

```bash
echo ORT_API_VERSION::$(cat onnxruntime_c_api.h | grep "#define ORT_API_VERSION" | sed 's/.* //') > ORT_API_VERSION.version
```

5.2 Dump AST
```bash
python pycparser.c_json.py > dump.json
```

5.3 Parse AST
```bash
python translate_header.py --libonnxruntime-path="/thirdparty/onnxruntime/lib/libonnxruntime.so" --os="Linux" --package="onnx_bindings"
```

This will generate [onnxbinding.odin](onnxbinding.odin) file with Odin bindings to ONNX API.
