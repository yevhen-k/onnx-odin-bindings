get-header:
    wget -O onnxruntime_c_api.h https://raw.githubusercontent.com/microsoft/onnxruntime/v1.17.3/include/onnxruntime/core/session/onnxruntime_c_api.h

preprocess: get-header
    gcc -E -std=c99 onnxruntime_c_api.h > onnxruntime_c_api_prep.h

cleanup-preprocess: preprocess
    perl -pi -e 's/onnxruntime_c_api.h/onnxruntime_c_api_prep.h/g' onnxruntime_c_api_prep.h
    perl -i -nle 'print if !/^\#/' onnxruntime_c_api_prep.h 
    perl -pi -e 's/__\w+__//g' onnxruntime_c_api_prep.h
    perl -pi -e 's/\(\(.+\)\)//g' onnxruntime_c_api_prep.h
    perl -pi -e 's/__restrict//g' onnxruntime_c_api_prep.h

dump-ast-json: cleanup-preprocess
    echo ORT_API_VERSION::$(cat onnxruntime_c_api.h | grep "#define ORT_API_VERSION" | sed 's/.* //') > ORT_API_VERSION.version
    python pycparser.c_json.py > dump.json

parse-ast: dump-ast-json 
    python translate_header.py --libonnxruntime-path="/thirdparty/onnxruntime/lib/libonnxruntime.so" --os="Linux" --package="onnx_bindings"