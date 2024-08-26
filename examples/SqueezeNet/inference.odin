package squeezenet_inference_example

import "core:c"
import "core:fmt"
import "core:os"


CheckStatus :: proc(ort: ^OrtApi, status: OrtStatusPtr) {
	if status != nil {
		msg: cstring = ort.GetErrorMessage(status)
		fmt.eprintln(msg)
		ort.ReleaseStatus(status)
		os.exit(1)
	}
}


main :: proc() {
	g_ort: ^OrtApi
	if g_ort = OrtGetApiBase().GetApi(ORT_API_VERSION); cast(rawptr)g_ort == nil {
		fmt.println(">>> OrtApi is nil")
	} else {
		fmt.printfln(">>> Values: g_ort=%p, nil=%p", g_ort, nil)
	}
	fmt.printfln(">>> API (str): %s", OrtGetApiBase().GetVersionString())

	//// Get available providers:
	providers_count: c.int
	providers: [^]cstring
	g_ort.GetAvailableProviders(cast(^^^c.char)(&providers), &providers_count)
	defer g_ort.ReleaseAvailableProviders(providers, providers_count)
	fmt.println(">>> Available providers:")
	for i: c.int = 0; i < providers_count; i += 1 {
		fmt.printfln("\t%d) %s", i, providers[i])
	}
	/*
	>>> 0) TensorrtExecutionProvider
	>>> 1) CUDAExecutionProvider
	>>> 2) CPUExecutionProvider
	*/

	is_cuda_available: bool
	for i: c.int = 0; i < providers_count; i += 1 {
		if providers[i] == "CUDAExecutionProvider" {
			is_cuda_available = true
			break
		}
	}
	fmt.printfln(">>> CUDA is available: %t", is_cuda_available)

	////*************************************************************************
	//// initialize  enviroment...one enviroment per process
	//// enviroment maintains thread pools and other state info
	env: ^OrtEnv
	status: OrtStatusPtr = g_ort.CreateEnv(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, "test", &env)
	CheckStatus(g_ort, status)
	defer g_ort.ReleaseEnv(env)

	//// initialize session options if needed
	session_options: ^OrtSessionOptions
	status = g_ort.CreateSessionOptions(&session_options)
	CheckStatus(g_ort, status)
	defer g_ort.ReleaseSessionOptions(session_options)

	status = g_ort.SetIntraOpNumThreads(session_options, 1)
	CheckStatus(g_ort, status)

	//// Sets graph optimization level
	status = g_ort.SetSessionGraphOptimizationLevel(
		session_options,
		GraphOptimizationLevel.ORT_ENABLE_BASIC,
	)
	CheckStatus(g_ort, status)

	//// Enable CUDA acceleration
	if is_cuda_available {
		fmt.println(">>> Setting up CUDA...")
		status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0)
		CheckStatus(g_ort, status)
	}

	////*************************************************************************
	//// create session and load model into memory
	//// using squeezenet version 1.3
	//// URL = https://github.com/onnx/models/tree/master/squeezenet
	session: ^OrtSession
	model_path :: "squeezenet1.0-8.onnx"

	fmt.printfln(">>> Using Odin API (based on C API)")
	status = g_ort.CreateSession(env, model_path, session_options, &session)
	CheckStatus(g_ort, status)
	defer g_ort.ReleaseSession(session)

	//*************************************************************************
	input_tensor_size: c.size_t = 224 * 224 * 3

	input_tensor_values := make([dynamic]c.float, input_tensor_size)
	defer delete(input_tensor_values)

	input_node_dims := make([dynamic]c.int64_t)
	defer delete(input_node_dims)
	append(&input_node_dims, 1, 3, 224, 224)

	input_node_names := make([dynamic]cstring)
	defer delete(input_node_names)
	append(&input_node_names, "data_0")

	output_node_names := make([dynamic]cstring)
	defer delete(output_node_names)
	append(&output_node_names, "softmaxout_1")

	// initialize input data with values in [0.0, 1.0]
	for i: c.size_t = 0; i < input_tensor_size; i += 1 {
		input_tensor_values[i] = cast(c.float)i / (cast(c.float)input_tensor_size + 1)
	}

	// create input tensor object from data values
	memory_info: ^OrtMemoryInfo
	status = g_ort.CreateCpuMemoryInfo(
		OrtAllocatorType.OrtArenaAllocator,
		OrtMemType.OrtMemTypeDefault,
		&memory_info,
	)
	CheckStatus(g_ort, status)
	defer g_ort.ReleaseMemoryInfo(memory_info)
	input_tensor: ^OrtValue
	status = g_ort.CreateTensorWithDataAsOrtValue(
		memory_info,
		cast(rawptr)raw_data(input_tensor_values),
		input_tensor_size * size_of(c.float),
		cast(^c.int64_t)raw_data(input_node_dims),
		len(input_node_dims),
		ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
		&input_tensor,
	)
	CheckStatus(g_ort, status)
	defer g_ort.ReleaseValue(input_tensor)

	is_tensor: c.int
	status = g_ort.IsTensor(input_tensor, &is_tensor)
	CheckStatus(g_ort, status)
	assert(is_tensor == 1, "input_tensor not a tensor")

	// score model & input tensor, get back output tensor
	output_tensor: ^OrtValue
	run_options: ^OrtRunOptions
	status = g_ort.Run(
		session,
		run_options,
		raw_data(input_node_names),
		&input_tensor,
		len(input_node_names),
		raw_data(output_node_names),
		len(output_node_names),
		&output_tensor,
	)
	defer g_ort.ReleaseValue(output_tensor)
	CheckStatus(g_ort, status)

	status = g_ort.IsTensor(output_tensor, &is_tensor)
	CheckStatus(g_ort, status)
	assert(is_tensor == 1, "output_tensor not a tensor")

	// Get pointer to output tensor float values
	floatarr: [^]c.float
	status = g_ort.GetTensorMutableData(output_tensor, cast(^rawptr)&floatarr)
	CheckStatus(g_ort, status)
	assert(abs(floatarr[0]) - 0.000045 < 1e-6, "computition failed")

	for i := 0; i < 5; i += 1 {
		fmt.printfln(">>> Score for class [%d] =  %.6f", i, floatarr[i])
	}

	// Result should be
	// >>> Score for class [0] =  0.000045
	// >>> Score for class [1] =  0.003846
	// >>> Score for class [2] =  0.000125
	// >>> Score for class [3] =  0.001180
	// >>> Score for class [4] =  0.001317

	fmt.println("DONE!")
}
