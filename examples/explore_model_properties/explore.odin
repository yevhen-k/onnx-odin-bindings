package explore_model

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
		fmt.printfln(">>> Values: g_ort=%p", g_ort)
	}
	fmt.printfln(">>> API (str): %s", OrtGetApiBase().GetVersionString())

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

	////*************************************************************************
	//// create session and load model into memory
	//// using squeezenet version 1.3
	//// URL = https://github.com/onnx/models/tree/master/squeezenet
	session: ^OrtSession
	model_path :: "./squeezenet1.0-8.onnx"

	fmt.printfln(">>> Using Odin API (based on C API)")
	status = g_ort.CreateSession(env, model_path, session_options, &session)
	CheckStatus(g_ort, status)
	defer g_ort.ReleaseSession(session)

	////*************************************************************************
	//// print model input layer (node names, types, shape etc.)
	num_input_nodes: c.size_t
	allocator: ^OrtAllocator
	status = g_ort.GetAllocatorWithDefaultOptions(&allocator)
	CheckStatus(g_ort, status)

	//// print number of model input nodes
	status = g_ort.SessionGetInputCount(session, &num_input_nodes)
	CheckStatus(g_ort, status)

	input_node_names := make([dynamic]cstring, num_input_nodes)
	defer delete(input_node_names)

	input_node_dims := make([dynamic]c.int64_t)
	defer delete(input_node_dims)
	fmt.printfln(">>> Number of inputs = %d", num_input_nodes)

	// iterate over all input nodes
	for i: c.size_t = 0; i < num_input_nodes; i += 1 {
		// print input node names
		input_name: cstring
		status = g_ort.SessionGetInputName(session, i, allocator, &input_name)
		fmt.printfln(">>>\tInput %d : name=%s", i, input_name)
		input_node_names[i] = input_name

		// print input node types
		typeinfo: ^OrtTypeInfo
		status = g_ort.SessionGetInputTypeInfo(session, i, &typeinfo)
		CheckStatus(g_ort, status)
		defer g_ort.ReleaseTypeInfo(typeinfo)
		tensor_info: ^OrtTensorTypeAndShapeInfo
		status = g_ort.CastTypeInfoToTensorInfo(typeinfo, &tensor_info)
		CheckStatus(g_ort, status)
		type: ONNXTensorElementDataType
		status = g_ort.GetTensorElementType(tensor_info, &type)
		CheckStatus(g_ort, status)
		fmt.printfln(">>>\tInput %d : type=%d", i, type)

		// print input shapes/dims
		num_dims: c.size_t
		status = g_ort.GetDimensionsCount(tensor_info, &num_dims)
		CheckStatus(g_ort, status)
		fmt.printfln(">>>\tInput %d : num_dims=%d", i, num_dims)

		resize(&input_node_dims, cast(int)num_dims)
		status = g_ort.GetDimensions(
			tensor_info,
			cast(^c.int64_t)raw_data(input_node_dims),
			num_dims,
		)
		CheckStatus(g_ort, status)
		for j: c.size_t = 0; j < num_dims; j += 1 {
			fmt.printfln(">>>\tInput %d : dim %d=%d", i, j, input_node_dims[j])
		}
	}
	// Result should be
	// >>> Number of inputs = 1
	// >>>     Input 0 : name=data_0
	// >>>     Input 0 : type=1
	// >>>     Input 0 : num_dims=4
	// >>>     Input 0 : dim 0=1
	// >>>     Input 0 : dim 1=3
	// >>>     Input 0 : dim 2=224
	// >>>     Input 0 : dim 3=224

	//*************************************************************************

	num_output_nodes: c.size_t

	//// print number of model output nodes
	status = g_ort.SessionGetOutputCount(session, &num_output_nodes)
	CheckStatus(g_ort, status)

	output_node_names := make([dynamic]cstring, num_output_nodes)
	defer delete(output_node_names)

	output_node_dims := make([dynamic]c.int64_t)
	defer delete(output_node_dims)
	fmt.printfln(">>> Number of output = %d", num_output_nodes)

	// iterate over all output nodes
	for i: c.size_t = 0; i < num_output_nodes; i += 1 {
		// print output node names
		output_name: cstring
		status = g_ort.SessionGetOutputName(session, i, allocator, &output_name)
		fmt.printfln(">>>\tOutput %d : name=%s", i, output_name)
		output_node_names[i] = output_name

		// print output node types
		typeinfo: ^OrtTypeInfo
		status = g_ort.SessionGetOutputTypeInfo(session, i, &typeinfo)
		CheckStatus(g_ort, status)
		defer g_ort.ReleaseTypeInfo(typeinfo)
		tensor_info: ^OrtTensorTypeAndShapeInfo
		status = g_ort.CastTypeInfoToTensorInfo(typeinfo, &tensor_info)
		CheckStatus(g_ort, status)
		type: ONNXTensorElementDataType
		status = g_ort.GetTensorElementType(tensor_info, &type)
		CheckStatus(g_ort, status)
		fmt.printfln(">>>\tOutput %d : type=%d", i, type)

		// print output shapes/dims
		num_dims: c.size_t
		status = g_ort.GetDimensionsCount(tensor_info, &num_dims)
		CheckStatus(g_ort, status)
		fmt.printfln(">>>\tOutput %d : num_dims=%d", i, num_dims)

		resize(&output_node_dims, cast(int)num_dims)
		status = g_ort.GetDimensions(
			tensor_info,
			cast(^c.int64_t)raw_data(output_node_dims),
			num_dims,
		)
		CheckStatus(g_ort, status)
		for j: c.size_t = 0; j < num_dims; j += 1 {
			fmt.printfln(">>>\tOutput %d : dim %d=%d", i, j, output_node_dims[j])
		}
	}
	// Result should be
	// >>> Number of output = 1
	// >>>     Output 0 : name=softmaxout_1
	// >>>     Output 0 : type=1
	// >>>     Output 0 : num_dims=4
	// >>>     Output 0 : dim 0=1
	// >>>     Output 0 : dim 1=1000
	// >>>     Output 0 : dim 2=1
	// >>>     Output 0 : dim 3=1

}
