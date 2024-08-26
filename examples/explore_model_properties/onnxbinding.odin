package explore_model

import "core:c"

when ODIN_OS == .Linux do foreign import onnx "../../../thirdparty/onnxruntime/lib/libonnxruntime.so"

ORT_API_VERSION :: 17

// Enums:
ONNXTensorElementDataType :: enum c.int {
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_STRING,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_BOOL,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT32,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT64,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX64,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_COMPLEX128,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FN,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E4M3FNUZ,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2,
	ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT8E5M2FNUZ,
}


ONNXType :: enum c.int {
	ONNX_TYPE_UNKNOWN,
	ONNX_TYPE_TENSOR,
	ONNX_TYPE_SEQUENCE,
	ONNX_TYPE_MAP,
	ONNX_TYPE_OPAQUE,
	ONNX_TYPE_SPARSETENSOR,
	ONNX_TYPE_OPTIONAL,
}


OrtSparseFormat :: enum c.int {
	ORT_SPARSE_UNDEFINED    = 0,
	ORT_SPARSE_COO          = 0x1,
	ORT_SPARSE_CSRC         = 0x2,
	ORT_SPARSE_BLOCK_SPARSE = 0x4,
}


OrtSparseIndicesFormat :: enum c.int {
	ORT_SPARSE_COO_INDICES,
	ORT_SPARSE_CSR_INNER_INDICES,
	ORT_SPARSE_CSR_OUTER_INDICES,
	ORT_SPARSE_BLOCK_SPARSE_INDICES,
}


OrtLoggingLevel :: enum c.int {
	ORT_LOGGING_LEVEL_VERBOSE,
	ORT_LOGGING_LEVEL_INFO,
	ORT_LOGGING_LEVEL_WARNING,
	ORT_LOGGING_LEVEL_ERROR,
	ORT_LOGGING_LEVEL_FATAL,
}


OrtErrorCode :: enum c.int {
	ORT_OK,
	ORT_FAIL,
	ORT_INVALID_ARGUMENT,
	ORT_NO_SUCHFILE,
	ORT_NO_MODEL,
	ORT_ENGINE_ERROR,
	ORT_RUNTIME_EXCEPTION,
	ORT_INVALID_PROTOBUF,
	ORT_MODEL_LOADED,
	ORT_NOT_IMPLEMENTED,
	ORT_INVALID_GRAPH,
	ORT_EP_FAIL,
}


OrtOpAttrType :: enum c.int {
	ORT_OP_ATTR_UNDEFINED = 0,
	ORT_OP_ATTR_INT,
	ORT_OP_ATTR_INTS,
	ORT_OP_ATTR_FLOAT,
	ORT_OP_ATTR_FLOATS,
	ORT_OP_ATTR_STRING,
	ORT_OP_ATTR_STRINGS,
}


GraphOptimizationLevel :: enum c.int {
	ORT_DISABLE_ALL     = 0,
	ORT_ENABLE_BASIC    = 1,
	ORT_ENABLE_EXTENDED = 2,
	ORT_ENABLE_ALL      = 99,
}


ExecutionMode :: enum c.int {
	ORT_SEQUENTIAL = 0,
	ORT_PARALLEL   = 1,
}


OrtLanguageProjection :: enum c.int {
	ORT_PROJECTION_C         = 0,
	ORT_PROJECTION_CPLUSPLUS = 1,
	ORT_PROJECTION_CSHARP    = 2,
	ORT_PROJECTION_PYTHON    = 3,
	ORT_PROJECTION_JAVA      = 4,
	ORT_PROJECTION_WINML     = 5,
	ORT_PROJECTION_NODEJS    = 6,
}


OrtAllocatorType :: enum c.int {
	OrtInvalidAllocator = -1,
	OrtDeviceAllocator  = 0,
	OrtArenaAllocator   = 1,
}


OrtMemType :: enum c.int {
	OrtMemTypeCPUInput  = -2,
	OrtMemTypeCPUOutput = -1,
	OrtMemTypeDefault   = 0,
}


OrtMemoryInfoDeviceType :: enum c.int {
	OrtMemoryInfoDeviceType_CPU  = 0,
	OrtMemoryInfoDeviceType_GPU  = 1,
	OrtMemoryInfoDeviceType_FPGA = 2,
}


OrtCudnnConvAlgoSearch :: enum c.int {
	OrtCudnnConvAlgoSearchExhaustive,
	OrtCudnnConvAlgoSearchHeuristic,
	OrtCudnnConvAlgoSearchDefault,
}


OrtCustomOpInputOutputCharacteristic :: enum c.int {
	INPUT_OUTPUT_REQUIRED = 0,
	INPUT_OUTPUT_OPTIONAL,
	INPUT_OUTPUT_VARIADIC,
}


// End Enums

// Function pointer declarations:
OrtLoggingFunction :: #type proc "c" (
	param: rawptr,
	severity: OrtLoggingLevel,
	category: cstring,
	logid: cstring,
	code_location: cstring,
	message: cstring,
)

OrtThreadWorkerFn :: #type proc "c" (ort_worker_fn_param: rawptr)

OrtCustomCreateThreadFn :: #type proc "c" (
	ort_custom_thread_creation_options: rawptr,
	ort_thread_worker_fn: OrtThreadWorkerFn,
	ort_worker_fn_param: rawptr,
) -> OrtCustomThreadHandle

OrtCustomJoinThreadFn :: #type proc "c" (ort_custom_thread_handle: OrtCustomThreadHandle)

RegisterCustomOpsFn :: #type proc "c" (options: ^OrtSessionOptions, api: ^OrtApiBase) -> ^OrtStatus

RunAsyncCallbackFn :: #type proc "c" (
	user_data: rawptr,
	outputs: ^^OrtValue,
	num_outputs: c.size_t,
	status: OrtStatusPtr,
)

// End Function pointer declarations

// Function pointer definitions:
@(default_calling_convention = "c")
foreign onnx {
	OrtGetApiBase :: proc() -> ^OrtApiBase ---
	OrtSessionOptionsAppendExecutionProvider_CUDA :: proc(options: ^OrtSessionOptions, device_id: c.int) -> OrtStatusPtr ---
	OrtSessionOptionsAppendExecutionProvider_ROCM :: proc(options: ^OrtSessionOptions, device_id: c.int) -> OrtStatusPtr ---
	OrtSessionOptionsAppendExecutionProvider_MIGraphX :: proc(options: ^OrtSessionOptions, device_id: c.int) -> OrtStatusPtr ---
	OrtSessionOptionsAppendExecutionProvider_Dnnl :: proc(options: ^OrtSessionOptions, use_arena: c.int) -> OrtStatusPtr ---
	OrtSessionOptionsAppendExecutionProvider_Tensorrt :: proc(options: ^OrtSessionOptions, device_id: c.int) -> OrtStatusPtr ---

}

// End Function pointer definitions

// Type definition of struct pointers:
OrtStatusPtr :: ^OrtStatus

OrtCustomHandleType :: struct {
	__place_holder: c.char,
}

OrtCustomThreadHandle :: ^OrtCustomHandleType

// End Type definition of struct pointers

// Declared structs:
OrtEnv :: struct {}
OrtStatus :: struct {}
OrtMemoryInfo :: struct {}
OrtIoBinding :: struct {}
OrtSession :: struct {}
OrtValue :: struct {}
OrtRunOptions :: struct {}
OrtTypeInfo :: struct {}
OrtTensorTypeAndShapeInfo :: struct {}
OrtMapTypeInfo :: struct {}
OrtSequenceTypeInfo :: struct {}
OrtOptionalTypeInfo :: struct {}
OrtSessionOptions :: struct {}
OrtCustomOpDomain :: struct {}
OrtModelMetadata :: struct {}
OrtThreadPoolParams :: struct {}
OrtThreadingOptions :: struct {}
OrtArenaCfg :: struct {}
OrtPrepackedWeightsContainer :: struct {}
OrtTensorRTProviderOptionsV2 :: struct {}
OrtCUDAProviderOptionsV2 :: struct {}
OrtCANNProviderOptions :: struct {}
OrtDnnlProviderOptions :: struct {}
OrtOp :: struct {}
OrtOpAttr :: struct {}
OrtLogger :: struct {}
OrtShapeInferContext :: struct {}
OrtKernelInfo :: struct {}
OrtKernelContext :: struct {}
OrtTrainingApi :: struct {}

// End Declared structs

// Declared typedef structs:

// End Declared typedef structs

OrtAllocator :: struct {
	version: c.uint32_t,
	Alloc:   proc(this_: ^OrtAllocator, size: c.size_t) -> rawptr,
	Free:    proc(this_: ^OrtAllocator, p: rawptr),
	Info:    proc(this_: ^OrtAllocator) -> ^OrtMemoryInfo,
}

OrtCUDAProviderOptions :: struct {
	device_id:                         c.int,
	cudnn_conv_algo_search:            OrtCudnnConvAlgoSearch,
	gpu_mem_limit:                     c.size_t,
	arena_extend_strategy:             c.int,
	do_copy_in_default_stream:         c.int,
	has_user_compute_stream:           c.int,
	user_compute_stream:               rawptr,
	default_memory_arena_cfg:          ^OrtArenaCfg,
	tunable_op_enable:                 c.int,
	tunable_op_tuning_enable:          c.int,
	tunable_op_max_tuning_duration_ms: c.int,
}

OrtROCMProviderOptions :: struct {
	device_id:                         c.int,
	miopen_conv_exhaustive_search:     c.int,
	gpu_mem_limit:                     c.size_t,
	arena_extend_strategy:             c.int,
	do_copy_in_default_stream:         c.int,
	has_user_compute_stream:           c.int,
	user_compute_stream:               rawptr,
	default_memory_arena_cfg:          ^OrtArenaCfg,
	tunable_op_enable:                 c.int,
	tunable_op_tuning_enable:          c.int,
	tunable_op_max_tuning_duration_ms: c.int,
}

OrtTensorRTProviderOptions :: struct {
	device_id:                             c.int,
	has_user_compute_stream:               c.int,
	user_compute_stream:                   rawptr,
	trt_max_partition_iterations:          c.int,
	trt_min_subgraph_size:                 c.int,
	trt_max_workspace_size:                c.size_t,
	trt_fp16_enable:                       c.int,
	trt_int8_enable:                       c.int,
	trt_int8_calibration_table_name:       cstring,
	trt_int8_use_native_calibration_table: c.int,
	trt_dla_enable:                        c.int,
	trt_dla_core:                          c.int,
	trt_dump_subgraphs:                    c.int,
	trt_engine_cache_enable:               c.int,
	trt_engine_cache_path:                 cstring,
	trt_engine_decryption_enable:          c.int,
	trt_engine_decryption_lib_path:        cstring,
	trt_force_sequential_engine_build:     c.int,
}

OrtMIGraphXProviderOptions :: struct {
	device_id:                             c.int,
	migraphx_fp16_enable:                  c.int,
	migraphx_int8_enable:                  c.int,
	migraphx_use_native_calibration_table: c.int,
	migraphx_int8_calibration_table_name:  cstring,
}

OrtOpenVINOProviderOptions :: struct {
	device_type:              cstring,
	enable_npu_fast_compile:  c.char,
	device_id:                cstring,
	num_of_threads:           c.size_t,
	cache_dir:                cstring,
	context_:                 rawptr,
	enable_opencl_throttling: c.char,
	enable_dynamic_shapes:    c.char,
}

OrtApiBase :: struct {
	GetApi:           proc(version: c.uint32_t) -> ^OrtApi,
	GetVersionString: proc() -> cstring,
}

OrtApi :: struct {
	CreateStatus:                                        proc(
		code: OrtErrorCode,
		msg: cstring,
	) -> ^OrtStatus,
	GetErrorCode:                                        proc(status: ^OrtStatus) -> OrtErrorCode,
	GetErrorMessage:                                     proc(status: ^OrtStatus) -> cstring,
	CreateEnv:                                           proc(
		log_severity_level: OrtLoggingLevel,
		logid: cstring,
		out: ^^OrtEnv,
	) -> OrtStatusPtr,
	CreateEnvWithCustomLogger:                           proc(
		logging_function: OrtLoggingFunction,
		logger_param: rawptr,
		log_severity_level: OrtLoggingLevel,
		logid: cstring,
		out: ^^OrtEnv,
	) -> OrtStatusPtr,
	EnableTelemetryEvents:                               proc(env: ^OrtEnv) -> OrtStatusPtr,
	DisableTelemetryEvents:                              proc(env: ^OrtEnv) -> OrtStatusPtr,
	CreateSession:                                       proc(
		env: ^OrtEnv,
		model_path: cstring,
		options: ^OrtSessionOptions,
		out: ^^OrtSession,
	) -> OrtStatusPtr,
	CreateSessionFromArray:                              proc(
		env: ^OrtEnv,
		model_data: rawptr,
		model_data_length: c.size_t,
		options: ^OrtSessionOptions,
		out: ^^OrtSession,
	) -> OrtStatusPtr,
	Run:                                                 proc(
		session: ^OrtSession,
		run_options: ^OrtRunOptions,
		input_names: [^]cstring,
		inputs: ^^OrtValue,
		input_len: c.size_t,
		output_names: [^]cstring,
		output_names_len: c.size_t,
		outputs: ^^OrtValue,
	) -> OrtStatusPtr,
	CreateSessionOptions:                                proc(
		options: ^^OrtSessionOptions,
	) -> OrtStatusPtr,
	SetOptimizedModelFilePath:                           proc(
		options: ^OrtSessionOptions,
		optimized_model_filepath: cstring,
	) -> OrtStatusPtr,
	CloneSessionOptions:                                 proc(
		in_options: ^OrtSessionOptions,
		out_options: ^^OrtSessionOptions,
	) -> OrtStatusPtr,
	SetSessionExecutionMode:                             proc(
		options: ^OrtSessionOptions,
		execution_mode: ExecutionMode,
	) -> OrtStatusPtr,
	EnableProfiling:                                     proc(
		options: ^OrtSessionOptions,
		profile_file_prefix: cstring,
	) -> OrtStatusPtr,
	DisableProfiling:                                    proc(
		options: ^OrtSessionOptions,
	) -> OrtStatusPtr,
	EnableMemPattern:                                    proc(
		options: ^OrtSessionOptions,
	) -> OrtStatusPtr,
	DisableMemPattern:                                   proc(
		options: ^OrtSessionOptions,
	) -> OrtStatusPtr,
	EnableCpuMemArena:                                   proc(
		options: ^OrtSessionOptions,
	) -> OrtStatusPtr,
	DisableCpuMemArena:                                  proc(
		options: ^OrtSessionOptions,
	) -> OrtStatusPtr,
	SetSessionLogId:                                     proc(
		options: ^OrtSessionOptions,
		logid: cstring,
	) -> OrtStatusPtr,
	SetSessionLogVerbosityLevel:                         proc(
		options: ^OrtSessionOptions,
		session_log_verbosity_level: c.int,
	) -> OrtStatusPtr,
	SetSessionLogSeverityLevel:                          proc(
		options: ^OrtSessionOptions,
		session_log_severity_level: c.int,
	) -> OrtStatusPtr,
	SetSessionGraphOptimizationLevel:                    proc(
		options: ^OrtSessionOptions,
		graph_optimization_level: GraphOptimizationLevel,
	) -> OrtStatusPtr,
	SetIntraOpNumThreads:                                proc(
		options: ^OrtSessionOptions,
		intra_op_num_threads: c.int,
	) -> OrtStatusPtr,
	SetInterOpNumThreads:                                proc(
		options: ^OrtSessionOptions,
		inter_op_num_threads: c.int,
	) -> OrtStatusPtr,
	CreateCustomOpDomain:                                proc(
		domain: cstring,
		out: ^^OrtCustomOpDomain,
	) -> OrtStatusPtr,
	CustomOpDomain_Add:                                  proc(
		custom_op_domain: ^OrtCustomOpDomain,
		op: ^OrtCustomOp,
	) -> OrtStatusPtr,
	AddCustomOpDomain:                                   proc(
		options: ^OrtSessionOptions,
		custom_op_domain: ^OrtCustomOpDomain,
	) -> OrtStatusPtr,
	RegisterCustomOpsLibrary:                            proc(
		options: ^OrtSessionOptions,
		library_path: cstring,
		library_handle: ^rawptr,
	) -> OrtStatusPtr,
	SessionGetInputCount:                                proc(
		session: ^OrtSession,
		out: ^c.size_t,
	) -> OrtStatusPtr,
	SessionGetOutputCount:                               proc(
		session: ^OrtSession,
		out: ^c.size_t,
	) -> OrtStatusPtr,
	SessionGetOverridableInitializerCount:               proc(
		session: ^OrtSession,
		out: ^c.size_t,
	) -> OrtStatusPtr,
	SessionGetInputTypeInfo:                             proc(
		session: ^OrtSession,
		index: c.size_t,
		type_info: ^^OrtTypeInfo,
	) -> OrtStatusPtr,
	SessionGetOutputTypeInfo:                            proc(
		session: ^OrtSession,
		index: c.size_t,
		type_info: ^^OrtTypeInfo,
	) -> OrtStatusPtr,
	SessionGetOverridableInitializerTypeInfo:            proc(
		session: ^OrtSession,
		index: c.size_t,
		type_info: ^^OrtTypeInfo,
	) -> OrtStatusPtr,
	SessionGetInputName:                                 proc(
		session: ^OrtSession,
		index: c.size_t,
		allocator: ^OrtAllocator,
		value: [^]cstring,
	) -> OrtStatusPtr,
	SessionGetOutputName:                                proc(
		session: ^OrtSession,
		index: c.size_t,
		allocator: ^OrtAllocator,
		value: [^]cstring,
	) -> OrtStatusPtr,
	SessionGetOverridableInitializerName:                proc(
		session: ^OrtSession,
		index: c.size_t,
		allocator: ^OrtAllocator,
		value: [^]cstring,
	) -> OrtStatusPtr,
	CreateRunOptions:                                    proc(
		out: ^^OrtRunOptions,
	) -> OrtStatusPtr,
	RunOptionsSetRunLogVerbosityLevel:                   proc(
		options: ^OrtRunOptions,
		log_verbosity_level: c.int,
	) -> OrtStatusPtr,
	RunOptionsSetRunLogSeverityLevel:                    proc(
		options: ^OrtRunOptions,
		log_severity_level: c.int,
	) -> OrtStatusPtr,
	RunOptionsSetRunTag:                                 proc(
		options: ^OrtRunOptions,
		run_tag: cstring,
	) -> OrtStatusPtr,
	RunOptionsGetRunLogVerbosityLevel:                   proc(
		options: ^OrtRunOptions,
		log_verbosity_level: ^c.int,
	) -> OrtStatusPtr,
	RunOptionsGetRunLogSeverityLevel:                    proc(
		options: ^OrtRunOptions,
		log_severity_level: ^c.int,
	) -> OrtStatusPtr,
	RunOptionsGetRunTag:                                 proc(
		options: ^OrtRunOptions,
		run_tag: [^]cstring,
	) -> OrtStatusPtr,
	RunOptionsSetTerminate:                              proc(
		options: ^OrtRunOptions,
	) -> OrtStatusPtr,
	RunOptionsUnsetTerminate:                            proc(
		options: ^OrtRunOptions,
	) -> OrtStatusPtr,
	CreateTensorAsOrtValue:                              proc(
		allocator: ^OrtAllocator,
		shape: ^c.int64_t,
		shape_len: c.size_t,
		type: ONNXTensorElementDataType,
		out: ^^OrtValue,
	) -> OrtStatusPtr,
	CreateTensorWithDataAsOrtValue:                      proc(
		info: ^OrtMemoryInfo,
		p_data: rawptr,
		p_data_len: c.size_t,
		shape: ^c.int64_t,
		shape_len: c.size_t,
		type: ONNXTensorElementDataType,
		out: ^^OrtValue,
	) -> OrtStatusPtr,
	IsTensor:                                            proc(
		value: ^OrtValue,
		out: ^c.int,
	) -> OrtStatusPtr,
	GetTensorMutableData:                                proc(
		value: ^OrtValue,
		out: ^rawptr,
	) -> OrtStatusPtr,
	FillStringTensor:                                    proc(
		value: ^OrtValue,
		s: [^]cstring,
		s_len: c.size_t,
	) -> OrtStatusPtr,
	GetStringTensorDataLength:                           proc(
		value: ^OrtValue,
		len: ^c.size_t,
	) -> OrtStatusPtr,
	GetStringTensorContent:                              proc(
		value: ^OrtValue,
		s: rawptr,
		s_len: c.size_t,
		offsets: ^c.size_t,
		offsets_len: c.size_t,
	) -> OrtStatusPtr,
	CastTypeInfoToTensorInfo:                            proc(
		type_info: ^OrtTypeInfo,
		out: ^^OrtTensorTypeAndShapeInfo,
	) -> OrtStatusPtr,
	GetOnnxTypeFromTypeInfo:                             proc(
		type_info: ^OrtTypeInfo,
		out: ^ONNXType,
	) -> OrtStatusPtr,
	CreateTensorTypeAndShapeInfo:                        proc(
		out: ^^OrtTensorTypeAndShapeInfo,
	) -> OrtStatusPtr,
	SetTensorElementType:                                proc(
		info: ^OrtTensorTypeAndShapeInfo,
		type: ONNXTensorElementDataType,
	) -> OrtStatusPtr,
	SetDimensions:                                       proc(
		info: ^OrtTensorTypeAndShapeInfo,
		dim_values: ^c.int64_t,
		dim_count: c.size_t,
	) -> OrtStatusPtr,
	GetTensorElementType:                                proc(
		info: ^OrtTensorTypeAndShapeInfo,
		out: ^ONNXTensorElementDataType,
	) -> OrtStatusPtr,
	GetDimensionsCount:                                  proc(
		info: ^OrtTensorTypeAndShapeInfo,
		out: ^c.size_t,
	) -> OrtStatusPtr,
	GetDimensions:                                       proc(
		info: ^OrtTensorTypeAndShapeInfo,
		dim_values: ^c.int64_t,
		dim_values_length: c.size_t,
	) -> OrtStatusPtr,
	GetSymbolicDimensions:                               proc(
		info: ^OrtTensorTypeAndShapeInfo,
		dim_params: [^]cstring,
		dim_params_length: c.size_t,
	) -> OrtStatusPtr,
	GetTensorShapeElementCount:                          proc(
		info: ^OrtTensorTypeAndShapeInfo,
		out: ^c.size_t,
	) -> OrtStatusPtr,
	GetTensorTypeAndShape:                               proc(
		value: ^OrtValue,
		out: ^^OrtTensorTypeAndShapeInfo,
	) -> OrtStatusPtr,
	GetTypeInfo:                                         proc(
		value: ^OrtValue,
		out: ^^OrtTypeInfo,
	) -> OrtStatusPtr,
	GetValueType:                                        proc(
		value: ^OrtValue,
		out: ^ONNXType,
	) -> OrtStatusPtr,
	CreateMemoryInfo:                                    proc(
		name: cstring,
		type: OrtAllocatorType,
		id: c.int,
		mem_type: OrtMemType,
		out: ^^OrtMemoryInfo,
	) -> OrtStatusPtr,
	CreateCpuMemoryInfo:                                 proc(
		type: OrtAllocatorType,
		mem_type: OrtMemType,
		out: ^^OrtMemoryInfo,
	) -> OrtStatusPtr,
	CompareMemoryInfo:                                   proc(
		info1: ^OrtMemoryInfo,
		info2: ^OrtMemoryInfo,
		out: ^c.int,
	) -> OrtStatusPtr,
	MemoryInfoGetName:                                   proc(
		ptr: ^OrtMemoryInfo,
		out: [^]cstring,
	) -> OrtStatusPtr,
	MemoryInfoGetId:                                     proc(
		ptr: ^OrtMemoryInfo,
		out: ^c.int,
	) -> OrtStatusPtr,
	MemoryInfoGetMemType:                                proc(
		ptr: ^OrtMemoryInfo,
		out: ^OrtMemType,
	) -> OrtStatusPtr,
	MemoryInfoGetType:                                   proc(
		ptr: ^OrtMemoryInfo,
		out: ^OrtAllocatorType,
	) -> OrtStatusPtr,
	AllocatorAlloc:                                      proc(
		ort_allocator: ^OrtAllocator,
		size: c.size_t,
		out: ^rawptr,
	) -> OrtStatusPtr,
	AllocatorFree:                                       proc(
		ort_allocator: ^OrtAllocator,
		p: rawptr,
	) -> OrtStatusPtr,
	AllocatorGetInfo:                                    proc(
		ort_allocator: ^OrtAllocator,
		out: ^^OrtMemoryInfo,
	) -> OrtStatusPtr,
	GetAllocatorWithDefaultOptions:                      proc(out: ^^OrtAllocator) -> OrtStatusPtr,
	AddFreeDimensionOverride:                            proc(
		options: ^OrtSessionOptions,
		dim_denotation: cstring,
		dim_value: c.int64_t,
	) -> OrtStatusPtr,
	GetValue:                                            proc(
		value: ^OrtValue,
		index: c.int,
		allocator: ^OrtAllocator,
		out: ^^OrtValue,
	) -> OrtStatusPtr,
	GetValueCount:                                       proc(
		value: ^OrtValue,
		out: ^c.size_t,
	) -> OrtStatusPtr,
	CreateValue:                                         proc(
		in_: ^^OrtValue,
		num_values: c.size_t,
		value_type: ONNXType,
		out: ^^OrtValue,
	) -> OrtStatusPtr,
	CreateOpaqueValue:                                   proc(
		domain_name: cstring,
		type_name: cstring,
		data_container: rawptr,
		data_container_size: c.size_t,
		out: ^^OrtValue,
	) -> OrtStatusPtr,
	GetOpaqueValue:                                      proc(
		domain_name: cstring,
		type_name: cstring,
		in_: ^OrtValue,
		data_container: rawptr,
		data_container_size: c.size_t,
	) -> OrtStatusPtr,
	KernelInfoGetAttribute_float:                        proc(
		info: ^OrtKernelInfo,
		name: cstring,
		out: ^c.float,
	) -> OrtStatusPtr,
	KernelInfoGetAttribute_int64:                        proc(
		info: ^OrtKernelInfo,
		name: cstring,
		out: ^c.int64_t,
	) -> OrtStatusPtr,
	KernelInfoGetAttribute_string:                       proc(
		info: ^OrtKernelInfo,
		name: cstring,
		out: cstring,
		size: ^c.size_t,
	) -> OrtStatusPtr,
	KernelContext_GetInputCount:                         proc(
		context_: ^OrtKernelContext,
		out: ^c.size_t,
	) -> OrtStatusPtr,
	KernelContext_GetOutputCount:                        proc(
		context_: ^OrtKernelContext,
		out: ^c.size_t,
	) -> OrtStatusPtr,
	KernelContext_GetInput:                              proc(
		context_: ^OrtKernelContext,
		index: c.size_t,
		out: ^^OrtValue,
	) -> OrtStatusPtr,
	KernelContext_GetOutput:                             proc(
		context_: ^OrtKernelContext,
		index: c.size_t,
		dim_values: ^c.int64_t,
		dim_count: c.size_t,
		out: ^^OrtValue,
	) -> OrtStatusPtr,
	ReleaseEnv:                                          proc(input: ^OrtEnv),
	ReleaseStatus:                                       proc(input: ^OrtStatus),
	ReleaseMemoryInfo:                                   proc(input: ^OrtMemoryInfo),
	ReleaseSession:                                      proc(input: ^OrtSession),
	ReleaseValue:                                        proc(input: ^OrtValue),
	ReleaseRunOptions:                                   proc(input: ^OrtRunOptions),
	ReleaseTypeInfo:                                     proc(input: ^OrtTypeInfo),
	ReleaseTensorTypeAndShapeInfo:                       proc(input: ^OrtTensorTypeAndShapeInfo),
	ReleaseSessionOptions:                               proc(input: ^OrtSessionOptions),
	ReleaseCustomOpDomain:                               proc(input: ^OrtCustomOpDomain),
	GetDenotationFromTypeInfo:                           proc(
		type_info: ^OrtTypeInfo,
		denotation: [^]cstring,
		len: ^c.size_t,
	) -> OrtStatusPtr,
	CastTypeInfoToMapTypeInfo:                           proc(
		type_info: ^OrtTypeInfo,
		out: ^^OrtMapTypeInfo,
	) -> OrtStatusPtr,
	CastTypeInfoToSequenceTypeInfo:                      proc(
		type_info: ^OrtTypeInfo,
		out: ^^OrtSequenceTypeInfo,
	) -> OrtStatusPtr,
	GetMapKeyType:                                       proc(
		map_type_info: ^OrtMapTypeInfo,
		out: ^ONNXTensorElementDataType,
	) -> OrtStatusPtr,
	GetMapValueType:                                     proc(
		map_type_info: ^OrtMapTypeInfo,
		type_info: ^^OrtTypeInfo,
	) -> OrtStatusPtr,
	GetSequenceElementType:                              proc(
		sequence_type_info: ^OrtSequenceTypeInfo,
		type_info: ^^OrtTypeInfo,
	) -> OrtStatusPtr,
	ReleaseMapTypeInfo:                                  proc(input: ^OrtMapTypeInfo),
	ReleaseSequenceTypeInfo:                             proc(input: ^OrtSequenceTypeInfo),
	SessionEndProfiling:                                 proc(
		session: ^OrtSession,
		allocator: ^OrtAllocator,
		out: [^]cstring,
	) -> OrtStatusPtr,
	SessionGetModelMetadata:                             proc(
		session: ^OrtSession,
		out: ^^OrtModelMetadata,
	) -> OrtStatusPtr,
	ModelMetadataGetProducerName:                        proc(
		model_metadata: ^OrtModelMetadata,
		allocator: ^OrtAllocator,
		value: [^]cstring,
	) -> OrtStatusPtr,
	ModelMetadataGetGraphName:                           proc(
		model_metadata: ^OrtModelMetadata,
		allocator: ^OrtAllocator,
		value: [^]cstring,
	) -> OrtStatusPtr,
	ModelMetadataGetDomain:                              proc(
		model_metadata: ^OrtModelMetadata,
		allocator: ^OrtAllocator,
		value: [^]cstring,
	) -> OrtStatusPtr,
	ModelMetadataGetDescription:                         proc(
		model_metadata: ^OrtModelMetadata,
		allocator: ^OrtAllocator,
		value: [^]cstring,
	) -> OrtStatusPtr,
	ModelMetadataLookupCustomMetadataMap:                proc(
		model_metadata: ^OrtModelMetadata,
		allocator: ^OrtAllocator,
		key: cstring,
		value: [^]cstring,
	) -> OrtStatusPtr,
	ModelMetadataGetVersion:                             proc(
		model_metadata: ^OrtModelMetadata,
		value: ^c.int64_t,
	) -> OrtStatusPtr,
	ReleaseModelMetadata:                                proc(input: ^OrtModelMetadata),
	CreateEnvWithGlobalThreadPools:                      proc(
		log_severity_level: OrtLoggingLevel,
		logid: cstring,
		tp_options: ^OrtThreadingOptions,
		out: ^^OrtEnv,
	) -> OrtStatusPtr,
	DisablePerSessionThreads:                            proc(
		options: ^OrtSessionOptions,
	) -> OrtStatusPtr,
	CreateThreadingOptions:                              proc(
		out: ^^OrtThreadingOptions,
	) -> OrtStatusPtr,
	ReleaseThreadingOptions:                             proc(input: ^OrtThreadingOptions),
	ModelMetadataGetCustomMetadataMapKeys:               proc(
		model_metadata: ^OrtModelMetadata,
		allocator: ^OrtAllocator,
		keys: ^^^c.char,
		num_keys: ^c.int64_t,
	) -> OrtStatusPtr,
	AddFreeDimensionOverrideByName:                      proc(
		options: ^OrtSessionOptions,
		dim_name: cstring,
		dim_value: c.int64_t,
	) -> OrtStatusPtr,
	GetAvailableProviders:                               proc(
		out_ptr: ^^^c.char,
		provider_length: ^c.int,
	) -> OrtStatusPtr,
	ReleaseAvailableProviders:                           proc(
		ptr: [^]cstring,
		providers_length: c.int,
	) -> OrtStatusPtr,
	GetStringTensorElementLength:                        proc(
		value: ^OrtValue,
		index: c.size_t,
		out: ^c.size_t,
	) -> OrtStatusPtr,
	GetStringTensorElement:                              proc(
		value: ^OrtValue,
		s_len: c.size_t,
		index: c.size_t,
		s: rawptr,
	) -> OrtStatusPtr,
	FillStringTensorElement:                             proc(
		value: ^OrtValue,
		s: cstring,
		index: c.size_t,
	) -> OrtStatusPtr,
	AddSessionConfigEntry:                               proc(
		options: ^OrtSessionOptions,
		config_key: cstring,
		config_value: cstring,
	) -> OrtStatusPtr,
	CreateAllocator:                                     proc(
		session: ^OrtSession,
		mem_info: ^OrtMemoryInfo,
		out: ^^OrtAllocator,
	) -> OrtStatusPtr,
	ReleaseAllocator:                                    proc(input: ^OrtAllocator),
	RunWithBinding:                                      proc(
		session: ^OrtSession,
		run_options: ^OrtRunOptions,
		binding_ptr: ^OrtIoBinding,
	) -> OrtStatusPtr,
	CreateIoBinding:                                     proc(
		session: ^OrtSession,
		out: ^^OrtIoBinding,
	) -> OrtStatusPtr,
	ReleaseIoBinding:                                    proc(input: ^OrtIoBinding),
	BindInput:                                           proc(
		binding_ptr: ^OrtIoBinding,
		name: cstring,
		val_ptr: ^OrtValue,
	) -> OrtStatusPtr,
	BindOutput:                                          proc(
		binding_ptr: ^OrtIoBinding,
		name: cstring,
		val_ptr: ^OrtValue,
	) -> OrtStatusPtr,
	BindOutputToDevice:                                  proc(
		binding_ptr: ^OrtIoBinding,
		name: cstring,
		mem_info_ptr: ^OrtMemoryInfo,
	) -> OrtStatusPtr,
	GetBoundOutputNames:                                 proc(
		binding_ptr: ^OrtIoBinding,
		allocator: ^OrtAllocator,
		buffer: [^]cstring,
		lengths: ^^c.size_t,
		count: ^c.size_t,
	) -> OrtStatusPtr,
	GetBoundOutputValues:                                proc(
		binding_ptr: ^OrtIoBinding,
		allocator: ^OrtAllocator,
		output: ^^^OrtValue,
		output_count: ^c.size_t,
	) -> OrtStatusPtr,
	ClearBoundInputs:                                    proc(binding_ptr: ^OrtIoBinding),
	ClearBoundOutputs:                                   proc(binding_ptr: ^OrtIoBinding),
	TensorAt:                                            proc(
		value: ^OrtValue,
		location_values: ^c.int64_t,
		location_values_count: c.size_t,
		out: ^rawptr,
	) -> OrtStatusPtr,
	CreateAndRegisterAllocator:                          proc(
		env: ^OrtEnv,
		mem_info: ^OrtMemoryInfo,
		arena_cfg: ^OrtArenaCfg,
	) -> OrtStatusPtr,
	SetLanguageProjection:                               proc(
		ort_env: ^OrtEnv,
		projection: OrtLanguageProjection,
	) -> OrtStatusPtr,
	SessionGetProfilingStartTimeNs:                      proc(
		session: ^OrtSession,
		out: ^c.uint64_t,
	) -> OrtStatusPtr,
	SetGlobalIntraOpNumThreads:                          proc(
		tp_options: ^OrtThreadingOptions,
		intra_op_num_threads: c.int,
	) -> OrtStatusPtr,
	SetGlobalInterOpNumThreads:                          proc(
		tp_options: ^OrtThreadingOptions,
		inter_op_num_threads: c.int,
	) -> OrtStatusPtr,
	SetGlobalSpinControl:                                proc(
		tp_options: ^OrtThreadingOptions,
		allow_spinning: c.int,
	) -> OrtStatusPtr,
	AddInitializer:                                      proc(
		options: ^OrtSessionOptions,
		name: cstring,
		val: ^OrtValue,
	) -> OrtStatusPtr,
	CreateEnvWithCustomLoggerAndGlobalThreadPools:       proc(
		logging_function: OrtLoggingFunction,
		logger_param: rawptr,
		log_severity_level: OrtLoggingLevel,
		logid: cstring,
		tp_options: ^OrtThreadingOptions,
		out: ^^OrtEnv,
	) -> OrtStatusPtr,
	SessionOptionsAppendExecutionProvider_CUDA:          proc(
		options: ^OrtSessionOptions,
		cuda_options: ^OrtCUDAProviderOptions,
	) -> OrtStatusPtr,
	SessionOptionsAppendExecutionProvider_ROCM:          proc(
		options: ^OrtSessionOptions,
		rocm_options: ^OrtROCMProviderOptions,
	) -> OrtStatusPtr,
	SessionOptionsAppendExecutionProvider_OpenVINO:      proc(
		options: ^OrtSessionOptions,
		provider_options: ^OrtOpenVINOProviderOptions,
	) -> OrtStatusPtr,
	SetGlobalDenormalAsZero:                             proc(
		tp_options: ^OrtThreadingOptions,
	) -> OrtStatusPtr,
	CreateArenaCfg:                                      proc(
		max_mem: c.size_t,
		arena_extend_strategy: c.int,
		initial_chunk_size_bytes: c.int,
		max_dead_bytes_per_chunk: c.int,
		out: ^^OrtArenaCfg,
	) -> OrtStatusPtr,
	ReleaseArenaCfg:                                     proc(input: ^OrtArenaCfg),
	ModelMetadataGetGraphDescription:                    proc(
		model_metadata: ^OrtModelMetadata,
		allocator: ^OrtAllocator,
		value: [^]cstring,
	) -> OrtStatusPtr,
	SessionOptionsAppendExecutionProvider_TensorRT:      proc(
		options: ^OrtSessionOptions,
		tensorrt_options: ^OrtTensorRTProviderOptions,
	) -> OrtStatusPtr,
	SetCurrentGpuDeviceId:                               proc(device_id: c.int) -> OrtStatusPtr,
	GetCurrentGpuDeviceId:                               proc(device_id: ^c.int) -> OrtStatusPtr,
	KernelInfoGetAttributeArray_float:                   proc(
		info: ^OrtKernelInfo,
		name: cstring,
		out: ^c.float,
		size: ^c.size_t,
	) -> OrtStatusPtr,
	KernelInfoGetAttributeArray_int64:                   proc(
		info: ^OrtKernelInfo,
		name: cstring,
		out: ^c.int64_t,
		size: ^c.size_t,
	) -> OrtStatusPtr,
	CreateArenaCfgV2:                                    proc(
		arena_config_keys: [^]cstring,
		arena_config_values: ^c.size_t,
		num_keys: c.size_t,
		out: ^^OrtArenaCfg,
	) -> OrtStatusPtr,
	AddRunConfigEntry:                                   proc(
		options: ^OrtRunOptions,
		config_key: cstring,
		config_value: cstring,
	) -> OrtStatusPtr,
	CreatePrepackedWeightsContainer:                     proc(
		out: ^^OrtPrepackedWeightsContainer,
	) -> OrtStatusPtr,
	ReleasePrepackedWeightsContainer:                    proc(
		input: ^OrtPrepackedWeightsContainer,
	),
	CreateSessionWithPrepackedWeightsContainer:          proc(
		env: ^OrtEnv,
		model_path: cstring,
		options: ^OrtSessionOptions,
		prepacked_weights_container: ^OrtPrepackedWeightsContainer,
		out: ^^OrtSession,
	) -> OrtStatusPtr,
	CreateSessionFromArrayWithPrepackedWeightsContainer: proc(
		env: ^OrtEnv,
		model_data: rawptr,
		model_data_length: c.size_t,
		options: ^OrtSessionOptions,
		prepacked_weights_container: ^OrtPrepackedWeightsContainer,
		out: ^^OrtSession,
	) -> OrtStatusPtr,
	SessionOptionsAppendExecutionProvider_TensorRT_V2:   proc(
		options: ^OrtSessionOptions,
		tensorrt_options: ^OrtTensorRTProviderOptionsV2,
	) -> OrtStatusPtr,
	CreateTensorRTProviderOptions:                       proc(
		out: ^^OrtTensorRTProviderOptionsV2,
	) -> OrtStatusPtr,
	UpdateTensorRTProviderOptions:                       proc(
		tensorrt_options: ^OrtTensorRTProviderOptionsV2,
		provider_options_keys: [^]cstring,
		provider_options_values: [^]cstring,
		num_keys: c.size_t,
	) -> OrtStatusPtr,
	GetTensorRTProviderOptionsAsString:                  proc(
		tensorrt_options: ^OrtTensorRTProviderOptionsV2,
		allocator: ^OrtAllocator,
		ptr: [^]cstring,
	) -> OrtStatusPtr,
	ReleaseTensorRTProviderOptions:                      proc(
		input: ^OrtTensorRTProviderOptionsV2,
	),
	EnableOrtCustomOps:                                  proc(
		options: ^OrtSessionOptions,
	) -> OrtStatusPtr,
	RegisterAllocator:                                   proc(
		env: ^OrtEnv,
		allocator: ^OrtAllocator,
	) -> OrtStatusPtr,
	UnregisterAllocator:                                 proc(
		env: ^OrtEnv,
		mem_info: ^OrtMemoryInfo,
	) -> OrtStatusPtr,
	IsSparseTensor:                                      proc(
		value: ^OrtValue,
		out: ^c.int,
	) -> OrtStatusPtr,
	CreateSparseTensorAsOrtValue:                        proc(
		allocator: ^OrtAllocator,
		dense_shape: ^c.int64_t,
		dense_shape_len: c.size_t,
		type: ONNXTensorElementDataType,
		out: ^^OrtValue,
	) -> OrtStatusPtr,
	FillSparseTensorCoo:                                 proc(
		ort_value: ^OrtValue,
		data_mem_info: ^OrtMemoryInfo,
		values_shape: ^c.int64_t,
		values_shape_len: c.size_t,
		values: rawptr,
		indices_data: ^c.int64_t,
		indices_num: c.size_t,
	) -> OrtStatusPtr,
	FillSparseTensorCsr:                                 proc(
		ort_value: ^OrtValue,
		data_mem_info: ^OrtMemoryInfo,
		values_shape: ^c.int64_t,
		values_shape_len: c.size_t,
		values: rawptr,
		inner_indices_data: ^c.int64_t,
		inner_indices_num: c.size_t,
		outer_indices_data: ^c.int64_t,
		outer_indices_num: c.size_t,
	) -> OrtStatusPtr,
	FillSparseTensorBlockSparse:                         proc(
		ort_value: ^OrtValue,
		data_mem_info: ^OrtMemoryInfo,
		values_shape: ^c.int64_t,
		values_shape_len: c.size_t,
		values: rawptr,
		indices_shape_data: ^c.int64_t,
		indices_shape_len: c.size_t,
		indices_data: ^c.int32_t,
	) -> OrtStatusPtr,
	CreateSparseTensorWithValuesAsOrtValue:              proc(
		info: ^OrtMemoryInfo,
		p_data: rawptr,
		dense_shape: ^c.int64_t,
		dense_shape_len: c.size_t,
		values_shape: ^c.int64_t,
		values_shape_len: c.size_t,
		type: ONNXTensorElementDataType,
		out: ^^OrtValue,
	) -> OrtStatusPtr,
	UseCooIndices:                                       proc(
		ort_value: ^OrtValue,
		indices_data: ^c.int64_t,
		indices_num: c.size_t,
	) -> OrtStatusPtr,
	UseCsrIndices:                                       proc(
		ort_value: ^OrtValue,
		inner_data: ^c.int64_t,
		inner_num: c.size_t,
		outer_data: ^c.int64_t,
		outer_num: c.size_t,
	) -> OrtStatusPtr,
	UseBlockSparseIndices:                               proc(
		ort_value: ^OrtValue,
		indices_shape: ^c.int64_t,
		indices_shape_len: c.size_t,
		indices_data: ^c.int32_t,
	) -> OrtStatusPtr,
	GetSparseTensorFormat:                               proc(
		ort_value: ^OrtValue,
		out: ^OrtSparseFormat,
	) -> OrtStatusPtr,
	GetSparseTensorValuesTypeAndShape:                   proc(
		ort_value: ^OrtValue,
		out: ^^OrtTensorTypeAndShapeInfo,
	) -> OrtStatusPtr,
	GetSparseTensorValues:                               proc(
		ort_value: ^OrtValue,
		out: ^rawptr,
	) -> OrtStatusPtr,
	GetSparseTensorIndicesTypeShape:                     proc(
		ort_value: ^OrtValue,
		indices_format: OrtSparseIndicesFormat,
		out: ^^OrtTensorTypeAndShapeInfo,
	) -> OrtStatusPtr,
	GetSparseTensorIndices:                              proc(
		ort_value: ^OrtValue,
		indices_format: OrtSparseIndicesFormat,
		num_indices: ^c.size_t,
		indices: ^rawptr,
	) -> OrtStatusPtr,
	HasValue:                                            proc(
		value: ^OrtValue,
		out: ^c.int,
	) -> OrtStatusPtr,
	KernelContext_GetGPUComputeStream:                   proc(
		context_: ^OrtKernelContext,
		out: ^rawptr,
	) -> OrtStatusPtr,
	GetTensorMemoryInfo:                                 proc(
		value: ^OrtValue,
		mem_info: ^^OrtMemoryInfo,
	) -> OrtStatusPtr,
	GetExecutionProviderApi:                             proc(
		provider_name: cstring,
		version: c.uint32_t,
		provider_api: ^rawptr,
	) -> OrtStatusPtr,
	SessionOptionsSetCustomCreateThreadFn:               proc(
		options: ^OrtSessionOptions,
		ort_custom_create_thread_fn: OrtCustomCreateThreadFn,
	) -> OrtStatusPtr,
	SessionOptionsSetCustomThreadCreationOptions:        proc(
		options: ^OrtSessionOptions,
		ort_custom_thread_creation_options: rawptr,
	) -> OrtStatusPtr,
	SessionOptionsSetCustomJoinThreadFn:                 proc(
		options: ^OrtSessionOptions,
		ort_custom_join_thread_fn: OrtCustomJoinThreadFn,
	) -> OrtStatusPtr,
	SetGlobalCustomCreateThreadFn:                       proc(
		tp_options: ^OrtThreadingOptions,
		ort_custom_create_thread_fn: OrtCustomCreateThreadFn,
	) -> OrtStatusPtr,
	SetGlobalCustomThreadCreationOptions:                proc(
		tp_options: ^OrtThreadingOptions,
		ort_custom_thread_creation_options: rawptr,
	) -> OrtStatusPtr,
	SetGlobalCustomJoinThreadFn:                         proc(
		tp_options: ^OrtThreadingOptions,
		ort_custom_join_thread_fn: OrtCustomJoinThreadFn,
	) -> OrtStatusPtr,
	SynchronizeBoundInputs:                              proc(
		binding_ptr: ^OrtIoBinding,
	) -> OrtStatusPtr,
	SynchronizeBoundOutputs:                             proc(
		binding_ptr: ^OrtIoBinding,
	) -> OrtStatusPtr,
	SessionOptionsAppendExecutionProvider_CUDA_V2:       proc(
		options: ^OrtSessionOptions,
		cuda_options: ^OrtCUDAProviderOptionsV2,
	) -> OrtStatusPtr,
	CreateCUDAProviderOptions:                           proc(
		out: ^^OrtCUDAProviderOptionsV2,
	) -> OrtStatusPtr,
	UpdateCUDAProviderOptions:                           proc(
		cuda_options: ^OrtCUDAProviderOptionsV2,
		provider_options_keys: [^]cstring,
		provider_options_values: [^]cstring,
		num_keys: c.size_t,
	) -> OrtStatusPtr,
	GetCUDAProviderOptionsAsString:                      proc(
		cuda_options: ^OrtCUDAProviderOptionsV2,
		allocator: ^OrtAllocator,
		ptr: [^]cstring,
	) -> OrtStatusPtr,
	ReleaseCUDAProviderOptions:                          proc(input: ^OrtCUDAProviderOptionsV2),
	SessionOptionsAppendExecutionProvider_MIGraphX:      proc(
		options: ^OrtSessionOptions,
		migraphx_options: ^OrtMIGraphXProviderOptions,
	) -> OrtStatusPtr,
	AddExternalInitializers:                             proc(
		options: ^OrtSessionOptions,
		initializer_names: [^]cstring,
		initializers: ^^OrtValue,
		initializers_num: c.size_t,
	) -> OrtStatusPtr,
	CreateOpAttr:                                        proc(
		name: cstring,
		data: rawptr,
		len: c.int,
		type: OrtOpAttrType,
		op_attr: ^^OrtOpAttr,
	) -> OrtStatusPtr,
	ReleaseOpAttr:                                       proc(input: ^OrtOpAttr),
	CreateOp:                                            proc(
		info: ^OrtKernelInfo,
		op_name: cstring,
		domain: cstring,
		version: c.int,
		type_constraint_names: [^]cstring,
		type_constraint_values: ^ONNXTensorElementDataType,
		type_constraint_count: c.int,
		attr_values: ^^OrtOpAttr,
		attr_count: c.int,
		input_count: c.int,
		output_count: c.int,
		ort_op: ^^OrtOp,
	) -> OrtStatusPtr,
	InvokeOp:                                            proc(
		context_: ^OrtKernelContext,
		ort_op: ^OrtOp,
		input_values: ^^OrtValue,
		input_count: c.int,
		output_values: ^^OrtValue,
		output_count: c.int,
	) -> OrtStatusPtr,
	ReleaseOp:                                           proc(input: ^OrtOp),
	SessionOptionsAppendExecutionProvider:               proc(
		options: ^OrtSessionOptions,
		provider_name: cstring,
		provider_options_keys: [^]cstring,
		provider_options_values: [^]cstring,
		num_keys: c.size_t,
	) -> OrtStatusPtr,
	CopyKernelInfo:                                      proc(
		info: ^OrtKernelInfo,
		info_copy: ^^OrtKernelInfo,
	) -> OrtStatusPtr,
	ReleaseKernelInfo:                                   proc(input: ^OrtKernelInfo),
	GetTrainingApi:                                      proc(
		version: c.uint32_t,
	) -> ^OrtTrainingApi,
	SessionOptionsAppendExecutionProvider_CANN:          proc(
		options: ^OrtSessionOptions,
		cann_options: ^OrtCANNProviderOptions,
	) -> OrtStatusPtr,
	CreateCANNProviderOptions:                           proc(
		out: ^^OrtCANNProviderOptions,
	) -> OrtStatusPtr,
	UpdateCANNProviderOptions:                           proc(
		cann_options: ^OrtCANNProviderOptions,
		provider_options_keys: [^]cstring,
		provider_options_values: [^]cstring,
		num_keys: c.size_t,
	) -> OrtStatusPtr,
	GetCANNProviderOptionsAsString:                      proc(
		cann_options: ^OrtCANNProviderOptions,
		allocator: ^OrtAllocator,
		ptr: [^]cstring,
	) -> OrtStatusPtr,
	ReleaseCANNProviderOptions:                          proc(input: ^OrtCANNProviderOptions),
	MemoryInfoGetDeviceType:                             proc(
		ptr: ^OrtMemoryInfo,
		out: ^OrtMemoryInfoDeviceType,
	),
	UpdateEnvWithCustomLogLevel:                         proc(
		ort_env: ^OrtEnv,
		log_severity_level: OrtLoggingLevel,
	) -> OrtStatusPtr,
	SetGlobalIntraOpThreadAffinity:                      proc(
		tp_options: ^OrtThreadingOptions,
		affinity_string: cstring,
	) -> OrtStatusPtr,
	RegisterCustomOpsLibrary_V2:                         proc(
		options: ^OrtSessionOptions,
		library_name: cstring,
	) -> OrtStatusPtr,
	RegisterCustomOpsUsingFunction:                      proc(
		options: ^OrtSessionOptions,
		registration_func_name: cstring,
	) -> OrtStatusPtr,
	KernelInfo_GetInputCount:                            proc(
		info: ^OrtKernelInfo,
		out: ^c.size_t,
	) -> OrtStatusPtr,
	KernelInfo_GetOutputCount:                           proc(
		info: ^OrtKernelInfo,
		out: ^c.size_t,
	) -> OrtStatusPtr,
	KernelInfo_GetInputName:                             proc(
		info: ^OrtKernelInfo,
		index: c.size_t,
		out: cstring,
		size: ^c.size_t,
	) -> OrtStatusPtr,
	KernelInfo_GetOutputName:                            proc(
		info: ^OrtKernelInfo,
		index: c.size_t,
		out: cstring,
		size: ^c.size_t,
	) -> OrtStatusPtr,
	KernelInfo_GetInputTypeInfo:                         proc(
		info: ^OrtKernelInfo,
		index: c.size_t,
		type_info: ^^OrtTypeInfo,
	) -> OrtStatusPtr,
	KernelInfo_GetOutputTypeInfo:                        proc(
		info: ^OrtKernelInfo,
		index: c.size_t,
		type_info: ^^OrtTypeInfo,
	) -> OrtStatusPtr,
	KernelInfoGetAttribute_tensor:                       proc(
		info: ^OrtKernelInfo,
		name: cstring,
		allocator: ^OrtAllocator,
		out: ^^OrtValue,
	) -> OrtStatusPtr,
	HasSessionConfigEntry:                               proc(
		options: ^OrtSessionOptions,
		config_key: cstring,
		out: ^c.int,
	) -> OrtStatusPtr,
	GetSessionConfigEntry:                               proc(
		options: ^OrtSessionOptions,
		config_key: cstring,
		config_value: cstring,
		size: ^c.size_t,
	) -> OrtStatusPtr,
	SessionOptionsAppendExecutionProvider_Dnnl:          proc(
		options: ^OrtSessionOptions,
		dnnl_options: ^OrtDnnlProviderOptions,
	) -> OrtStatusPtr,
	CreateDnnlProviderOptions:                           proc(
		out: ^^OrtDnnlProviderOptions,
	) -> OrtStatusPtr,
	UpdateDnnlProviderOptions:                           proc(
		dnnl_options: ^OrtDnnlProviderOptions,
		provider_options_keys: [^]cstring,
		provider_options_values: [^]cstring,
		num_keys: c.size_t,
	) -> OrtStatusPtr,
	GetDnnlProviderOptionsAsString:                      proc(
		dnnl_options: ^OrtDnnlProviderOptions,
		allocator: ^OrtAllocator,
		ptr: [^]cstring,
	) -> OrtStatusPtr,
	ReleaseDnnlProviderOptions:                          proc(input: ^OrtDnnlProviderOptions),
	KernelInfo_GetNodeName:                              proc(
		info: ^OrtKernelInfo,
		out: cstring,
		size: ^c.size_t,
	) -> OrtStatusPtr,
	KernelInfo_GetLogger:                                proc(
		info: ^OrtKernelInfo,
		logger: ^^OrtLogger,
	) -> OrtStatusPtr,
	KernelContext_GetLogger:                             proc(
		context_: ^OrtKernelContext,
		logger: ^^OrtLogger,
	) -> OrtStatusPtr,
	Logger_LogMessage:                                   proc(
		logger: ^OrtLogger,
		log_severity_level: OrtLoggingLevel,
		message: cstring,
		file_path: cstring,
		line_number: c.int,
		func_name: cstring,
	) -> OrtStatusPtr,
	Logger_GetLoggingSeverityLevel:                      proc(
		logger: ^OrtLogger,
		out: ^OrtLoggingLevel,
	) -> OrtStatusPtr,
	KernelInfoGetConstantInput_tensor:                   proc(
		info: ^OrtKernelInfo,
		index: c.size_t,
		is_constant: ^c.int,
		out: ^^OrtValue,
	) -> OrtStatusPtr,
	CastTypeInfoToOptionalTypeInfo:                      proc(
		type_info: ^OrtTypeInfo,
		out: ^^OrtOptionalTypeInfo,
	) -> OrtStatusPtr,
	GetOptionalContainedTypeInfo:                        proc(
		optional_type_info: ^OrtOptionalTypeInfo,
		out: ^^OrtTypeInfo,
	) -> OrtStatusPtr,
	GetResizedStringTensorElementBuffer:                 proc(
		value: ^OrtValue,
		index: c.size_t,
		length_in_bytes: c.size_t,
		buffer: [^]cstring,
	) -> OrtStatusPtr,
	KernelContext_GetAllocator:                          proc(
		context_: ^OrtKernelContext,
		mem_info: ^OrtMemoryInfo,
		out: ^^OrtAllocator,
	) -> OrtStatusPtr,
	GetBuildInfoString:                                  proc() -> cstring,
	CreateROCMProviderOptions:                           proc(
		out: ^^OrtROCMProviderOptions,
	) -> OrtStatusPtr,
	UpdateROCMProviderOptions:                           proc(
		rocm_options: ^OrtROCMProviderOptions,
		provider_options_keys: [^]cstring,
		provider_options_values: [^]cstring,
		num_keys: c.size_t,
	) -> OrtStatusPtr,
	GetROCMProviderOptionsAsString:                      proc(
		rocm_options: ^OrtROCMProviderOptions,
		allocator: ^OrtAllocator,
		ptr: [^]cstring,
	) -> OrtStatusPtr,
	ReleaseROCMProviderOptions:                          proc(input: ^OrtROCMProviderOptions),
	CreateAndRegisterAllocatorV2:                        proc(
		env: ^OrtEnv,
		provider_type: cstring,
		mem_info: ^OrtMemoryInfo,
		arena_cfg: ^OrtArenaCfg,
		provider_options_keys: [^]cstring,
		provider_options_values: [^]cstring,
		num_keys: c.size_t,
	) -> OrtStatusPtr,
	RunAsync:                                            proc(
		session: ^OrtSession,
		run_options: ^OrtRunOptions,
		input_names: [^]cstring,
		input: ^^OrtValue,
		input_len: c.size_t,
		output_names: [^]cstring,
		output_names_len: c.size_t,
		output: ^^OrtValue,
		run_async_callback: RunAsyncCallbackFn,
		user_data: rawptr,
	) -> OrtStatusPtr,
	UpdateTensorRTProviderOptionsWithValue:              proc(
		tensorrt_options: ^OrtTensorRTProviderOptionsV2,
		key: cstring,
		value: rawptr,
	) -> OrtStatusPtr,
	GetTensorRTProviderOptionsByName:                    proc(
		tensorrt_options: ^OrtTensorRTProviderOptionsV2,
		key: cstring,
		ptr: ^rawptr,
	) -> OrtStatusPtr,
	UpdateCUDAProviderOptionsWithValue:                  proc(
		cuda_options: ^OrtCUDAProviderOptionsV2,
		key: cstring,
		value: rawptr,
	) -> OrtStatusPtr,
	GetCUDAProviderOptionsByName:                        proc(
		cuda_options: ^OrtCUDAProviderOptionsV2,
		key: cstring,
		ptr: ^rawptr,
	) -> OrtStatusPtr,
	KernelContext_GetResource:                           proc(
		context_: ^OrtKernelContext,
		resouce_version: c.int,
		resource_id: c.int,
		resource: ^rawptr,
	) -> OrtStatusPtr,
	SetUserLoggingFunction:                              proc(
		options: ^OrtSessionOptions,
		user_logging_function: OrtLoggingFunction,
		user_logging_param: rawptr,
	) -> OrtStatusPtr,
	ShapeInferContext_GetInputCount:                     proc(
		context_: ^OrtShapeInferContext,
		out: ^c.size_t,
	) -> OrtStatusPtr,
	ShapeInferContext_GetInputTypeShape:                 proc(
		context_: ^OrtShapeInferContext,
		index: c.size_t,
		info: ^^OrtTensorTypeAndShapeInfo,
	) -> OrtStatusPtr,
	ShapeInferContext_GetAttribute:                      proc(
		context_: ^OrtShapeInferContext,
		attr_name: cstring,
		attr: ^^OrtOpAttr,
	) -> OrtStatusPtr,
	ShapeInferContext_SetOutputTypeShape:                proc(
		context_: ^OrtShapeInferContext,
		index: c.size_t,
		info: ^OrtTensorTypeAndShapeInfo,
	) -> OrtStatusPtr,
	SetSymbolicDimensions:                               proc(
		info: ^OrtTensorTypeAndShapeInfo,
		dim_params: [^]cstring,
		dim_params_length: c.size_t,
	) -> OrtStatusPtr,
	ReadOpAttr:                                          proc(
		op_attr: ^OrtOpAttr,
		type: OrtOpAttrType,
		data: rawptr,
		len: c.size_t,
		out: ^c.size_t,
	) -> OrtStatusPtr,
	SetDeterministicCompute:                             proc(
		options: ^OrtSessionOptions,
		value: c.bool,
	) -> OrtStatusPtr,
	KernelContext_ParallelFor:                           proc(
		context_: ^OrtKernelContext,
		fn: proc(_: rawptr, _: c.size_t),
		total: c.size_t,
		num_batch: c.size_t,
		usr_data: rawptr,
	) -> OrtStatusPtr,
	SessionOptionsAppendExecutionProvider_OpenVINO_V2:   proc(
		options: ^OrtSessionOptions,
		provider_options_keys: [^]cstring,
		provider_options_values: [^]cstring,
		num_keys: c.size_t,
	) -> OrtStatusPtr,
}

OrtCustomOp :: struct {
	version:                      c.uint32_t,
	CreateKernel:                 proc(
		op: ^OrtCustomOp,
		api: ^OrtApi,
		info: ^OrtKernelInfo,
	) -> rawptr,
	GetName:                      proc(op: ^OrtCustomOp) -> cstring,
	GetExecutionProviderType:     proc(op: ^OrtCustomOp) -> cstring,
	GetInputType:                 proc(
		op: ^OrtCustomOp,
		index: c.size_t,
	) -> ONNXTensorElementDataType,
	GetInputTypeCount:            proc(op: ^OrtCustomOp) -> c.size_t,
	GetOutputType:                proc(
		op: ^OrtCustomOp,
		index: c.size_t,
	) -> ONNXTensorElementDataType,
	GetOutputTypeCount:           proc(op: ^OrtCustomOp) -> c.size_t,
	KernelCompute:                proc(op_kernel: rawptr, context_: ^OrtKernelContext),
	KernelDestroy:                proc(op_kernel: rawptr),
	GetInputCharacteristic:       proc(
		op: ^OrtCustomOp,
		index: c.size_t,
	) -> OrtCustomOpInputOutputCharacteristic,
	GetOutputCharacteristic:      proc(
		op: ^OrtCustomOp,
		index: c.size_t,
	) -> OrtCustomOpInputOutputCharacteristic,
	GetInputMemoryType:           proc(op: ^OrtCustomOp, index: c.size_t) -> OrtMemType,
	GetVariadicInputMinArity:     proc(op: ^OrtCustomOp) -> c.int,
	GetVariadicInputHomogeneity:  proc(op: ^OrtCustomOp) -> c.int,
	GetVariadicOutputMinArity:    proc(op: ^OrtCustomOp) -> c.int,
	GetVariadicOutputHomogeneity: proc(op: ^OrtCustomOp) -> c.int,
	CreateKernelV2:               proc(
		op: ^OrtCustomOp,
		api: ^OrtApi,
		info: ^OrtKernelInfo,
		kernel: ^rawptr,
	) -> OrtStatusPtr,
	KernelComputeV2:              proc(
		op_kernel: rawptr,
		context_: ^OrtKernelContext,
	) -> OrtStatusPtr,
	InferOutputShapeFn:           proc(op: ^OrtCustomOp, _: ^OrtShapeInferContext) -> OrtStatusPtr,
	GetStartVersion:              proc(op: ^OrtCustomOp) -> c.int,
	GetEndVersion:                proc(op: ^OrtCustomOp) -> c.int,
}

// Functions:
@(default_calling_convention = "c")
foreign onnx {
	Alloc :: proc(this_: ^OrtAllocator, size: c.size_t) -> rawptr ---
	Free :: proc(this_: ^OrtAllocator, p: rawptr) ---
	Info :: proc(this_: ^OrtAllocator) -> ^OrtMemoryInfo ---
	GetApi :: proc(version: c.uint32_t) -> ^OrtApi ---
	GetVersionString :: proc() -> cstring ---
	CreateStatus :: proc(code: OrtErrorCode, msg: cstring) -> ^OrtStatus ---
	GetErrorCode :: proc(status: ^OrtStatus) -> OrtErrorCode ---
	GetErrorMessage :: proc(status: ^OrtStatus) -> cstring ---
	CreateEnv :: proc(log_severity_level: OrtLoggingLevel, logid: cstring, out: ^^OrtEnv) -> OrtStatusPtr ---
	CreateEnvWithCustomLogger :: proc(logging_function: OrtLoggingFunction, logger_param: rawptr, log_severity_level: OrtLoggingLevel, logid: cstring, out: ^^OrtEnv) -> OrtStatusPtr ---
	EnableTelemetryEvents :: proc(env: ^OrtEnv) -> OrtStatusPtr ---
	DisableTelemetryEvents :: proc(env: ^OrtEnv) -> OrtStatusPtr ---
	CreateSession :: proc(env: ^OrtEnv, model_path: cstring, options: ^OrtSessionOptions, out: ^^OrtSession) -> OrtStatusPtr ---
	CreateSessionFromArray :: proc(env: ^OrtEnv, model_data: rawptr, model_data_length: c.size_t, options: ^OrtSessionOptions, out: ^^OrtSession) -> OrtStatusPtr ---
	Run :: proc(session: ^OrtSession, run_options: ^OrtRunOptions, input_names: [^]cstring, inputs: ^^OrtValue, input_len: c.size_t, output_names: [^]cstring, output_names_len: c.size_t, outputs: ^^OrtValue) -> OrtStatusPtr ---
	CreateSessionOptions :: proc(options: ^^OrtSessionOptions) -> OrtStatusPtr ---
	SetOptimizedModelFilePath :: proc(options: ^OrtSessionOptions, optimized_model_filepath: cstring) -> OrtStatusPtr ---
	CloneSessionOptions :: proc(in_options: ^OrtSessionOptions, out_options: ^^OrtSessionOptions) -> OrtStatusPtr ---
	SetSessionExecutionMode :: proc(options: ^OrtSessionOptions, execution_mode: ExecutionMode) -> OrtStatusPtr ---
	EnableProfiling :: proc(options: ^OrtSessionOptions, profile_file_prefix: cstring) -> OrtStatusPtr ---
	DisableProfiling :: proc(options: ^OrtSessionOptions) -> OrtStatusPtr ---
	EnableMemPattern :: proc(options: ^OrtSessionOptions) -> OrtStatusPtr ---
	DisableMemPattern :: proc(options: ^OrtSessionOptions) -> OrtStatusPtr ---
	EnableCpuMemArena :: proc(options: ^OrtSessionOptions) -> OrtStatusPtr ---
	DisableCpuMemArena :: proc(options: ^OrtSessionOptions) -> OrtStatusPtr ---
	SetSessionLogId :: proc(options: ^OrtSessionOptions, logid: cstring) -> OrtStatusPtr ---
	SetSessionLogVerbosityLevel :: proc(options: ^OrtSessionOptions, session_log_verbosity_level: c.int) -> OrtStatusPtr ---
	SetSessionLogSeverityLevel :: proc(options: ^OrtSessionOptions, session_log_severity_level: c.int) -> OrtStatusPtr ---
	SetSessionGraphOptimizationLevel :: proc(options: ^OrtSessionOptions, graph_optimization_level: GraphOptimizationLevel) -> OrtStatusPtr ---
	SetIntraOpNumThreads :: proc(options: ^OrtSessionOptions, intra_op_num_threads: c.int) -> OrtStatusPtr ---
	SetInterOpNumThreads :: proc(options: ^OrtSessionOptions, inter_op_num_threads: c.int) -> OrtStatusPtr ---
	CreateCustomOpDomain :: proc(domain: cstring, out: ^^OrtCustomOpDomain) -> OrtStatusPtr ---
	CustomOpDomain_Add :: proc(custom_op_domain: ^OrtCustomOpDomain, op: ^OrtCustomOp) -> OrtStatusPtr ---
	AddCustomOpDomain :: proc(options: ^OrtSessionOptions, custom_op_domain: ^OrtCustomOpDomain) -> OrtStatusPtr ---
	RegisterCustomOpsLibrary :: proc(options: ^OrtSessionOptions, library_path: cstring, library_handle: ^rawptr) -> OrtStatusPtr ---
	SessionGetInputCount :: proc(session: ^OrtSession, out: ^c.size_t) -> OrtStatusPtr ---
	SessionGetOutputCount :: proc(session: ^OrtSession, out: ^c.size_t) -> OrtStatusPtr ---
	SessionGetOverridableInitializerCount :: proc(session: ^OrtSession, out: ^c.size_t) -> OrtStatusPtr ---
	SessionGetInputTypeInfo :: proc(session: ^OrtSession, index: c.size_t, type_info: ^^OrtTypeInfo) -> OrtStatusPtr ---
	SessionGetOutputTypeInfo :: proc(session: ^OrtSession, index: c.size_t, type_info: ^^OrtTypeInfo) -> OrtStatusPtr ---
	SessionGetOverridableInitializerTypeInfo :: proc(session: ^OrtSession, index: c.size_t, type_info: ^^OrtTypeInfo) -> OrtStatusPtr ---
	SessionGetInputName :: proc(session: ^OrtSession, index: c.size_t, allocator: ^OrtAllocator, value: [^]cstring) -> OrtStatusPtr ---
	SessionGetOutputName :: proc(session: ^OrtSession, index: c.size_t, allocator: ^OrtAllocator, value: [^]cstring) -> OrtStatusPtr ---
	SessionGetOverridableInitializerName :: proc(session: ^OrtSession, index: c.size_t, allocator: ^OrtAllocator, value: [^]cstring) -> OrtStatusPtr ---
	CreateRunOptions :: proc(out: ^^OrtRunOptions) -> OrtStatusPtr ---
	RunOptionsSetRunLogVerbosityLevel :: proc(options: ^OrtRunOptions, log_verbosity_level: c.int) -> OrtStatusPtr ---
	RunOptionsSetRunLogSeverityLevel :: proc(options: ^OrtRunOptions, log_severity_level: c.int) -> OrtStatusPtr ---
	RunOptionsSetRunTag :: proc(options: ^OrtRunOptions, run_tag: cstring) -> OrtStatusPtr ---
	RunOptionsGetRunLogVerbosityLevel :: proc(options: ^OrtRunOptions, log_verbosity_level: ^c.int) -> OrtStatusPtr ---
	RunOptionsGetRunLogSeverityLevel :: proc(options: ^OrtRunOptions, log_severity_level: ^c.int) -> OrtStatusPtr ---
	RunOptionsGetRunTag :: proc(options: ^OrtRunOptions, run_tag: [^]cstring) -> OrtStatusPtr ---
	RunOptionsSetTerminate :: proc(options: ^OrtRunOptions) -> OrtStatusPtr ---
	RunOptionsUnsetTerminate :: proc(options: ^OrtRunOptions) -> OrtStatusPtr ---
	CreateTensorAsOrtValue :: proc(allocator: ^OrtAllocator, shape: ^c.int64_t, shape_len: c.size_t, type: ONNXTensorElementDataType, out: ^^OrtValue) -> OrtStatusPtr ---
	CreateTensorWithDataAsOrtValue :: proc(info: ^OrtMemoryInfo, p_data: rawptr, p_data_len: c.size_t, shape: ^c.int64_t, shape_len: c.size_t, type: ONNXTensorElementDataType, out: ^^OrtValue) -> OrtStatusPtr ---
	IsTensor :: proc(value: ^OrtValue, out: ^c.int) -> OrtStatusPtr ---
	GetTensorMutableData :: proc(value: ^OrtValue, out: ^rawptr) -> OrtStatusPtr ---
	FillStringTensor :: proc(value: ^OrtValue, s: [^]cstring, s_len: c.size_t) -> OrtStatusPtr ---
	GetStringTensorDataLength :: proc(value: ^OrtValue, len: ^c.size_t) -> OrtStatusPtr ---
	GetStringTensorContent :: proc(value: ^OrtValue, s: rawptr, s_len: c.size_t, offsets: ^c.size_t, offsets_len: c.size_t) -> OrtStatusPtr ---
	CastTypeInfoToTensorInfo :: proc(type_info: ^OrtTypeInfo, out: ^^OrtTensorTypeAndShapeInfo) -> OrtStatusPtr ---
	GetOnnxTypeFromTypeInfo :: proc(type_info: ^OrtTypeInfo, out: ^ONNXType) -> OrtStatusPtr ---
	CreateTensorTypeAndShapeInfo :: proc(out: ^^OrtTensorTypeAndShapeInfo) -> OrtStatusPtr ---
	SetTensorElementType :: proc(info: ^OrtTensorTypeAndShapeInfo, type: ONNXTensorElementDataType) -> OrtStatusPtr ---
	SetDimensions :: proc(info: ^OrtTensorTypeAndShapeInfo, dim_values: ^c.int64_t, dim_count: c.size_t) -> OrtStatusPtr ---
	GetTensorElementType :: proc(info: ^OrtTensorTypeAndShapeInfo, out: ^ONNXTensorElementDataType) -> OrtStatusPtr ---
	GetDimensionsCount :: proc(info: ^OrtTensorTypeAndShapeInfo, out: ^c.size_t) -> OrtStatusPtr ---
	GetDimensions :: proc(info: ^OrtTensorTypeAndShapeInfo, dim_values: ^c.int64_t, dim_values_length: c.size_t) -> OrtStatusPtr ---
	GetSymbolicDimensions :: proc(info: ^OrtTensorTypeAndShapeInfo, dim_params: [^]cstring, dim_params_length: c.size_t) -> OrtStatusPtr ---
	GetTensorShapeElementCount :: proc(info: ^OrtTensorTypeAndShapeInfo, out: ^c.size_t) -> OrtStatusPtr ---
	GetTensorTypeAndShape :: proc(value: ^OrtValue, out: ^^OrtTensorTypeAndShapeInfo) -> OrtStatusPtr ---
	GetTypeInfo :: proc(value: ^OrtValue, out: ^^OrtTypeInfo) -> OrtStatusPtr ---
	GetValueType :: proc(value: ^OrtValue, out: ^ONNXType) -> OrtStatusPtr ---
	CreateMemoryInfo :: proc(name: cstring, type: OrtAllocatorType, id: c.int, mem_type: OrtMemType, out: ^^OrtMemoryInfo) -> OrtStatusPtr ---
	CreateCpuMemoryInfo :: proc(type: OrtAllocatorType, mem_type: OrtMemType, out: ^^OrtMemoryInfo) -> OrtStatusPtr ---
	CompareMemoryInfo :: proc(info1: ^OrtMemoryInfo, info2: ^OrtMemoryInfo, out: ^c.int) -> OrtStatusPtr ---
	MemoryInfoGetName :: proc(ptr: ^OrtMemoryInfo, out: [^]cstring) -> OrtStatusPtr ---
	MemoryInfoGetId :: proc(ptr: ^OrtMemoryInfo, out: ^c.int) -> OrtStatusPtr ---
	MemoryInfoGetMemType :: proc(ptr: ^OrtMemoryInfo, out: ^OrtMemType) -> OrtStatusPtr ---
	MemoryInfoGetType :: proc(ptr: ^OrtMemoryInfo, out: ^OrtAllocatorType) -> OrtStatusPtr ---
	AllocatorAlloc :: proc(ort_allocator: ^OrtAllocator, size: c.size_t, out: ^rawptr) -> OrtStatusPtr ---
	AllocatorFree :: proc(ort_allocator: ^OrtAllocator, p: rawptr) -> OrtStatusPtr ---
	AllocatorGetInfo :: proc(ort_allocator: ^OrtAllocator, out: ^^OrtMemoryInfo) -> OrtStatusPtr ---
	GetAllocatorWithDefaultOptions :: proc(out: ^^OrtAllocator) -> OrtStatusPtr ---
	AddFreeDimensionOverride :: proc(options: ^OrtSessionOptions, dim_denotation: cstring, dim_value: c.int64_t) -> OrtStatusPtr ---
	GetValue :: proc(value: ^OrtValue, index: c.int, allocator: ^OrtAllocator, out: ^^OrtValue) -> OrtStatusPtr ---
	GetValueCount :: proc(value: ^OrtValue, out: ^c.size_t) -> OrtStatusPtr ---
	CreateValue :: proc(in_: ^^OrtValue, num_values: c.size_t, value_type: ONNXType, out: ^^OrtValue) -> OrtStatusPtr ---
	CreateOpaqueValue :: proc(domain_name: cstring, type_name: cstring, data_container: rawptr, data_container_size: c.size_t, out: ^^OrtValue) -> OrtStatusPtr ---
	GetOpaqueValue :: proc(domain_name: cstring, type_name: cstring, in_: ^OrtValue, data_container: rawptr, data_container_size: c.size_t) -> OrtStatusPtr ---
	KernelInfoGetAttribute_float :: proc(info: ^OrtKernelInfo, name: cstring, out: ^c.float) -> OrtStatusPtr ---
	KernelInfoGetAttribute_int64 :: proc(info: ^OrtKernelInfo, name: cstring, out: ^c.int64_t) -> OrtStatusPtr ---
	KernelInfoGetAttribute_string :: proc(info: ^OrtKernelInfo, name: cstring, out: cstring, size: ^c.size_t) -> OrtStatusPtr ---
	KernelContext_GetInputCount :: proc(context_: ^OrtKernelContext, out: ^c.size_t) -> OrtStatusPtr ---
	KernelContext_GetOutputCount :: proc(context_: ^OrtKernelContext, out: ^c.size_t) -> OrtStatusPtr ---
	KernelContext_GetInput :: proc(context_: ^OrtKernelContext, index: c.size_t, out: ^^OrtValue) -> OrtStatusPtr ---
	KernelContext_GetOutput :: proc(context_: ^OrtKernelContext, index: c.size_t, dim_values: ^c.int64_t, dim_count: c.size_t, out: ^^OrtValue) -> OrtStatusPtr ---
	ReleaseEnv :: proc(input: ^OrtEnv) ---
	ReleaseStatus :: proc(input: ^OrtStatus) ---
	ReleaseMemoryInfo :: proc(input: ^OrtMemoryInfo) ---
	ReleaseSession :: proc(input: ^OrtSession) ---
	ReleaseValue :: proc(input: ^OrtValue) ---
	ReleaseRunOptions :: proc(input: ^OrtRunOptions) ---
	ReleaseTypeInfo :: proc(input: ^OrtTypeInfo) ---
	ReleaseTensorTypeAndShapeInfo :: proc(input: ^OrtTensorTypeAndShapeInfo) ---
	ReleaseSessionOptions :: proc(input: ^OrtSessionOptions) ---
	ReleaseCustomOpDomain :: proc(input: ^OrtCustomOpDomain) ---
	GetDenotationFromTypeInfo :: proc(type_info: ^OrtTypeInfo, denotation: [^]cstring, len: ^c.size_t) -> OrtStatusPtr ---
	CastTypeInfoToMapTypeInfo :: proc(type_info: ^OrtTypeInfo, out: ^^OrtMapTypeInfo) -> OrtStatusPtr ---
	CastTypeInfoToSequenceTypeInfo :: proc(type_info: ^OrtTypeInfo, out: ^^OrtSequenceTypeInfo) -> OrtStatusPtr ---
	GetMapKeyType :: proc(map_type_info: ^OrtMapTypeInfo, out: ^ONNXTensorElementDataType) -> OrtStatusPtr ---
	GetMapValueType :: proc(map_type_info: ^OrtMapTypeInfo, type_info: ^^OrtTypeInfo) -> OrtStatusPtr ---
	GetSequenceElementType :: proc(sequence_type_info: ^OrtSequenceTypeInfo, type_info: ^^OrtTypeInfo) -> OrtStatusPtr ---
	ReleaseMapTypeInfo :: proc(input: ^OrtMapTypeInfo) ---
	ReleaseSequenceTypeInfo :: proc(input: ^OrtSequenceTypeInfo) ---
	SessionEndProfiling :: proc(session: ^OrtSession, allocator: ^OrtAllocator, out: [^]cstring) -> OrtStatusPtr ---
	SessionGetModelMetadata :: proc(session: ^OrtSession, out: ^^OrtModelMetadata) -> OrtStatusPtr ---
	ModelMetadataGetProducerName :: proc(model_metadata: ^OrtModelMetadata, allocator: ^OrtAllocator, value: [^]cstring) -> OrtStatusPtr ---
	ModelMetadataGetGraphName :: proc(model_metadata: ^OrtModelMetadata, allocator: ^OrtAllocator, value: [^]cstring) -> OrtStatusPtr ---
	ModelMetadataGetDomain :: proc(model_metadata: ^OrtModelMetadata, allocator: ^OrtAllocator, value: [^]cstring) -> OrtStatusPtr ---
	ModelMetadataGetDescription :: proc(model_metadata: ^OrtModelMetadata, allocator: ^OrtAllocator, value: [^]cstring) -> OrtStatusPtr ---
	ModelMetadataLookupCustomMetadataMap :: proc(model_metadata: ^OrtModelMetadata, allocator: ^OrtAllocator, key: cstring, value: [^]cstring) -> OrtStatusPtr ---
	ModelMetadataGetVersion :: proc(model_metadata: ^OrtModelMetadata, value: ^c.int64_t) -> OrtStatusPtr ---
	ReleaseModelMetadata :: proc(input: ^OrtModelMetadata) ---
	CreateEnvWithGlobalThreadPools :: proc(log_severity_level: OrtLoggingLevel, logid: cstring, tp_options: ^OrtThreadingOptions, out: ^^OrtEnv) -> OrtStatusPtr ---
	DisablePerSessionThreads :: proc(options: ^OrtSessionOptions) -> OrtStatusPtr ---
	CreateThreadingOptions :: proc(out: ^^OrtThreadingOptions) -> OrtStatusPtr ---
	ReleaseThreadingOptions :: proc(input: ^OrtThreadingOptions) ---
	ModelMetadataGetCustomMetadataMapKeys :: proc(model_metadata: ^OrtModelMetadata, allocator: ^OrtAllocator, keys: ^^^c.char, num_keys: ^c.int64_t) -> OrtStatusPtr ---
	AddFreeDimensionOverrideByName :: proc(options: ^OrtSessionOptions, dim_name: cstring, dim_value: c.int64_t) -> OrtStatusPtr ---
	GetAvailableProviders :: proc(out_ptr: ^^^c.char, provider_length: ^c.int) -> OrtStatusPtr ---
	ReleaseAvailableProviders :: proc(ptr: [^]cstring, providers_length: c.int) -> OrtStatusPtr ---
	GetStringTensorElementLength :: proc(value: ^OrtValue, index: c.size_t, out: ^c.size_t) -> OrtStatusPtr ---
	GetStringTensorElement :: proc(value: ^OrtValue, s_len: c.size_t, index: c.size_t, s: rawptr) -> OrtStatusPtr ---
	FillStringTensorElement :: proc(value: ^OrtValue, s: cstring, index: c.size_t) -> OrtStatusPtr ---
	AddSessionConfigEntry :: proc(options: ^OrtSessionOptions, config_key: cstring, config_value: cstring) -> OrtStatusPtr ---
	CreateAllocator :: proc(session: ^OrtSession, mem_info: ^OrtMemoryInfo, out: ^^OrtAllocator) -> OrtStatusPtr ---
	ReleaseAllocator :: proc(input: ^OrtAllocator) ---
	RunWithBinding :: proc(session: ^OrtSession, run_options: ^OrtRunOptions, binding_ptr: ^OrtIoBinding) -> OrtStatusPtr ---
	CreateIoBinding :: proc(session: ^OrtSession, out: ^^OrtIoBinding) -> OrtStatusPtr ---
	ReleaseIoBinding :: proc(input: ^OrtIoBinding) ---
	BindInput :: proc(binding_ptr: ^OrtIoBinding, name: cstring, val_ptr: ^OrtValue) -> OrtStatusPtr ---
	BindOutput :: proc(binding_ptr: ^OrtIoBinding, name: cstring, val_ptr: ^OrtValue) -> OrtStatusPtr ---
	BindOutputToDevice :: proc(binding_ptr: ^OrtIoBinding, name: cstring, mem_info_ptr: ^OrtMemoryInfo) -> OrtStatusPtr ---
	GetBoundOutputNames :: proc(binding_ptr: ^OrtIoBinding, allocator: ^OrtAllocator, buffer: [^]cstring, lengths: ^^c.size_t, count: ^c.size_t) -> OrtStatusPtr ---
	GetBoundOutputValues :: proc(binding_ptr: ^OrtIoBinding, allocator: ^OrtAllocator, output: ^^^OrtValue, output_count: ^c.size_t) -> OrtStatusPtr ---
	ClearBoundInputs :: proc(binding_ptr: ^OrtIoBinding) ---
	ClearBoundOutputs :: proc(binding_ptr: ^OrtIoBinding) ---
	TensorAt :: proc(value: ^OrtValue, location_values: ^c.int64_t, location_values_count: c.size_t, out: ^rawptr) -> OrtStatusPtr ---
	CreateAndRegisterAllocator :: proc(env: ^OrtEnv, mem_info: ^OrtMemoryInfo, arena_cfg: ^OrtArenaCfg) -> OrtStatusPtr ---
	SetLanguageProjection :: proc(ort_env: ^OrtEnv, projection: OrtLanguageProjection) -> OrtStatusPtr ---
	SessionGetProfilingStartTimeNs :: proc(session: ^OrtSession, out: ^c.uint64_t) -> OrtStatusPtr ---
	SetGlobalIntraOpNumThreads :: proc(tp_options: ^OrtThreadingOptions, intra_op_num_threads: c.int) -> OrtStatusPtr ---
	SetGlobalInterOpNumThreads :: proc(tp_options: ^OrtThreadingOptions, inter_op_num_threads: c.int) -> OrtStatusPtr ---
	SetGlobalSpinControl :: proc(tp_options: ^OrtThreadingOptions, allow_spinning: c.int) -> OrtStatusPtr ---
	AddInitializer :: proc(options: ^OrtSessionOptions, name: cstring, val: ^OrtValue) -> OrtStatusPtr ---
	CreateEnvWithCustomLoggerAndGlobalThreadPools :: proc(logging_function: OrtLoggingFunction, logger_param: rawptr, log_severity_level: OrtLoggingLevel, logid: cstring, tp_options: ^OrtThreadingOptions, out: ^^OrtEnv) -> OrtStatusPtr ---
	SessionOptionsAppendExecutionProvider_CUDA :: proc(options: ^OrtSessionOptions, cuda_options: ^OrtCUDAProviderOptions) -> OrtStatusPtr ---
	SessionOptionsAppendExecutionProvider_ROCM :: proc(options: ^OrtSessionOptions, rocm_options: ^OrtROCMProviderOptions) -> OrtStatusPtr ---
	SessionOptionsAppendExecutionProvider_OpenVINO :: proc(options: ^OrtSessionOptions, provider_options: ^OrtOpenVINOProviderOptions) -> OrtStatusPtr ---
	SetGlobalDenormalAsZero :: proc(tp_options: ^OrtThreadingOptions) -> OrtStatusPtr ---
	CreateArenaCfg :: proc(max_mem: c.size_t, arena_extend_strategy: c.int, initial_chunk_size_bytes: c.int, max_dead_bytes_per_chunk: c.int, out: ^^OrtArenaCfg) -> OrtStatusPtr ---
	ReleaseArenaCfg :: proc(input: ^OrtArenaCfg) ---
	ModelMetadataGetGraphDescription :: proc(model_metadata: ^OrtModelMetadata, allocator: ^OrtAllocator, value: [^]cstring) -> OrtStatusPtr ---
	SessionOptionsAppendExecutionProvider_TensorRT :: proc(options: ^OrtSessionOptions, tensorrt_options: ^OrtTensorRTProviderOptions) -> OrtStatusPtr ---
	SetCurrentGpuDeviceId :: proc(device_id: c.int) -> OrtStatusPtr ---
	GetCurrentGpuDeviceId :: proc(device_id: ^c.int) -> OrtStatusPtr ---
	KernelInfoGetAttributeArray_float :: proc(info: ^OrtKernelInfo, name: cstring, out: ^c.float, size: ^c.size_t) -> OrtStatusPtr ---
	KernelInfoGetAttributeArray_int64 :: proc(info: ^OrtKernelInfo, name: cstring, out: ^c.int64_t, size: ^c.size_t) -> OrtStatusPtr ---
	CreateArenaCfgV2 :: proc(arena_config_keys: [^]cstring, arena_config_values: ^c.size_t, num_keys: c.size_t, out: ^^OrtArenaCfg) -> OrtStatusPtr ---
	AddRunConfigEntry :: proc(options: ^OrtRunOptions, config_key: cstring, config_value: cstring) -> OrtStatusPtr ---
	CreatePrepackedWeightsContainer :: proc(out: ^^OrtPrepackedWeightsContainer) -> OrtStatusPtr ---
	ReleasePrepackedWeightsContainer :: proc(input: ^OrtPrepackedWeightsContainer) ---
	CreateSessionWithPrepackedWeightsContainer :: proc(env: ^OrtEnv, model_path: cstring, options: ^OrtSessionOptions, prepacked_weights_container: ^OrtPrepackedWeightsContainer, out: ^^OrtSession) -> OrtStatusPtr ---
	CreateSessionFromArrayWithPrepackedWeightsContainer :: proc(env: ^OrtEnv, model_data: rawptr, model_data_length: c.size_t, options: ^OrtSessionOptions, prepacked_weights_container: ^OrtPrepackedWeightsContainer, out: ^^OrtSession) -> OrtStatusPtr ---
	SessionOptionsAppendExecutionProvider_TensorRT_V2 :: proc(options: ^OrtSessionOptions, tensorrt_options: ^OrtTensorRTProviderOptionsV2) -> OrtStatusPtr ---
	CreateTensorRTProviderOptions :: proc(out: ^^OrtTensorRTProviderOptionsV2) -> OrtStatusPtr ---
	UpdateTensorRTProviderOptions :: proc(tensorrt_options: ^OrtTensorRTProviderOptionsV2, provider_options_keys: [^]cstring, provider_options_values: [^]cstring, num_keys: c.size_t) -> OrtStatusPtr ---
	GetTensorRTProviderOptionsAsString :: proc(tensorrt_options: ^OrtTensorRTProviderOptionsV2, allocator: ^OrtAllocator, ptr: [^]cstring) -> OrtStatusPtr ---
	ReleaseTensorRTProviderOptions :: proc(input: ^OrtTensorRTProviderOptionsV2) ---
	EnableOrtCustomOps :: proc(options: ^OrtSessionOptions) -> OrtStatusPtr ---
	RegisterAllocator :: proc(env: ^OrtEnv, allocator: ^OrtAllocator) -> OrtStatusPtr ---
	UnregisterAllocator :: proc(env: ^OrtEnv, mem_info: ^OrtMemoryInfo) -> OrtStatusPtr ---
	IsSparseTensor :: proc(value: ^OrtValue, out: ^c.int) -> OrtStatusPtr ---
	CreateSparseTensorAsOrtValue :: proc(allocator: ^OrtAllocator, dense_shape: ^c.int64_t, dense_shape_len: c.size_t, type: ONNXTensorElementDataType, out: ^^OrtValue) -> OrtStatusPtr ---
	FillSparseTensorCoo :: proc(ort_value: ^OrtValue, data_mem_info: ^OrtMemoryInfo, values_shape: ^c.int64_t, values_shape_len: c.size_t, values: rawptr, indices_data: ^c.int64_t, indices_num: c.size_t) -> OrtStatusPtr ---
	FillSparseTensorCsr :: proc(ort_value: ^OrtValue, data_mem_info: ^OrtMemoryInfo, values_shape: ^c.int64_t, values_shape_len: c.size_t, values: rawptr, inner_indices_data: ^c.int64_t, inner_indices_num: c.size_t, outer_indices_data: ^c.int64_t, outer_indices_num: c.size_t) -> OrtStatusPtr ---
	FillSparseTensorBlockSparse :: proc(ort_value: ^OrtValue, data_mem_info: ^OrtMemoryInfo, values_shape: ^c.int64_t, values_shape_len: c.size_t, values: rawptr, indices_shape_data: ^c.int64_t, indices_shape_len: c.size_t, indices_data: ^c.int32_t) -> OrtStatusPtr ---
	CreateSparseTensorWithValuesAsOrtValue :: proc(info: ^OrtMemoryInfo, p_data: rawptr, dense_shape: ^c.int64_t, dense_shape_len: c.size_t, values_shape: ^c.int64_t, values_shape_len: c.size_t, type: ONNXTensorElementDataType, out: ^^OrtValue) -> OrtStatusPtr ---
	UseCooIndices :: proc(ort_value: ^OrtValue, indices_data: ^c.int64_t, indices_num: c.size_t) -> OrtStatusPtr ---
	UseCsrIndices :: proc(ort_value: ^OrtValue, inner_data: ^c.int64_t, inner_num: c.size_t, outer_data: ^c.int64_t, outer_num: c.size_t) -> OrtStatusPtr ---
	UseBlockSparseIndices :: proc(ort_value: ^OrtValue, indices_shape: ^c.int64_t, indices_shape_len: c.size_t, indices_data: ^c.int32_t) -> OrtStatusPtr ---
	GetSparseTensorFormat :: proc(ort_value: ^OrtValue, out: ^OrtSparseFormat) -> OrtStatusPtr ---
	GetSparseTensorValuesTypeAndShape :: proc(ort_value: ^OrtValue, out: ^^OrtTensorTypeAndShapeInfo) -> OrtStatusPtr ---
	GetSparseTensorValues :: proc(ort_value: ^OrtValue, out: ^rawptr) -> OrtStatusPtr ---
	GetSparseTensorIndicesTypeShape :: proc(ort_value: ^OrtValue, indices_format: OrtSparseIndicesFormat, out: ^^OrtTensorTypeAndShapeInfo) -> OrtStatusPtr ---
	GetSparseTensorIndices :: proc(ort_value: ^OrtValue, indices_format: OrtSparseIndicesFormat, num_indices: ^c.size_t, indices: ^rawptr) -> OrtStatusPtr ---
	HasValue :: proc(value: ^OrtValue, out: ^c.int) -> OrtStatusPtr ---
	KernelContext_GetGPUComputeStream :: proc(context_: ^OrtKernelContext, out: ^rawptr) -> OrtStatusPtr ---
	GetTensorMemoryInfo :: proc(value: ^OrtValue, mem_info: ^^OrtMemoryInfo) -> OrtStatusPtr ---
	GetExecutionProviderApi :: proc(provider_name: cstring, version: c.uint32_t, provider_api: ^rawptr) -> OrtStatusPtr ---
	SessionOptionsSetCustomCreateThreadFn :: proc(options: ^OrtSessionOptions, ort_custom_create_thread_fn: OrtCustomCreateThreadFn) -> OrtStatusPtr ---
	SessionOptionsSetCustomThreadCreationOptions :: proc(options: ^OrtSessionOptions, ort_custom_thread_creation_options: rawptr) -> OrtStatusPtr ---
	SessionOptionsSetCustomJoinThreadFn :: proc(options: ^OrtSessionOptions, ort_custom_join_thread_fn: OrtCustomJoinThreadFn) -> OrtStatusPtr ---
	SetGlobalCustomCreateThreadFn :: proc(tp_options: ^OrtThreadingOptions, ort_custom_create_thread_fn: OrtCustomCreateThreadFn) -> OrtStatusPtr ---
	SetGlobalCustomThreadCreationOptions :: proc(tp_options: ^OrtThreadingOptions, ort_custom_thread_creation_options: rawptr) -> OrtStatusPtr ---
	SetGlobalCustomJoinThreadFn :: proc(tp_options: ^OrtThreadingOptions, ort_custom_join_thread_fn: OrtCustomJoinThreadFn) -> OrtStatusPtr ---
	SynchronizeBoundInputs :: proc(binding_ptr: ^OrtIoBinding) -> OrtStatusPtr ---
	SynchronizeBoundOutputs :: proc(binding_ptr: ^OrtIoBinding) -> OrtStatusPtr ---
	SessionOptionsAppendExecutionProvider_CUDA_V2 :: proc(options: ^OrtSessionOptions, cuda_options: ^OrtCUDAProviderOptionsV2) -> OrtStatusPtr ---
	CreateCUDAProviderOptions :: proc(out: ^^OrtCUDAProviderOptionsV2) -> OrtStatusPtr ---
	UpdateCUDAProviderOptions :: proc(cuda_options: ^OrtCUDAProviderOptionsV2, provider_options_keys: [^]cstring, provider_options_values: [^]cstring, num_keys: c.size_t) -> OrtStatusPtr ---
	GetCUDAProviderOptionsAsString :: proc(cuda_options: ^OrtCUDAProviderOptionsV2, allocator: ^OrtAllocator, ptr: [^]cstring) -> OrtStatusPtr ---
	ReleaseCUDAProviderOptions :: proc(input: ^OrtCUDAProviderOptionsV2) ---
	SessionOptionsAppendExecutionProvider_MIGraphX :: proc(options: ^OrtSessionOptions, migraphx_options: ^OrtMIGraphXProviderOptions) -> OrtStatusPtr ---
	AddExternalInitializers :: proc(options: ^OrtSessionOptions, initializer_names: [^]cstring, initializers: ^^OrtValue, initializers_num: c.size_t) -> OrtStatusPtr ---
	CreateOpAttr :: proc(name: cstring, data: rawptr, len: c.int, type: OrtOpAttrType, op_attr: ^^OrtOpAttr) -> OrtStatusPtr ---
	ReleaseOpAttr :: proc(input: ^OrtOpAttr) ---
	CreateOp :: proc(info: ^OrtKernelInfo, op_name: cstring, domain: cstring, version: c.int, type_constraint_names: [^]cstring, type_constraint_values: ^ONNXTensorElementDataType, type_constraint_count: c.int, attr_values: ^^OrtOpAttr, attr_count: c.int, input_count: c.int, output_count: c.int, ort_op: ^^OrtOp) -> OrtStatusPtr ---
	InvokeOp :: proc(context_: ^OrtKernelContext, ort_op: ^OrtOp, input_values: ^^OrtValue, input_count: c.int, output_values: ^^OrtValue, output_count: c.int) -> OrtStatusPtr ---
	ReleaseOp :: proc(input: ^OrtOp) ---
	SessionOptionsAppendExecutionProvider :: proc(options: ^OrtSessionOptions, provider_name: cstring, provider_options_keys: [^]cstring, provider_options_values: [^]cstring, num_keys: c.size_t) -> OrtStatusPtr ---
	CopyKernelInfo :: proc(info: ^OrtKernelInfo, info_copy: ^^OrtKernelInfo) -> OrtStatusPtr ---
	ReleaseKernelInfo :: proc(input: ^OrtKernelInfo) ---
	GetTrainingApi :: proc(version: c.uint32_t) -> ^OrtTrainingApi ---
	SessionOptionsAppendExecutionProvider_CANN :: proc(options: ^OrtSessionOptions, cann_options: ^OrtCANNProviderOptions) -> OrtStatusPtr ---
	CreateCANNProviderOptions :: proc(out: ^^OrtCANNProviderOptions) -> OrtStatusPtr ---
	UpdateCANNProviderOptions :: proc(cann_options: ^OrtCANNProviderOptions, provider_options_keys: [^]cstring, provider_options_values: [^]cstring, num_keys: c.size_t) -> OrtStatusPtr ---
	GetCANNProviderOptionsAsString :: proc(cann_options: ^OrtCANNProviderOptions, allocator: ^OrtAllocator, ptr: [^]cstring) -> OrtStatusPtr ---
	ReleaseCANNProviderOptions :: proc(input: ^OrtCANNProviderOptions) ---
	MemoryInfoGetDeviceType :: proc(ptr: ^OrtMemoryInfo, out: ^OrtMemoryInfoDeviceType) ---
	UpdateEnvWithCustomLogLevel :: proc(ort_env: ^OrtEnv, log_severity_level: OrtLoggingLevel) -> OrtStatusPtr ---
	SetGlobalIntraOpThreadAffinity :: proc(tp_options: ^OrtThreadingOptions, affinity_string: cstring) -> OrtStatusPtr ---
	RegisterCustomOpsLibrary_V2 :: proc(options: ^OrtSessionOptions, library_name: cstring) -> OrtStatusPtr ---
	RegisterCustomOpsUsingFunction :: proc(options: ^OrtSessionOptions, registration_func_name: cstring) -> OrtStatusPtr ---
	KernelInfo_GetInputCount :: proc(info: ^OrtKernelInfo, out: ^c.size_t) -> OrtStatusPtr ---
	KernelInfo_GetOutputCount :: proc(info: ^OrtKernelInfo, out: ^c.size_t) -> OrtStatusPtr ---
	KernelInfo_GetInputName :: proc(info: ^OrtKernelInfo, index: c.size_t, out: cstring, size: ^c.size_t) -> OrtStatusPtr ---
	KernelInfo_GetOutputName :: proc(info: ^OrtKernelInfo, index: c.size_t, out: cstring, size: ^c.size_t) -> OrtStatusPtr ---
	KernelInfo_GetInputTypeInfo :: proc(info: ^OrtKernelInfo, index: c.size_t, type_info: ^^OrtTypeInfo) -> OrtStatusPtr ---
	KernelInfo_GetOutputTypeInfo :: proc(info: ^OrtKernelInfo, index: c.size_t, type_info: ^^OrtTypeInfo) -> OrtStatusPtr ---
	KernelInfoGetAttribute_tensor :: proc(info: ^OrtKernelInfo, name: cstring, allocator: ^OrtAllocator, out: ^^OrtValue) -> OrtStatusPtr ---
	HasSessionConfigEntry :: proc(options: ^OrtSessionOptions, config_key: cstring, out: ^c.int) -> OrtStatusPtr ---
	GetSessionConfigEntry :: proc(options: ^OrtSessionOptions, config_key: cstring, config_value: cstring, size: ^c.size_t) -> OrtStatusPtr ---
	SessionOptionsAppendExecutionProvider_Dnnl :: proc(options: ^OrtSessionOptions, dnnl_options: ^OrtDnnlProviderOptions) -> OrtStatusPtr ---
	CreateDnnlProviderOptions :: proc(out: ^^OrtDnnlProviderOptions) -> OrtStatusPtr ---
	UpdateDnnlProviderOptions :: proc(dnnl_options: ^OrtDnnlProviderOptions, provider_options_keys: [^]cstring, provider_options_values: [^]cstring, num_keys: c.size_t) -> OrtStatusPtr ---
	GetDnnlProviderOptionsAsString :: proc(dnnl_options: ^OrtDnnlProviderOptions, allocator: ^OrtAllocator, ptr: [^]cstring) -> OrtStatusPtr ---
	ReleaseDnnlProviderOptions :: proc(input: ^OrtDnnlProviderOptions) ---
	KernelInfo_GetNodeName :: proc(info: ^OrtKernelInfo, out: cstring, size: ^c.size_t) -> OrtStatusPtr ---
	KernelInfo_GetLogger :: proc(info: ^OrtKernelInfo, logger: ^^OrtLogger) -> OrtStatusPtr ---
	KernelContext_GetLogger :: proc(context_: ^OrtKernelContext, logger: ^^OrtLogger) -> OrtStatusPtr ---
	Logger_LogMessage :: proc(logger: ^OrtLogger, log_severity_level: OrtLoggingLevel, message: cstring, file_path: cstring, line_number: c.int, func_name: cstring) -> OrtStatusPtr ---
	Logger_GetLoggingSeverityLevel :: proc(logger: ^OrtLogger, out: ^OrtLoggingLevel) -> OrtStatusPtr ---
	KernelInfoGetConstantInput_tensor :: proc(info: ^OrtKernelInfo, index: c.size_t, is_constant: ^c.int, out: ^^OrtValue) -> OrtStatusPtr ---
	CastTypeInfoToOptionalTypeInfo :: proc(type_info: ^OrtTypeInfo, out: ^^OrtOptionalTypeInfo) -> OrtStatusPtr ---
	GetOptionalContainedTypeInfo :: proc(optional_type_info: ^OrtOptionalTypeInfo, out: ^^OrtTypeInfo) -> OrtStatusPtr ---
	GetResizedStringTensorElementBuffer :: proc(value: ^OrtValue, index: c.size_t, length_in_bytes: c.size_t, buffer: [^]cstring) -> OrtStatusPtr ---
	KernelContext_GetAllocator :: proc(context_: ^OrtKernelContext, mem_info: ^OrtMemoryInfo, out: ^^OrtAllocator) -> OrtStatusPtr ---
	GetBuildInfoString :: proc() -> cstring ---
	CreateROCMProviderOptions :: proc(out: ^^OrtROCMProviderOptions) -> OrtStatusPtr ---
	UpdateROCMProviderOptions :: proc(rocm_options: ^OrtROCMProviderOptions, provider_options_keys: [^]cstring, provider_options_values: [^]cstring, num_keys: c.size_t) -> OrtStatusPtr ---
	GetROCMProviderOptionsAsString :: proc(rocm_options: ^OrtROCMProviderOptions, allocator: ^OrtAllocator, ptr: [^]cstring) -> OrtStatusPtr ---
	ReleaseROCMProviderOptions :: proc(input: ^OrtROCMProviderOptions) ---
	CreateAndRegisterAllocatorV2 :: proc(env: ^OrtEnv, provider_type: cstring, mem_info: ^OrtMemoryInfo, arena_cfg: ^OrtArenaCfg, provider_options_keys: [^]cstring, provider_options_values: [^]cstring, num_keys: c.size_t) -> OrtStatusPtr ---
	RunAsync :: proc(session: ^OrtSession, run_options: ^OrtRunOptions, input_names: [^]cstring, input: ^^OrtValue, input_len: c.size_t, output_names: [^]cstring, output_names_len: c.size_t, output: ^^OrtValue, run_async_callback: RunAsyncCallbackFn, user_data: rawptr) -> OrtStatusPtr ---
	UpdateTensorRTProviderOptionsWithValue :: proc(tensorrt_options: ^OrtTensorRTProviderOptionsV2, key: cstring, value: rawptr) -> OrtStatusPtr ---
	GetTensorRTProviderOptionsByName :: proc(tensorrt_options: ^OrtTensorRTProviderOptionsV2, key: cstring, ptr: ^rawptr) -> OrtStatusPtr ---
	UpdateCUDAProviderOptionsWithValue :: proc(cuda_options: ^OrtCUDAProviderOptionsV2, key: cstring, value: rawptr) -> OrtStatusPtr ---
	GetCUDAProviderOptionsByName :: proc(cuda_options: ^OrtCUDAProviderOptionsV2, key: cstring, ptr: ^rawptr) -> OrtStatusPtr ---
	KernelContext_GetResource :: proc(context_: ^OrtKernelContext, resouce_version: c.int, resource_id: c.int, resource: ^rawptr) -> OrtStatusPtr ---
	SetUserLoggingFunction :: proc(options: ^OrtSessionOptions, user_logging_function: OrtLoggingFunction, user_logging_param: rawptr) -> OrtStatusPtr ---
	ShapeInferContext_GetInputCount :: proc(context_: ^OrtShapeInferContext, out: ^c.size_t) -> OrtStatusPtr ---
	ShapeInferContext_GetInputTypeShape :: proc(context_: ^OrtShapeInferContext, index: c.size_t, info: ^^OrtTensorTypeAndShapeInfo) -> OrtStatusPtr ---
	ShapeInferContext_GetAttribute :: proc(context_: ^OrtShapeInferContext, attr_name: cstring, attr: ^^OrtOpAttr) -> OrtStatusPtr ---
	ShapeInferContext_SetOutputTypeShape :: proc(context_: ^OrtShapeInferContext, index: c.size_t, info: ^OrtTensorTypeAndShapeInfo) -> OrtStatusPtr ---
	SetSymbolicDimensions :: proc(info: ^OrtTensorTypeAndShapeInfo, dim_params: [^]cstring, dim_params_length: c.size_t) -> OrtStatusPtr ---
	ReadOpAttr :: proc(op_attr: ^OrtOpAttr, type: OrtOpAttrType, data: rawptr, len: c.size_t, out: ^c.size_t) -> OrtStatusPtr ---
	SetDeterministicCompute :: proc(options: ^OrtSessionOptions, value: c.bool) -> OrtStatusPtr ---
	KernelContext_ParallelFor :: proc(context_: ^OrtKernelContext, fn: proc(_: rawptr, _: c.size_t), total: c.size_t, num_batch: c.size_t, usr_data: rawptr) -> OrtStatusPtr ---
	SessionOptionsAppendExecutionProvider_OpenVINO_V2 :: proc(options: ^OrtSessionOptions, provider_options_keys: [^]cstring, provider_options_values: [^]cstring, num_keys: c.size_t) -> OrtStatusPtr ---
	CreateKernel :: proc(op: ^OrtCustomOp, api: ^OrtApi, info: ^OrtKernelInfo) -> rawptr ---
	GetName :: proc(op: ^OrtCustomOp) -> cstring ---
	GetExecutionProviderType :: proc(op: ^OrtCustomOp) -> cstring ---
	GetInputType :: proc(op: ^OrtCustomOp, index: c.size_t) -> ONNXTensorElementDataType ---
	GetInputTypeCount :: proc(op: ^OrtCustomOp) -> c.size_t ---
	GetOutputType :: proc(op: ^OrtCustomOp, index: c.size_t) -> ONNXTensorElementDataType ---
	GetOutputTypeCount :: proc(op: ^OrtCustomOp) -> c.size_t ---
	KernelCompute :: proc(op_kernel: rawptr, context_: ^OrtKernelContext) ---
	KernelDestroy :: proc(op_kernel: rawptr) ---
	GetInputCharacteristic :: proc(op: ^OrtCustomOp, index: c.size_t) -> OrtCustomOpInputOutputCharacteristic ---
	GetOutputCharacteristic :: proc(op: ^OrtCustomOp, index: c.size_t) -> OrtCustomOpInputOutputCharacteristic ---
	GetInputMemoryType :: proc(op: ^OrtCustomOp, index: c.size_t) -> OrtMemType ---
	GetVariadicInputMinArity :: proc(op: ^OrtCustomOp) -> c.int ---
	GetVariadicInputHomogeneity :: proc(op: ^OrtCustomOp) -> c.int ---
	GetVariadicOutputMinArity :: proc(op: ^OrtCustomOp) -> c.int ---
	GetVariadicOutputHomogeneity :: proc(op: ^OrtCustomOp) -> c.int ---
	CreateKernelV2 :: proc(op: ^OrtCustomOp, api: ^OrtApi, info: ^OrtKernelInfo, kernel: ^rawptr) -> OrtStatusPtr ---
	KernelComputeV2 :: proc(op_kernel: rawptr, context_: ^OrtKernelContext) -> OrtStatusPtr ---
	InferOutputShapeFn :: proc(op: ^OrtCustomOp, _: ^OrtShapeInferContext) -> OrtStatusPtr ---
	GetStartVersion :: proc(op: ^OrtCustomOp) -> c.int ---
	GetEndVersion :: proc(op: ^OrtCustomOp) -> c.int ---
}

// End Functions
