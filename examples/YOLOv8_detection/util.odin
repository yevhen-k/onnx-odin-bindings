package yolov8_inference

import "core:c"
import "core:fmt"
import "core:log"
import "core:os"

check_ort_status :: proc(ort: ^OrtApi, status: OrtStatusPtr) {
	if status != nil {
		msg: cstring = ort.GetErrorMessage(status)
		fmt.eprintln(msg)
		ort.ReleaseStatus(status)
		os.exit(1)
	}
}

get_available_providers :: proc(ort: ^OrtApi) -> (providers: [^]cstring, providers_count: c.int) {
	ort.GetAvailableProviders(cast(^^^c.char)(&providers), &providers_count)

	log.debug("Available providers:")
	for i: c.int = 0; i < (providers_count); i += 1 {
		log.debugf("\t%d) %s", i, providers[i])
	}
	/*
    >>> 0) TensorrtExecutionProvider
    >>> 1) CUDAExecutionProvider
    >>> 2) CPUExecutionProvider
    */
	return
}

is_cuda_available :: proc(providers: [^]cstring, providers_count: c.int) -> bool {
	cuda_available: bool
	for i: c.int = 0; i < providers_count; i += 1 {
		if providers[i] == "CUDAExecutionProvider" {
			cuda_available = true
			break
		}
	}
	return cuda_available
}

print_model_layers :: proc(ort: ^OrtApi, session: ^OrtSession) {
	num_input_nodes: c.size_t
	allocator: ^OrtAllocator
	status := ort.GetAllocatorWithDefaultOptions(&allocator)
	check_ort_status(ort, status)

	//// print number of model input nodes
	status = ort.SessionGetInputCount(session, &num_input_nodes)
	check_ort_status(ort, status)

	input_node_names := make([dynamic]cstring, num_input_nodes)
	defer delete(input_node_names)

	input_node_dims := make([dynamic]c.int64_t)
	defer delete(input_node_dims)
	fmt.printfln(">>> Number of inputs = %d", num_input_nodes)

	// iterate over all input nodes
	for i: c.size_t = 0; i < num_input_nodes; i += 1 {
		// print input node names
		input_name: cstring
		status = ort.SessionGetInputName(session, i, allocator, &input_name)
		fmt.printfln(">>>\tInput %d : name=%s", i, input_name)
		input_node_names[i] = input_name

		// print input node types
		typeinfo: ^OrtTypeInfo
		status = ort.SessionGetInputTypeInfo(session, i, &typeinfo)
		check_ort_status(ort, status)
		defer ort.ReleaseTypeInfo(typeinfo)
		tensor_info: ^OrtTensorTypeAndShapeInfo
		status = ort.CastTypeInfoToTensorInfo(typeinfo, &tensor_info)
		check_ort_status(ort, status)
		type: ONNXTensorElementDataType
		status = ort.GetTensorElementType(tensor_info, &type)
		check_ort_status(ort, status)
		fmt.printfln(">>>\tInput %d : type=%d", i, type)

		// print input shapes/dims
		num_dims: c.size_t
		status = ort.GetDimensionsCount(tensor_info, &num_dims)
		check_ort_status(ort, status)
		fmt.printfln(">>>\tInput %d : num_dims=%d", i, num_dims)

		resize(&input_node_dims, cast(int)num_dims)
		status = ort.GetDimensions(
			tensor_info,
			cast(^c.int64_t)raw_data(input_node_dims),
			num_dims,
		)
		check_ort_status(ort, status)
		for j: c.size_t = 0; j < num_dims; j += 1 {
			fmt.printfln(">>>\tInput %d : dim %d=%d", i, j, input_node_dims[j])
		}
	}
	//*************************************************************************

	num_output_nodes: c.size_t
	//// print number of model output nodes
	status = ort.SessionGetOutputCount(session, &num_output_nodes)
	check_ort_status(ort, status)

	output_node_names := make([dynamic]cstring, num_output_nodes)
	defer delete(output_node_names)

	output_node_dims := make([dynamic]c.int64_t)
	defer delete(output_node_dims)
	fmt.printfln(">>> Number of output = %d", num_output_nodes)

	// iterate over all output nodes
	for i: c.size_t = 0; i < num_output_nodes; i += 1 {
		// print output node names
		output_name: cstring
		status = ort.SessionGetOutputName(session, i, allocator, &output_name)
		fmt.printfln(">>>\tOutput %d : name=%s", i, output_name)
		output_node_names[i] = output_name

		// print output node types
		typeinfo: ^OrtTypeInfo
		status = ort.SessionGetOutputTypeInfo(session, i, &typeinfo)
		check_ort_status(ort, status)
		defer ort.ReleaseTypeInfo(typeinfo)
		tensor_info: ^OrtTensorTypeAndShapeInfo
		status = ort.CastTypeInfoToTensorInfo(typeinfo, &tensor_info)
		check_ort_status(ort, status)
		type: ONNXTensorElementDataType
		status = ort.GetTensorElementType(tensor_info, &type)
		check_ort_status(ort, status)
		fmt.printfln(">>>\tOutput %d : type=%d", i, type)

		// print output shapes/dims
		num_dims: c.size_t
		status = ort.GetDimensionsCount(tensor_info, &num_dims)
		check_ort_status(ort, status)
		fmt.printfln(">>>\tOutput %d : num_dims=%d", i, num_dims)

		resize(&output_node_dims, cast(int)num_dims)
		status = ort.GetDimensions(
			tensor_info,
			cast(^c.int64_t)raw_data(output_node_dims),
			num_dims,
		)
		check_ort_status(ort, status)
		for j: c.size_t = 0; j < num_dims; j += 1 {
			fmt.printfln(">>>\tOutput %d : dim %d=%d", i, j, output_node_dims[j])
		}
	}
}

run_inference :: proc(
	ort: ^OrtApi,
	session: ^OrtSession,
	data: [^]u8,
	input_tensor_size: c.size_t,
	input_node_dims: [dynamic]c.int64_t,
	input_node_names: [dynamic]cstring,
	output_node_names: [dynamic]cstring,
) -> (
	[^]c.float,
	^OrtValue,
) {

	memory_info: ^OrtMemoryInfo
	status := ort.CreateCpuMemoryInfo(
		OrtAllocatorType.OrtArenaAllocator,
		OrtMemType.OrtMemTypeDefault,
		&memory_info,
	)
	check_ort_status(ort, status)
	defer ort.ReleaseMemoryInfo(memory_info)
	input_tensor: ^OrtValue
	status = ort.CreateTensorWithDataAsOrtValue(
		memory_info,
		cast(rawptr)(data),
		input_tensor_size * size_of(c.float),
		cast(^c.int64_t)raw_data(input_node_dims),
		len(input_node_dims),
		ONNXTensorElementDataType.ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT,
		&input_tensor,
	)
	check_ort_status(ort, status)
	defer ort.ReleaseValue(input_tensor)

	is_tensor: c.int
	status = ort.IsTensor(input_tensor, &is_tensor)
	check_ort_status(ort, status)
	assert(is_tensor == 1, "input_tensor not a tensor")

	// score model & input tensor, get back output tensor
	output_tensor: ^OrtValue
	run_options: ^OrtRunOptions
	status = ort.Run(
		session,
		run_options,
		raw_data(input_node_names),
		&input_tensor,
		len(input_node_names),
		raw_data(output_node_names),
		len(output_node_names),
		&output_tensor,
	)
	// defer ort.ReleaseValue(output_tensor)
	check_ort_status(ort, status)

	status = ort.IsTensor(output_tensor, &is_tensor)
	check_ort_status(ort, status)
	assert(is_tensor == 1, "output_tensor not a tensor")

	// Get pointer to output tensor float values
	prediction: [^]c.float
	status = ort.GetTensorMutableData(output_tensor, cast(^rawptr)&prediction)
	check_ort_status(ort, status)
	return prediction, output_tensor
}

MODEL_WH :: 640
YOLOV8_CLASSES := [?]string {
	"person",
	"bicycle",
	"car",
	"motorcycle",
	"airplane",
	"bus",
	"train",
	"truck",
	"boat",
	"traffic light",
	"fire hydrant",
	"stop sign",
	"parking meter",
	"bench",
	"bird",
	"cat",
	"dog",
	"horse",
	"sheep",
	"cow",
	"elephant",
	"bear",
	"zebra",
	"giraffe",
	"backpack",
	"umbrella",
	"handbag",
	"tie",
	"suitcase",
	"frisbee",
	"skis",
	"snowboard",
	"sports ball",
	"kite",
	"baseball bat",
	"baseball glove",
	"skateboard",
	"surfboard",
	"tennis racket",
	"bottle",
	"wine glass",
	"cup",
	"fork",
	"knife",
	"spoon",
	"bowl",
	"banana",
	"apple",
	"sandwich",
	"orange",
	"broccoli",
	"carrot",
	"hot dog",
	"pizza",
	"donut",
	"cake",
	"chair",
	"couch",
	"potted plant",
	"bed",
	"dining table",
	"toilet",
	"tv",
	"laptop",
	"mouse",
	"remote",
	"keyboard",
	"cell phone",
	"microwave",
	"oven",
	"toaster",
	"sink",
	"refrigerator",
	"book",
	"clock",
	"vase",
	"scissors",
	"teddy bear",
	"hair drier",
	"toothbrush",
}
SCORE_THRESH :: 0.25
NMS_THRESHOLD :: 0.50
