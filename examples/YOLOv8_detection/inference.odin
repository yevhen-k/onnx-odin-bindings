package yolov8_inference

import "core:c"
import "core:c/libc"
import "core:fmt"
import "core:log"
import "core:os"
import "core:strings"

main :: proc() {
	context.logger = log.create_console_logger()

	ort: ^OrtApi
	if ort = OrtGetApiBase().GetApi(ORT_API_VERSION); cast(rawptr)ort == nil {
		fmt.eprintln(">>> Can't initialize OrtApi. Aborting.")
		os.exit(1)
	}
	log.debugf("API (str): %s", OrtGetApiBase().GetVersionString())

	providers, providers_count := get_available_providers(ort)
	defer ort.ReleaseAvailableProviders(providers, providers_count)

	cuda_available := is_cuda_available(providers, providers_count)
	log.debugf("CUDA is available: %t", cuda_available)

	//*************************************************************************
	// initialize  enviroment...one enviroment per process
	// enviroment maintains thread pools and other state info
	env: ^OrtEnv
	status: OrtStatusPtr = ort.CreateEnv(OrtLoggingLevel.ORT_LOGGING_LEVEL_WARNING, "test", &env)
	check_ort_status(ort, status)
	defer ort.ReleaseEnv(env)

	// initialize session options if needed
	session_options: ^OrtSessionOptions
	status = ort.CreateSessionOptions(&session_options)
	check_ort_status(ort, status)
	defer ort.ReleaseSessionOptions(session_options)

	status = ort.SetIntraOpNumThreads(session_options, 1)
	check_ort_status(ort, status)

	// Sets graph optimization level
	status = ort.SetSessionGraphOptimizationLevel(
		session_options,
		GraphOptimizationLevel.ORT_ENABLE_BASIC,
	)
	check_ort_status(ort, status)

	// Enable CUDA acceleration
	if cuda_available {
		log.debugf("Setting up CUDA...")
		status = OrtSessionOptionsAppendExecutionProvider_CUDA(session_options, 0)
		check_ort_status(ort, status)
	}

	//*************************************************************************
	// create session and load model into memory
	session: ^OrtSession
	model_path :: "./model/yolov8s.onnx"
	status = ort.CreateSession(env, model_path, session_options, &session)
	check_ort_status(ort, status)
	defer ort.ReleaseSession(session)

	//*************************************************************************
	// print model input layer (node names, types, shape etc.)
	print_model_layers(ort, session)
	// Result should be
	// >>> Number of inputs = 1
	// >>>     Input 0 : name=images
	// >>>     Input 0 : type=1
	// >>>     Input 0 : num_dims=4
	// >>>     Input 0 : dim 0=1
	// >>>     Input 0 : dim 1=3
	// >>>     Input 0 : dim 2=640
	// >>>     Input 0 : dim 3=640
	// >>> Number of output = 2
	// >>>     Output 0 : name=output0
	// >>>     Output 0 : type=1
	// >>>     Output 0 : num_dims=3
	// >>>     Output 0 : dim 0=1
	// >>>     Output 0 : dim 1=84
	// >>>     Output 0 : dim 2=8400

	//*************************************************************************
	// Set node names to pass tensor into and get tensor from
	input_tensor_size :: MODEL_WH * MODEL_WH * 3

	input_node_dims := make([dynamic]c.int64_t)
	defer delete(input_node_dims)
	append(&input_node_dims, 1, 3, MODEL_WH, MODEL_WH)

	input_node_names := make([dynamic]cstring)
	defer delete(input_node_names)
	append(&input_node_names, "images")

	output_node_names := make([dynamic]cstring)
	defer delete(output_node_names)
	append(&output_node_names, "output0")

	//*************************************************************************
	// Read image
	image_path :: "./images/test.jpg"

	image := cv_image_read(
		strings.unsafe_string_to_cstring(image_path),
		ImageReadModes.IMREAD_COLOR,
	)
	if cv_mat_isempty(image) {
		fmt.eprintfln("Error: can't find or open an image: %s", image_path)
		os.exit(1)
	}
	defer cv_free_mem(image)

	// Check image properties
	mat_view := cv_get_mat_view(image)
	log.debugf(">>> Image: %v", mat_view)
	scale_factor_hight := f32(mat_view.rows) / MODEL_WH
	scale_factor_width := f32(mat_view.cols) / MODEL_WH
	// MatView{rows = 1080, cols = 810, channels = 3, type = 16, dims = 2, data = 0x5817000}
	// type=16, which is CV_8U|C3

	// Image Preprocessing
	scalefactor :: 1.0 / 255.0
	size :: Size {
		dim1 = MODEL_WH,
		dim2 = MODEL_WH,
	}
	mean :: Scalar {
		dim1 = 0,
		dim2 = 0,
		dim3 = 0,
	}
	swapRGB :: true
	crop :: false
	ddepth :: DataTypes.CV_32F
	batch := cv_blob_from_image(image, scalefactor, size, mean, swapRGB, crop, ddepth)
	defer cv_free_mem(batch)
	mat_view = cv_get_mat_view(batch)
	log.debugf(">>> Batch: %v", mat_view)

	// **********************************************
	// Assume, batch with images is ready.
	// create input tensor object from data values
	prediction, output_tensor := run_inference(
		ort,
		session,
		mat_view.data,
		input_tensor_size,
		input_node_dims,
		input_node_names,
		output_node_names,
	)
	// output tensor holds data need to be released at the end of the scope
	defer ort.ReleaseValue(output_tensor)

	// ******************************************************
	// Parse predictions
	// Output tensor has [1x84x8400] size:
	// X Y W H ClassScores
	// 1 1 1 1    80

	yolo_output_shape :: Shape3i{1, 84, 8400} // we know that batch size is 1
	batch_detections := cv_parse_yolo_output(
		prediction,
		yolo_output_shape,
		SCORE_THRESH,
		NMS_THRESHOLD,
	)
	defer {
		for batch in 0 ..< batch_detections.batch_size {
			libc.free(rawptr(batch_detections.detections[batch].detection))
			libc.free(rawptr(batch_detections.detections))
		}
	}

	// iterate over batch predictions and get/render results
	for batch in 0 ..< batch_detections.batch_size {
		for i in 0 ..< batch_detections.detections[batch].count {
			detection := batch_detections.detections[batch].detection[i]
			log.debugf("%v", detection)
			class_name := strings.clone_to_cstring(YOLOV8_CLASSES[int(detection.class_id)])
			// scale back detection to the original image size
			detection.x = detection.x * scale_factor_width
			detection.w = detection.w * scale_factor_width
			detection.y = detection.y * scale_factor_hight
			detection.h = detection.h * scale_factor_hight
			cv_render_detection(image, detection, class_name)
		}
	}
	// show rendered results and save image
	window_name :: "Image"
	cv_named_window(window_name)
	defer cv_destroy_window(window_name)
	cv_image_show(window_name, image)
	cv_wait_key(0)

	cv_image_write("./images/test-result.jpg", image)

	log.debugf("DONE!")
}
