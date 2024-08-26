package explore_env

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
}
