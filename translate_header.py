import json
import io
from typing import List, Tuple, Dict
import argparse


type_table: Dict[str, str] = {
    "void": "",
    "void*": "rawptr",
    "char": "c.char",
    "char*": "cstring",
    "int": "c.int",
    "size_t": "c.size_t",
    "uint32_t": "c.uint32_t",
    "int64_t": "c.int64_t",
    "unsigned char": "c.char",
    "size_t*": "^c.size_t",
    "int*": "^c.int",
    "int64_t*": "^c.int64_t",
    "float*": "^c.float",
    "size_t**": "^^c.size_t",
    "uint64_t*": "^c.uint64_t",
    "int32_t*": "^c.int32_t",
    "_Bool": "c.bool",
    "OrtLoggingLevel": "OrtLoggingLevel",
    "OrtThreadWorkerFn": "OrtThreadWorkerFn",
    "OrtCustomThreadHandle": "OrtCustomThreadHandle",
    "OrtSessionOptions*": "^OrtSessionOptions",
    "OrtApiBase*": "^OrtApiBase",
    "OrtStatus*": "^OrtStatus",
    "OrtValue**": "^^OrtValue",
    "OrtStatusPtr": "OrtStatusPtr",
    "OrtCudnnConvAlgoSearch": "OrtCudnnConvAlgoSearch",
    "OrtArenaCfg*": "^OrtArenaCfg",
    "OrtAllocator*": "^OrtAllocator",
    "OrtMemoryInfo*": "^OrtMemoryInfo",
    "OrtApi*": "^OrtApi",
    "OrtErrorCode": "OrtErrorCode",
    "OrtLoggingFunction": "OrtLoggingFunction",
    "OrtEnv*": "^OrtEnv",
    "OrtSession**": "^^OrtSession",
    "OrtSession*": "^OrtSession",
    "OrtRunOptions*": "^OrtRunOptions",
    "char**": "[^]cstring",
    "ExecutionMode": "ExecutionMode",
    "GraphOptimizationLevel": "GraphOptimizationLevel",
    "OrtEnv**": "^^OrtEnv",
    "OrtSessionOptions**": "^^OrtSessionOptions",
    "OrtCustomOpDomain**": "^^OrtCustomOpDomain",
    "OrtCustomOpDomain*": "^OrtCustomOpDomain",
    "OrtCustomOp*": "^OrtCustomOp",
    "OrtTypeInfo**": "^^OrtTypeInfo",
    "OrtRunOptions**": "^^OrtRunOptions",
    "ONNXTensorElementDataType": "ONNXTensorElementDataType",
    "OrtValue*": "^OrtValue",
    "OrtTypeInfo*": "^OrtTypeInfo",
    "OrtTensorTypeAndShapeInfo**": "^^OrtTensorTypeAndShapeInfo",
    "ONNXType*": "^ONNXType",
    "OrtTensorTypeAndShapeInfo*": "^OrtTensorTypeAndShapeInfo",
    "ONNXTensorElementDataType*": "^ONNXTensorElementDataType",
    "OrtAllocatorType": "OrtAllocatorType",
    "OrtMemType": "OrtMemType",
    "OrtMemoryInfo**": "^^OrtMemoryInfo",
    "OrtMemType*": "^OrtMemType",
    "OrtAllocatorType*": "^OrtAllocatorType",
    "OrtAllocator**": "^^OrtAllocator",
    "ONNXType": "ONNXType",
    "OrtKernelInfo*": "^OrtKernelInfo",
    "OrtKernelContext*": "^OrtKernelContext",
    "OrtMapTypeInfo**": "^^OrtMapTypeInfo",
    "OrtSequenceTypeInfo**": "^^OrtSequenceTypeInfo",
    "OrtMapTypeInfo*": "^OrtMapTypeInfo",
    "OrtSequenceTypeInfo*": "^OrtSequenceTypeInfo",
    "OrtModelMetadata**": "^^OrtModelMetadata",
    "OrtModelMetadata*": "^OrtModelMetadata",
    "OrtThreadingOptions*": "^OrtThreadingOptions",
    "OrtThreadingOptions**": "^^OrtThreadingOptions",
    "OrtIoBinding*": "^OrtIoBinding",
    "OrtIoBinding**": "^^OrtIoBinding",
    "OrtLanguageProjection": "OrtLanguageProjection",
    "OrtCUDAProviderOptions*": "^OrtCUDAProviderOptions",
    "OrtROCMProviderOptions*": "^OrtROCMProviderOptions",
    "OrtOpenVINOProviderOptions*": "^OrtOpenVINOProviderOptions",
    "OrtArenaCfg**": "^^OrtArenaCfg",
    "OrtTensorRTProviderOptions*": "^OrtTensorRTProviderOptions",
    "OrtPrepackedWeightsContainer**": "^^OrtPrepackedWeightsContainer",
    "OrtPrepackedWeightsContainer*": "^OrtPrepackedWeightsContainer",
    "OrtTensorRTProviderOptionsV2*": "^OrtTensorRTProviderOptionsV2",
    "OrtTensorRTProviderOptionsV2**": "^^OrtTensorRTProviderOptionsV2",
    "OrtSparseFormat*": "^OrtSparseFormat",
    "OrtSparseIndicesFormat": "OrtSparseIndicesFormat",
    "OrtCustomCreateThreadFn": "OrtCustomCreateThreadFn",
    "OrtCustomJoinThreadFn": "OrtCustomJoinThreadFn",
    "OrtCUDAProviderOptionsV2*": "^OrtCUDAProviderOptionsV2",
    "OrtCUDAProviderOptionsV2**": "^^OrtCUDAProviderOptionsV2",
    "OrtMIGraphXProviderOptions*": "^OrtMIGraphXProviderOptions",
    "OrtOpAttrType": "OrtOpAttrType",
    "OrtOpAttr**": "^^OrtOpAttr",
    "OrtOpAttr*": "^OrtOpAttr",
    "OrtOp**": "^^OrtOp",
    "OrtOp*": "^OrtOp",
    "OrtKernelInfo**": "^^OrtKernelInfo",
    "OrtTrainingApi*": "^OrtTrainingApi",
    "OrtCANNProviderOptions*": "^OrtCANNProviderOptions",
    "OrtCANNProviderOptions**": "^^OrtCANNProviderOptions",
    "OrtMemoryInfoDeviceType*": "^OrtMemoryInfoDeviceType",
    "OrtDnnlProviderOptions*": "^OrtDnnlProviderOptions",
    "OrtDnnlProviderOptions**": "^^OrtDnnlProviderOptions",
    "OrtLogger**": "^^OrtLogger",
    "OrtLogger*": "^OrtLogger",
    "OrtLoggingLevel*": "^OrtLoggingLevel",
    "OrtOptionalTypeInfo**": "^^OrtOptionalTypeInfo",
    "OrtOptionalTypeInfo*": "^OrtOptionalTypeInfo",
    "OrtROCMProviderOptions**": "^^OrtROCMProviderOptions",
    "RunAsyncCallbackFn": "RunAsyncCallbackFn",
    "OrtShapeInferContext*": "^OrtShapeInferContext",
    "OrtCustomOpInputOutputCharacteristic": "OrtCustomOpInputOutputCharacteristic",
    # Questionable types
    "void**": "^rawptr",
    # char *dim_params[] : ^^dim_params? [^]cstring?
    "char***": "^^^c.char",
    "OrtValue***": "^^^OrtValue",
    # FIXME:
    "proc(_: rawptr,  _: c.size_t, )": "proc(_: rawptr,  _: c.size_t, )",
}


def normalize_field_name(field_name: str) -> str:
    if field_name in ("context", "in"):
        return field_name + "_"
    else:
        return field_name


def filter_ast(ast: Dict) -> Dict:

    ast_filtered = {"ext": []}

    for node in ast["ext"]:
        # > skipping system symbols
        if node["_nodetype"] == "Typedef" and node["name"].startswith("__"):
            # print(e["name"])
            continue
        if (
            node["_nodetype"] == "Decl"
            and len(node["storage"]) > 0
            and node["storage"][0] == "extern"
        ):
            # print(e["name"])
            continue
        # < strerror the last one external symbol
        # -----------------------------------------------------
        # > build-in types
        if (
            node["_nodetype"] == "Typedef"
            and node["type"]["_nodetype"] == "TypeDecl"
            and not node["type"]["type"].get("name")
        ):
            # print(node["name"], node["type"]["type"].get("name"))
            continue
        ast_filtered["ext"].append(node)

    return ast_filtered


def write_enums(ast_filtered: Dict, onnx_bindings_file: io.TextIOWrapper) -> Dict:
    rem = {"ext": []}
    for node in ast_filtered["ext"]:
        # > processing enums
        # typedef enum Name {...}
        if (
            node["_nodetype"] == "Typedef"
            and node["type"]["_nodetype"] == "TypeDecl"
            and node["type"]["type"]["_nodetype"] == "Enum"
        ):
            assert node["name"] == node["type"]["type"].get("name")
            # print(node["name"], node["type"]["type"].get("name"))
            onnx_bindings_file.write(f"{node['name']} :: enum c.int {{\n")
            for i in node["type"]["type"]["values"]["enumerators"]:
                if i["value"]:
                    if i["value"]["_nodetype"] == "Constant":
                        onnx_bindings_file.write(
                            f"\t{i['name']} = {i['value']['value']},\n"
                        )
                    elif i["value"]["_nodetype"] == "UnaryOp":
                        onnx_bindings_file.write(
                            f"\t{i['name']} = {i['value']['op']}{i['value']['expr']['value']},\n"
                        )
                else:
                    onnx_bindings_file.write(f"\t{i['name']},\n")
            onnx_bindings_file.write("}\n\n\n")
        # enum Name {...}
        elif node["_nodetype"] == "Decl" and node["type"]["_nodetype"] == "Enum":
            onnx_bindings_file.write(f"{node['type']['name']} :: enum c.int {{\n")
            for i in node["type"]["values"]["enumerators"]:
                if i["value"]:
                    if i["value"]["_nodetype"] == "Constant":
                        onnx_bindings_file.write(
                            f"\t{i['name']} = {i['value']['value']},\n"
                        )
                    elif i["value"]["_nodetype"] == "UnaryOp":
                        onnx_bindings_file.write(
                            f"\t{i['name']} = {i['value']['op']}{i['value']['expr']['value']},\n"
                        )
                else:
                    onnx_bindings_file.write(f"\t{i['name']},\n")
            onnx_bindings_file.write("}\n\n\n")
        # < processing enums
        else:
            rem["ext"].append(node)
    return rem


def write_func_definition(ast_rem: Dict, onnx_bindings_file: io.TextIOWrapper) -> Dict:
    rem = {"ext": []}
    for node in ast_rem["ext"]:
        if (
            node["_nodetype"] == "Typedef"
            and node["storage"]
            and node["storage"][0] == "typedef"
            and node["type"]["_nodetype"] == "PtrDecl"
            and node["type"]["type"]["_nodetype"] == "FuncDecl"
        ):
            # print("FUNCTION:\t", node["name"])

            if node["name"] not in (
                "OrtLoggingFunction",
                "OrtThreadWorkerFn",
                "OrtCustomCreateThreadFn",
                "OrtCustomJoinThreadFn",
                "RegisterCustomOpsFn",
                "RunAsyncCallbackFn",
            ):
                raise ValueError(f"Declaration of new funciton: {node['name']}")

            onnx_bindings_file.write(f'{node["name"]} :: #type proc "c" (\n')

            for param in node["type"]["type"]["args"]["params"]:

                param_name = param["name"]
                param_type = type_1(param)
                if not param_type:
                    param_type = type_2(param)
                if not param_type:
                    param_type = type_3(param)
                if not param_type:
                    raise ValueError(
                        "FUNCTION:", node["name"], f"param '{param_name}' parse failed"
                    )

                # print(
                #     param["name"], "\t", param["type"]["type"].keys(), "\t", param_type
                # )

                odin_type = type_table[param_type]
                onnx_bindings_file.write(f"\t{param_name}: {odin_type},\n")
            # parsing return type
            ret_type = type_1(node["type"]["type"])
            if not ret_type:
                ret_type = type_2(node["type"]["type"])
            if not ret_type:
                ret_type = type_3(node["type"]["type"])
            if not ret_type:
                raise ValueError(
                    "FUNCTION:",
                    node["name"],
                    f"return type `{node['type']['type']}` parse failed",
                )
            # print(f"\tret_type: {ret_type}\n\n")

            ret_type = type_table[ret_type]
            if ret_type:
                onnx_bindings_file.write(f") -> {ret_type}\n\n")
            else:
                onnx_bindings_file.write(")\n\n")

        else:
            rem["ext"].append(node)
    return rem


def write_func_declaration(ast_rem: Dict, onnx_bindings_file: io.TextIOWrapper) -> Dict:
    rem = {"ext": []}

    for node in ast_rem["ext"]:
        if (
            node["_nodetype"] == "Decl"
            and "funcspec" in node.keys()
            and node["type"]["_nodetype"] == "FuncDecl"
        ):
            # print("FUNCTION DECL:", node["name"])
            if node["name"] not in (
                "OrtGetApiBase",
                "OrtSessionOptionsAppendExecutionProvider_CUDA",
                "OrtSessionOptionsAppendExecutionProvider_ROCM",
                "OrtSessionOptionsAppendExecutionProvider_MIGraphX",
                "OrtSessionOptionsAppendExecutionProvider_Dnnl",
                "OrtSessionOptionsAppendExecutionProvider_Tensorrt",
            ):
                raise ValueError(f"Declaration of new funciton: {node['name']}")

            func_name = node["name"]

            # parse params
            param_list = []
            for param in node["type"]["args"]["params"]:

                param_name = param["name"]
                param_type = type_1(param)
                if not param_type:
                    param_type = type_2(param)
                if not param_type:
                    param_type = type_3(param)
                if not param_type:
                    raise ValueError(
                        "FUNCTION:", node["name"], f"param '{param_name}' parse failed"
                    )

                param_name = normalize_field_name(param_name)
                odin_type = type_table[param_type]
                if odin_type:
                    param_list.append(f"{param_name}: {odin_type}, ")

            # parse return
            ret_type = type_1(node["type"])
            if not ret_type:
                ret_type = type_2(node["type"])
            if not ret_type:
                ret_type = type_3(node["type"])
            if not ret_type:
                raise ValueError(
                    "FUNCTION:",
                    node["name"],
                    f"return type `{node['type']}` parse failed",
                )
            ret_type = type_table[ret_type]
            if ret_type:
                onnx_bindings_file.write(
                    f"\t{func_name} :: proc({''.join(param_list)}) -> {ret_type} ---\n"
                )
            else:
                onnx_bindings_file.write(
                    f"\t{func_name} :: proc({''.join(param_list)}) ---\n"
                )
        else:
            rem["ext"].append(node)
    onnx_bindings_file.write("\n")
    return rem


def _parse_func_pointer(node: Dict) -> Tuple[str, str, str, Dict]:
    func_name = node["name"]

    function = {"func_name": "", "params": [], "ret_type": ""}

    function["func_name"] = func_name

    # parse params
    param_list = []
    for param in node["type"]["type"]["args"]["params"]:

        param_name = param["name"]
        param_type = type_1(param)
        # print(f"|||||||{param_name}: {param_type}")
        if not param_type:
            param_type = type_2(param)
        if not param_type:
            param_type = type_3(param)
        if not param_type:
            param_type = type_4(param)
        if not param_type:
            raise ValueError(
                "FUNCTION:", node["name"], f"param '{param_name}' parse failed"
            )

        param_name = normalize_field_name(param_name) or "_"
        odin_type = type_table[param_type]
        if odin_type:
            param_list.append(f"{param_name}: {odin_type}, ")

    # parse return
    ret_type = type_1(node["type"]["type"])
    if not ret_type:
        ret_type = type_2(node["type"]["type"])
    if not ret_type:
        ret_type = type_3(node["type"]["type"])
    if not ret_type:
        raise ValueError(
            "FUNCTION:",
            node["name"],
            f"return type `{node['type']}` parse failed",
        )
    ret_type = type_table[ret_type]
    return func_name, param_list, ret_type, function


def write_func_pointer_as_struct_field(
    node: Dict, onnx_bindings_file: io.TextIOWrapper
) -> List[Dict]:
    # parse function pointer as struct field and return
    # a list of dicts representing parsed funcitons

    functions = []
    if (
        node["_nodetype"] == "Decl"
        and "funcspec" in node.keys()
        and node["type"]["_nodetype"] == "PtrDecl"
        and node["type"]["type"]["_nodetype"] == "FuncDecl"
    ):
        # print("FUNCTION DECL:", node["name"])
        func_name, param_list, ret_type, function = _parse_func_pointer(node)
        if ret_type:
            onnx_bindings_file.write(
                f"\t{func_name} : proc({''.join(param_list)}) -> {ret_type},\n"
            )
        else:
            onnx_bindings_file.write(f"\t{func_name} : proc({''.join(param_list)}),\n")

        function["params"] = param_list
        function["ret_type"] = ret_type
        functions.append(function)
    onnx_bindings_file.write("\n")
    return functions


def write_typedef_struct_pointer(
    ast_rem: Dict, onnx_bindings_file: io.TextIOWrapper
) -> Dict:
    # typedef OrtStatus *OrtStatusPtr;
    # typedef const struct OrtCustomHandleType {
    #     char __place_holder;
    # } *OrtCustomThreadHandle;
    rem = {"ext": []}
    for node in ast_rem["ext"]:
        if (
            node["_nodetype"] == "Typedef"
            and node["name"]
            and node["type"]["_nodetype"] == "PtrDecl"
            and node["type"]["type"]["_nodetype"] == "TypeDecl"
        ):
            # print("***", node["name"])
            # typedef OrtStatus *OrtStatusPtr;
            if node["type"]["type"]["type"]["_nodetype"] == "IdentifierType":
                alias_name = node["name"]
                alias_type = type_1(node)
                if not alias_type:
                    alias_type = type_2(node)
                if not alias_type:
                    alias_type = type_3(node)
                if not alias_type:
                    raise ValueError(f"Error parsing struct: {node}")

                alias_type = type_table[alias_type]
                onnx_bindings_file.write(f"{alias_name} :: {alias_type} \n\n")
            # typedef const struct OrtCustomHandleType {
            #     char __place_holder;
            # } *OrtCustomThreadHandle;
            elif node["type"]["type"]["type"]["_nodetype"] == "Struct":
                # 1. parsing struct
                struct_name = node["type"]["type"]["type"]["name"]
                onnx_bindings_file.write(f"{struct_name} :: struct {{\n")
                for decl in node["type"]["type"]["type"]["decls"]:
                    field_name = decl["name"]
                    field_type = type_1(decl)
                    if not field_type:
                        field_type = type_2(decl)
                    if not field_type:
                        field_type = type_3(decl)
                    if not field_type:
                        raise ValueError(
                            f"Error: failed parse struct field. {field_name}, {decl}"
                        )
                    field_type = type_table[field_type]
                    field_name = normalize_field_name(field_name)
                    onnx_bindings_file.write(f"\t{field_name}: {field_type},\n")
                onnx_bindings_file.write("}\n\n")
                # 2. making alias
                # because node["type"]["_nodetype"] == "PtrDecl" the type of alias is pointer
                onnx_bindings_file.write(f"{node['name']} :: ^{struct_name}\n\n")

            else:
                raise ValueError(
                    f"Error: unknown type of struct: {node['name']},{node['coord']} "
                )

        else:
            rem["ext"].append(node)
    return rem


def get_decl_typedef(
    ast_rem: Dict,
) -> Tuple[List[str], List[str], List[str], List[str]]:
    defined_struct = []  # struct Name {...}
    declared_struct = []  # struct OrtEnv;
    declared_typedef_struct = []  # typedef struct OrtEnv OrtEnv;
    defined_typedef_struct = []  # typedef struct Name {...}

    for node in ast_rem["ext"]:

        # > processing structs
        # >> processing ORT_RUNTIME_CLASS(Env);
        # // Expands to
        # struct OrtEnv;
        # typedef struct OrtEnv OrtEnv;

        # struct Name {...}
        if (
            node["_nodetype"] == "Decl"
            and not node["name"]
            and node["type"]["_nodetype"] == "Struct"
            and node["type"].get("decls")
            and node["type"].get("name")
        ):
            # print("+++", node["type"]["name"])
            defined_struct.append(node["type"]["name"])

        # struct OrtEnv;
        if (
            node["_nodetype"] == "Decl"
            and not node["name"]
            and node["type"]["_nodetype"] == "Struct"
            and not node["type"]["decls"]
            and not node["storage"]
        ):
            # print("struct", node["type"]["name"])
            declared_struct.append(node["type"]["name"])

        # typedef struct OrtEnv OrtEnv;
        if (
            node["_nodetype"] == "Typedef"
            and node["name"]
            and node["type"]["_nodetype"] == "TypeDecl"
            and node["type"]["type"]["_nodetype"] == "Struct"
            and not node["type"]["type"].get("decls")
        ):
            # print("typedef struct", node["type"]["name"])
            declared_typedef_struct.append(node["name"])

        # typedef struct Name {...}
        if (
            node["_nodetype"] == "Typedef"
            and node["name"]
            and node["type"]["_nodetype"] == "TypeDecl"
            and node["type"]["type"]["_nodetype"] == "Struct"
            and node["type"]["type"].get("decls")
        ):
            # print(">>>", node["name"])
            defined_typedef_struct.append(node["name"])
    return (
        defined_struct,
        declared_struct,
        declared_typedef_struct,
        defined_typedef_struct,
    )


def write_declared_structs(
    ast_rem: Dict, onnx_bindings_file: io.TextIOWrapper, defined: List[str]
) -> Tuple[Dict, List[str]]:
    rem = {"ext": []}

    written_structs = []

    for node in ast_rem["ext"]:
        # struct OrtEnv;
        if (
            node["_nodetype"] == "Decl"
            and not node["name"]
            and node["type"]["_nodetype"] == "Struct"
            and not node["type"]["decls"]
            and not node["storage"]
            # exclude defined
            and node["type"]["name"] not in defined
        ):
            # print("struct", node["type"]["name"])
            onnx_bindings_file.write(f'{node["type"]["name"]} :: struct {{}}\n')
            written_structs.append(node["type"]["name"])
        else:
            rem["ext"].append(node)
    onnx_bindings_file.write("\n")
    return rem, written_structs


def write_declared_typedef_struct(
    ast_rem: Dict,
    onnx_bindings_file: io.TextIOWrapper,
    written_structs: List[str],
    defined: List["str"],
) -> Dict:
    rem = {"ext": []}

    for node in ast_rem["ext"]:
        # typedef struct OrtEnv OrtEnv;
        if (
            node["_nodetype"] == "Typedef"
            and node["name"]
            and node["type"]["_nodetype"] == "TypeDecl"
            and node["type"]["type"]["_nodetype"] == "Struct"
            and not node["type"]["type"].get("decls")
            # exclude defined
            and node["name"] not in defined
        ):
            # print("typedef struct", node["type"]["name"])
            if node["name"] not in written_structs:
                onnx_bindings_file.write(f'{node["type"]["name"]} :: struct {{}}\n')
        else:
            rem["ext"].append(node)
    onnx_bindings_file.write("\n")
    return rem


def write_defined_structs(
    ast_rem: Dict, onnx_bindings_file: io.TextIOWrapper, defined: List[str]
) -> List[Dict]:
    rem = {"ext": []}
    functions: List[Dict] = []

    def is_func_pointer(param: dict) -> bool:
        return (
            # param["_nodetype"] == "Decl"
            param["type"]["_nodetype"] == "PtrDecl"
            and param["type"]["type"]["_nodetype"] == "FuncDecl"
        )

    for node in ast_rem["ext"]:
        # struct Name {...}
        if (
            node["_nodetype"] == "Decl"
            and not node["name"]
            and node["type"]["_nodetype"] == "Struct"
            and node["type"].get("decls")
            and node["type"].get("name")
        ):
            print("+++", node["type"]["name"])
            # -------------------------------------------------------------------------------
            struct_name = node["type"]["name"]
            onnx_bindings_file.write(f"{struct_name} :: struct {{\n")
            for decl in node["type"]["decls"]:
                if not is_func_pointer(decl):
                    field_name = decl["name"]
                    field_type = type_1(decl)
                    if not field_type:
                        field_type = type_2(decl)
                    if not field_type:
                        field_type = type_3(decl)
                    if not field_type:
                        raise ValueError(
                            f"Error: failed parse struct field. {field_name}, {decl}"
                        )
                    field_type = type_table[field_type]
                    print(f"\t{field_name}:{field_type} ")
                    field_name = normalize_field_name(field_name)
                    onnx_bindings_file.write(f"\t{field_name}: {field_type},\n")
                else:
                    # print(f"\tFUNC POINTER {decl['name']}")
                    functs = write_func_pointer_as_struct_field(
                        decl, onnx_bindings_file
                    )
                    functions.extend(functs)
            onnx_bindings_file.write("}\n\n")
            # -------------------------------------------------------------------------------

        # typedef struct Name {...}
        elif (
            node["_nodetype"] == "Typedef"
            and node["name"]
            and node["type"]["_nodetype"] == "TypeDecl"
            and node["type"]["type"]["_nodetype"] == "Struct"
            and node["type"]["type"].get("decls")
        ):
            print(">>>", node["name"])
            struct_name = node["name"]
            onnx_bindings_file.write(f"{struct_name} :: struct {{\n")
            for decl in node["type"]["type"]["decls"]:
                if not is_func_pointer(decl):
                    field_name = decl["name"]
                    field_type = type_1(decl)
                    if not field_type:
                        field_type = type_2(decl)
                    if not field_type:
                        field_type = type_3(decl)
                    if not field_type:
                        raise ValueError(
                            f"Error: failed parse struct field. {field_name}, {decl}"
                        )
                    field_type = type_table[field_type]
                    print(f"\t{field_name}:{field_type} ")
                    field_name = normalize_field_name(field_name)
                    onnx_bindings_file.write(f"\t{field_name}: {field_type},\n")
                else:
                    # print(f"\tFUNC POINTER {decl['name']}")
                    functs = write_func_pointer_as_struct_field(
                        decl, onnx_bindings_file
                    )
                    functions.extend(functs)
            onnx_bindings_file.write("}\n\n")

        else:
            rem["ext"].append(node)

    return rem, functions


def write_funcitons(functions: List[Dict], onnx_bindings_file: io.TextIOWrapper):
    for fn in functions:
        if fn["ret_type"]:
            onnx_bindings_file.write(
                f'\t{fn["func_name"]} :: proc({"".join(fn["params"])}) -> {fn["ret_type"]} ---\n'
            )
        else:
            onnx_bindings_file.write(
                f'\t{fn["func_name"]} :: proc({"".join(fn["params"])}) ---\n'
            )


def type_1(param: Dict) -> str:
    # int a;
    decl = param["type"]["_nodetype"]
    identifier = param["type"]["type"]["_nodetype"]  # IdentifierType
    if decl != "TypeDecl":
        return ""
    if identifier == "IdentifierType":
        type_ = " ".join(param["type"]["type"]["names"])
        return type_
    elif identifier == "Enum":
        type_ = param["type"]["type"]["name"]
        return type_
    else:
        raise Exception("Unknown type!")


def type_2(param: Dict) -> str:
    # int *a;
    decl = param["type"]["type"]["_nodetype"]
    identifier = param["type"]["type"]["type"]["_nodetype"]  # IdentifierType
    if decl != "TypeDecl":
        return ""

    if identifier == "IdentifierType":
        type_a = " ".join(param["type"]["type"]["type"]["names"])
        type_b = param["type"]["_nodetype"]
        if type_b == "PtrDecl":
            type_b = "*"
        else:
            raise ValueError(f"Error parsing type_2: {param['type']['coord']}")
        return f"{type_a}{type_b}"
    elif identifier == "Struct":
        # print(">>> parsing Struct!")
        type_a = param["type"]["type"]["type"]["name"]
        type_b = param["type"]["_nodetype"]
        if type_b == "PtrDecl":
            type_b = "*"
        else:
            raise ValueError(f"Error parsing type_2: {param['type']['coord']}")
        return f"{type_a}{type_b}"
    elif identifier == "Enum":
        # print(">>> parsing Enum!")
        type_a = param["type"]["type"]["type"]["name"]
        type_b = param["type"]["_nodetype"]
        if type_b == "PtrDecl":
            type_b = "*"
        else:
            raise ValueError(f"Error parsing type_2: {param['type']['coord']}")
        return f"{type_a}{type_b}"
    else:
        raise Exception("Unknown type!")


def type_3(param: Dict) -> str:
    # int **a;
    decl = param["type"]["type"]["type"]["_nodetype"]
    identifier = param["type"]["type"]["type"]["type"]["_nodetype"]  # IdentifierType
    if decl != "TypeDecl":
        return ""

    if identifier == "IdentifierType":
        type_a = " ".join(param["type"]["type"]["type"]["type"]["names"])
        assert type_a

        type_b = param["type"]["type"]["_nodetype"]
        if type_b == "PtrDecl":
            type_b = "*"
        elif (
            type_b == "FuncDecl"
            and param["type"]["_nodetype"] == "PtrDecl"
            and "funcspec" in param.keys()
            and param["_nodetype"] == "Decl"
        ):  # pointer to a function
            # KernelContext_ParallelFor: proc(
            #     context_: OrtKernelContext,
            #     fn: proc(_: rawptr, _: c.size_t),  <--------------------
            #     total: c.size_t,
            #     num_batch: c.size_t,
            #     user_dats: rawptr,
            # ) -> OrtStatusPtr,
            _, param_list, ret_type, _ = _parse_func_pointer(param)
            if ret_type:
                return f"proc({' '.join(param_list)}) -> {ret_type},"
            else:
                return f"proc({' '.join(param_list)})"
        else:
            raise ValueError(f"Error parsing type_3: {param['type']['coord']}")

        type_c = param["type"]["_nodetype"]
        if type_c == "PtrDecl":
            type_c = "*"
        elif type_c == "ArrayDecl":
            type_c = "*"
        else:
            raise ValueError(f"Error parsing type_3: {param['type']['coord']}")
        return f"{type_a}{type_b}{type_c}"

    elif identifier == "Struct":
        type_a = param["type"]["type"]["type"]["type"]["name"]

        type_b = param["type"]["type"]["_nodetype"]
        if type_b == "PtrDecl":
            type_b = "*"
        else:
            raise ValueError(f"Error parsing type_3: {param['type']['coord']}")

        type_c = param["type"]["_nodetype"]
        if type_c == "PtrDecl":
            type_c = "*"
        elif type_c == "ArrayDecl":
            type_c = "*"
        else:
            raise ValueError(f"Error parsing type_3: {param['type']['coord']}")

        return f"{type_a}{type_b}{type_c}"
    else:
        raise Exception("Unknown type!")


def type_4(param: Dict) -> str:
    # int ***a;
    decl = param["type"]["type"]["type"]["type"]["_nodetype"]
    identifier = param["type"]["type"]["type"]["type"]["type"][
        "_nodetype"
    ]  # IdentifierType
    if decl != "TypeDecl":
        return ""

    if identifier == "IdentifierType":
        type_a = " ".join(param["type"]["type"]["type"]["type"]["type"]["names"])
        assert type_a

        type_b = param["type"]["type"]["type"]["_nodetype"]
        if type_b == "PtrDecl":
            type_b = "*"
        else:
            raise ValueError(f"Error parsing type_4: {param['type']['coord']}")

        type_c = param["type"]["type"]["_nodetype"]
        if type_c == "PtrDecl":
            type_c = "*"
        elif type_c == "ArrayDecl":
            type_c = "*"
        else:
            raise ValueError(f"Error parsing type_4: {param['type']['coord']}")

        type_d = param["type"]["_nodetype"]
        if type_d == "PtrDecl":
            type_d = "*"
        elif type_d == "ArrayDecl":
            type_d = "*"
        else:
            raise ValueError(f"Error parsing type_4: {param['type']['coord']}")

        return f"{type_a}{type_b}{type_c}{type_d}"
    else:
        raise Exception("Unknown type!")


# ------------------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Translator (not a binfing generator!) of ONNX C header into Odin"
    )
    parser.add_argument(
        "--libonnxruntime-path",
        type=str,
        required=True,
        help="Path to libonnxruntime.so file",
    )
    parser.add_argument(
        "--os", type=str, required=True, choices=("Linux", "Windows", "Darwin")
    )
    parser.add_argument(
        "--package",
        type=str,
        required=True,
        help="Odin package name to be specified in odin binding file",
    )
    args = parser.parse_args()

    onnx_bindings_odin_path: str = "onnxbinding.odin"
    ort_version_path: str = "ORT_API_VERSION.version"
    ast_json_path: str = "dump.json"

    # onnx_shared_lib_path: Dict[str, str] = {
    #     "Linux": "/thirdparty/onnxruntime/lib/libonnxruntime.so",
    #     "Windows": "",
    #     "Darwin": "",
    # }
    os_ = args.os
    onnx_shared_lib_path = args.libonnxruntime_path
    package = args.package

    with open(
        file=onnx_bindings_odin_path, mode="wt", encoding="utf8"
    ) as onnx_bindings_file, open(
        file=ast_json_path, mode="rt", encoding="utf8"
    ) as ast_json_file:

        # load ast
        ast_json = json.load(fp=ast_json_file)

        # define package
        onnx_bindings_file.write(f"package {package}" + "\n\n")
        # make imports
        onnx_bindings_file.write('import "core:c"' + "\n\n")

        onnx_bindings_file.write(
            f'''when ODIN_OS == .{os_} do foreign import onnx "{onnx_shared_lib_path}"'''
            + "\n\n"
        )

        # load and write ORT_API_VERSION version
        with open(file=ort_version_path, mode="rt", encoding="utf8") as f:
            ort_api_version = f.readline().strip()
            onnx_bindings_file.write(ort_api_version + "\n\n")

        print("#nodes:", len(ast_json["ext"]))

        # TODO:
        # ifdef _WIN32
        # ifdef __APPLE__

        ast_filtered = filter_ast(ast_json)

        onnx_bindings_file.write("// Enums:\n")
        ast_rem = write_enums(ast_filtered, onnx_bindings_file)
        onnx_bindings_file.write("// End Enums\n\n")

        onnx_bindings_file.write("// Function pointer declarations:\n")
        ast_rem = write_func_definition(ast_rem, onnx_bindings_file)
        onnx_bindings_file.write("// End Function pointer declarations\n\n")

        onnx_bindings_file.write("// Function pointer definitions:\n")
        onnx_bindings_file.write('@(default_calling_convention = "c")\n')
        onnx_bindings_file.write("foreign onnx {\n")
        ast_rem = write_func_declaration(ast_rem, onnx_bindings_file)
        onnx_bindings_file.write("}\n\n")
        onnx_bindings_file.write("// End Function pointer definitions\n\n")

        onnx_bindings_file.write("// Type definition of struct pointers:\n")
        ast_rem = write_typedef_struct_pointer(ast_rem, onnx_bindings_file)
        onnx_bindings_file.write("// End Type definition of struct pointers\n\n")

        # struct can be forward declared, we should ignore forward declaration
        # because in Odin's terms forvard declaration is a redeclaration which is
        # compile error
        (
            defined_struct,  # struct Name {...}
            declared_struct,  #### struct OrtEnv;
            declared_typedef_struct,  #### typedef struct OrtEnv OrtEnv;
            defined_typedef_struct,  # typedef struct Name {...}
        ) = get_decl_typedef(ast_rem)
        # print(f">>> {len(defined_struct)=} {defined_struct = }\n")
        # print(f">>> {len(declared_struct)=} {declared_struct = }\n")
        # print(f">>> {len(declared_typedef_struct)=} {declared_typedef_struct = }\n")
        # print(f">>> {len(defined_typedef_struct)=} {defined_typedef_struct = }\n")

        onnx_bindings_file.write("// Declared structs:\n")
        defined = defined_struct + defined_typedef_struct
        ast_rem, written_structs = write_declared_structs(
            ast_rem, onnx_bindings_file, defined=defined
        )
        onnx_bindings_file.write("// End Declared structs\n\n")

        onnx_bindings_file.write("// Declared typedef structs:\n")
        ast_rem = write_declared_typedef_struct(
            ast_rem, onnx_bindings_file, written_structs, defined
        )
        onnx_bindings_file.write("// End Declared typedef structs\n\n")

        # >>> OrtAllocator
        # >>> OrtCUDAProviderOptions
        # >>> OrtROCMProviderOptions
        # >>> OrtTensorRTProviderOptions
        # >>> OrtMIGraphXProviderOptions
        # >>> OrtOpenVINOProviderOptions
        # +++ OrtApiBase
        # +++ OrtApi
        # +++ OrtCustomOp
        ast_rem, functions = write_defined_structs(
            ast_rem=ast_rem, onnx_bindings_file=onnx_bindings_file, defined=defined
        )

        onnx_bindings_file.write("// Functions:\n")
        onnx_bindings_file.write('@(default_calling_convention = "c")\n')
        onnx_bindings_file.write("foreign onnx {\n")
        write_funcitons(functions, onnx_bindings_file)
        onnx_bindings_file.write("}\n\n")
        onnx_bindings_file.write("// End Functions\n\n")

        # Check if all structs, enums, and functions were parsed:
        rem = {"ext": []}
        for node in ast_rem["ext"]:
            node_name = node["name"] or node["type"]["name"]
            if node_name in written_structs + defined:
                continue
            else:
                rem["ext"].append(node)

        assert len(rem["ext"]) == 0

        # with open("rem.json", "wt", encoding="utf8") as f:
        #     json.dump(rem, fp=f, indent=2)
