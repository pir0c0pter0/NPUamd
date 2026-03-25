#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <onnxruntime/core/session/onnxruntime_c_api.h>

typedef struct TensorMeta {
  char* name;
  int64_t* dims;
  size_t dim_count;
  size_t element_count;
  ONNXTensorElementDataType element_type;
} TensorMeta;

static void release_status(const OrtApi* api, OrtStatus* status) {
  if (status != NULL) {
    api->ReleaseStatus(status);
  }
}

static int fail_status(const OrtApi* api, const char* step, OrtStatus* status) {
  const char* message = api->GetErrorMessage(status);
  fprintf(stderr, "[FAIL] %s: %s\n", step, message);
  release_status(api, status);
  return 1;
}

static void reset_tensor_meta(TensorMeta* meta) {
  memset(meta, 0, sizeof(*meta));
}

static void free_tensor_meta(const OrtApi* api, OrtAllocator* allocator, TensorMeta* meta) {
  if (meta->name != NULL && allocator != NULL) {
    api->AllocatorFree(allocator, meta->name);
  }
  free(meta->dims);
  reset_tensor_meta(meta);
}

static const char* tensor_type_name(ONNXTensorElementDataType type) {
  switch (type) {
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:
      return "float32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:
      return "uint8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:
      return "int8";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT16:
      return "uint16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT16:
      return "int16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:
      return "int32";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:
      return "int64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_DOUBLE:
      return "float64";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16:
      return "float16";
    case ONNX_TENSOR_ELEMENT_DATA_TYPE_BFLOAT16:
      return "bfloat16";
    default:
      return "unsupported";
  }
}

static void print_dims(FILE* stream, const int64_t* dims, size_t dim_count) {
  size_t i;
  fputc('[', stream);
  for (i = 0; i < dim_count; ++i) {
    if (i > 0) {
      fputs(", ", stream);
    }
    fprintf(stream, "%lld", (long long)dims[i]);
  }
  fputc(']', stream);
}

static size_t safe_element_count(const int64_t* dims, size_t dim_count) {
  size_t i;
  size_t count = 1;

  for (i = 0; i < dim_count; ++i) {
    int64_t dim = dims[i];
    if (dim <= 0) {
      dim = 1;
    }
    count *= (size_t)dim;
  }

  return count;
}

static int load_tensor_meta(const OrtApi* api,
                            const OrtSession* session,
                            size_t index,
                            int is_input,
                            OrtAllocator* allocator,
                            TensorMeta* meta) {
  OrtStatus* status = NULL;
  OrtTypeInfo* type_info = NULL;
  const OrtTensorTypeAndShapeInfo* tensor_info = NULL;
  ONNXType onnx_type = ONNX_TYPE_UNKNOWN;

  reset_tensor_meta(meta);

  status = is_input ? api->SessionGetInputTypeInfo(session, index, &type_info)
                    : api->SessionGetOutputTypeInfo(session, index, &type_info);
  if (status != NULL) {
    return fail_status(api, is_input ? "SessionGetInputTypeInfo" : "SessionGetOutputTypeInfo", status);
  }

  status = api->GetOnnxTypeFromTypeInfo(type_info, &onnx_type);
  if (status != NULL) {
    api->ReleaseTypeInfo(type_info);
    return fail_status(api, "GetOnnxTypeFromTypeInfo", status);
  }
  if (onnx_type != ONNX_TYPE_TENSOR) {
    fprintf(stderr, "[FAIL] only tensor inputs/outputs are supported by this probe\n");
    api->ReleaseTypeInfo(type_info);
    return 1;
  }

  status = api->CastTypeInfoToTensorInfo(type_info, &tensor_info);
  if (status != NULL) {
    api->ReleaseTypeInfo(type_info);
    return fail_status(api, "CastTypeInfoToTensorInfo", status);
  }

  status = api->GetTensorElementType(tensor_info, &meta->element_type);
  if (status != NULL) {
    api->ReleaseTypeInfo(type_info);
    return fail_status(api, "GetTensorElementType", status);
  }

  status = api->GetDimensionsCount(tensor_info, &meta->dim_count);
  if (status != NULL) {
    api->ReleaseTypeInfo(type_info);
    return fail_status(api, "GetDimensionsCount", status);
  }

  meta->dims = (int64_t*)calloc(meta->dim_count == 0 ? 1 : meta->dim_count, sizeof(*meta->dims));
  if (meta->dims == NULL) {
    fprintf(stderr, "[FAIL] allocation failed for tensor dims\n");
    api->ReleaseTypeInfo(type_info);
    return 1;
  }

  if (meta->dim_count > 0) {
    size_t i;
    status = api->GetDimensions(tensor_info, meta->dims, meta->dim_count);
    if (status != NULL) {
      api->ReleaseTypeInfo(type_info);
      return fail_status(api, "GetDimensions", status);
    }
    for (i = 0; i < meta->dim_count; ++i) {
      if (meta->dims[i] <= 0) {
        meta->dims[i] = 1;
      }
    }
  }

  meta->element_count = safe_element_count(meta->dims, meta->dim_count);
  status = is_input ? api->SessionGetInputName(session, index, allocator, &meta->name)
                    : api->SessionGetOutputName(session, index, allocator, &meta->name);
  api->ReleaseTypeInfo(type_info);
  if (status != NULL) {
    return fail_status(api, is_input ? "SessionGetInputName" : "SessionGetOutputName", status);
  }

  return 0;
}

static int create_float_input_tensor(const OrtApi* api,
                                     const TensorMeta* meta,
                                     const OrtMemoryInfo* memory_info,
                                     OrtValue** input_tensor,
                                     float** input_buffer) {
  OrtStatus* status = NULL;
  float* buffer = NULL;
  size_t i;

  if (meta->element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    fprintf(stderr,
            "[FAIL] input tensor type %s is not supported by this probe; expected float32\n",
            tensor_type_name(meta->element_type));
    return 1;
  }

  buffer = (float*)malloc(meta->element_count * sizeof(*buffer));
  if (buffer == NULL) {
    fprintf(stderr, "[FAIL] allocation failed for input buffer\n");
    return 1;
  }

  for (i = 0; i < meta->element_count; ++i) {
    buffer[i] = (float)((int)(i % 19) - 9) / 8.0f;
  }

  status = api->CreateTensorWithDataAsOrtValue(memory_info,
                                               buffer,
                                               meta->element_count * sizeof(*buffer),
                                               meta->dims,
                                               meta->dim_count,
                                               meta->element_type,
                                               input_tensor);
  if (status != NULL) {
    free(buffer);
    return fail_status(api, "CreateTensorWithDataAsOrtValue", status);
  }

  *input_buffer = buffer;
  return 0;
}

static int print_first_output_summary(const OrtApi* api, OrtValue* output_tensor) {
  OrtStatus* status = NULL;
  OrtTensorTypeAndShapeInfo* tensor_info = NULL;
  ONNXTensorElementDataType element_type = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
  size_t dim_count = 0;
  int64_t* dims = NULL;
  size_t element_count = 0;
  int is_tensor = 0;
  int rc = 1;

  status = api->IsTensor(output_tensor, &is_tensor);
  if (status != NULL) {
    return fail_status(api, "IsTensor", status);
  }
  if (!is_tensor) {
    fprintf(stderr, "[FAIL] output is not a tensor\n");
    return 1;
  }

  status = api->GetTensorTypeAndShape(output_tensor, &tensor_info);
  if (status != NULL) {
    return fail_status(api, "GetTensorTypeAndShape", status);
  }

  status = api->GetTensorElementType(tensor_info, &element_type);
  if (status != NULL) {
    rc = fail_status(api, "GetTensorElementType(output)", status);
    goto cleanup;
  }

  status = api->GetDimensionsCount(tensor_info, &dim_count);
  if (status != NULL) {
    rc = fail_status(api, "GetDimensionsCount(output)", status);
    goto cleanup;
  }

  dims = (int64_t*)calloc(dim_count == 0 ? 1 : dim_count, sizeof(*dims));
  if (dims == NULL) {
    fprintf(stderr, "[FAIL] allocation failed for output dims\n");
    goto cleanup;
  }
  if (dim_count > 0) {
    size_t i;
    status = api->GetDimensions(tensor_info, dims, dim_count);
    if (status != NULL) {
      rc = fail_status(api, "GetDimensions(output)", status);
      goto cleanup;
    }
    for (i = 0; i < dim_count; ++i) {
      if (dims[i] <= 0) {
        dims[i] = 1;
      }
    }
  }
  element_count = safe_element_count(dims, dim_count);

  fprintf(stdout, "[OK] first output tensor type=%s shape=", tensor_type_name(element_type));
  print_dims(stdout, dims, dim_count);
  fputc('\n', stdout);

  if (element_count > 0 && element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
    float* output_data = NULL;
    status = api->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status != NULL) {
      rc = fail_status(api, "GetTensorMutableData(float)", status);
      goto cleanup;
    }
    fprintf(stdout, "[OK] first output value=%f\n", output_data[0]);
  } else if (element_count > 0 && element_type == ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64) {
    int64_t* output_data = NULL;
    status = api->GetTensorMutableData(output_tensor, (void**)&output_data);
    if (status != NULL) {
      rc = fail_status(api, "GetTensorMutableData(int64)", status);
      goto cleanup;
    }
    fprintf(stdout, "[OK] first output value=%lld\n", (long long)output_data[0]);
  }

  rc = 0;

cleanup:
  if (tensor_info != NULL) {
    api->ReleaseTensorTypeAndShapeInfo(tensor_info);
  }
  free(dims);
  return rc;
}

int main(int argc, char** argv) {
  const OrtApiBase* api_base;
  const OrtApi* api;
  OrtStatus* status = NULL;
  OrtEnv* env = NULL;
  OrtAllocator* allocator = NULL;
  OrtMemoryInfo* memory_info = NULL;
  OrtSessionOptions* session_options = NULL;
  OrtSession* session = NULL;
  OrtValue* input_tensor = NULL;
  OrtValue* output_tensor = NULL;
  float* input_buffer = NULL;
  const char* model_path;
  const char** keys = NULL;
  const char** values = NULL;
  TensorMeta input_meta;
  TensorMeta output_meta;
  size_t num_pairs = 0;
  size_t input_count = 0;
  size_t output_count = 0;
  int rc = 1;

  reset_tensor_meta(&input_meta);
  reset_tensor_meta(&output_meta);

  if (argc < 2) {
    fprintf(stderr, "usage: %s MODEL_PATH [key=value ...]\n", argv[0]);
    return 2;
  }

  model_path = argv[1];
  if (argc > 2) {
    int i;
    num_pairs = (size_t)(argc - 2);
    keys = (const char**)calloc(num_pairs, sizeof(*keys));
    values = (const char**)calloc(num_pairs, sizeof(*values));
    if (keys == NULL || values == NULL) {
      fprintf(stderr, "[FAIL] allocation failed for provider options\n");
      goto cleanup;
    }
    for (i = 2; i < argc; ++i) {
      char* pair = argv[i];
      char* eq = strchr(pair, '=');
      if (eq == NULL || eq == pair || eq[1] == '\0') {
        fprintf(stderr, "invalid provider option, expected key=value: %s\n", pair);
        rc = 2;
        goto cleanup;
      }
      *eq = '\0';
      keys[i - 2] = pair;
      values[i - 2] = eq + 1;
    }
  }

  api_base = OrtGetApiBase();
  api = api_base->GetApi(ORT_API_VERSION);
  if (api == NULL) {
    fprintf(stderr, "[FAIL] failed to get OrtApi version %d\n", ORT_API_VERSION);
    goto cleanup;
  }

  status = api->CreateEnv(ORT_LOGGING_LEVEL_VERBOSE, "npuamd-probe", &env);
  if (status != NULL) {
    return fail_status(api, "CreateEnv", status);
  }

  status = api->GetAllocatorWithDefaultOptions(&allocator);
  if (status != NULL) {
    return fail_status(api, "GetAllocatorWithDefaultOptions", status);
  }

  status = api->CreateSessionOptions(&session_options);
  if (status != NULL) {
    return fail_status(api, "CreateSessionOptions", status);
  }

  status = api->SetSessionLogSeverityLevel(session_options, 0);
  if (status != NULL) {
    return fail_status(api, "SetSessionLogSeverityLevel", status);
  }

  status = api->SetSessionLogVerbosityLevel(session_options, 1);
  if (status != NULL) {
    return fail_status(api, "SetSessionLogVerbosityLevel", status);
  }

  fprintf(stdout, "[INFO] appending VitisAIExecutionProvider with %zu option(s)\n", num_pairs);
  status = api->SessionOptionsAppendExecutionProvider_VitisAI(session_options, keys, values, num_pairs);
  if (status != NULL) {
    return fail_status(api, "SessionOptionsAppendExecutionProvider_VitisAI", status);
  }

  fprintf(stdout, "[INFO] creating session for %s\n", model_path);
  status = api->CreateSession(env, model_path, session_options, &session);
  if (status != NULL) {
    return fail_status(api, "CreateSession", status);
  }

  status = api->SessionGetInputCount(session, &input_count);
  if (status != NULL) {
    return fail_status(api, "SessionGetInputCount", status);
  }
  status = api->SessionGetOutputCount(session, &output_count);
  if (status != NULL) {
    return fail_status(api, "SessionGetOutputCount", status);
  }
  if (input_count != 1) {
    fprintf(stderr, "[FAIL] this probe currently supports exactly 1 input tensor, found %zu\n", input_count);
    goto cleanup;
  }
  if (output_count == 0) {
    fprintf(stderr, "[FAIL] model has no outputs\n");
    goto cleanup;
  }

  if (load_tensor_meta(api, session, 0, 1, allocator, &input_meta) != 0) {
    goto cleanup;
  }
  if (load_tensor_meta(api, session, 0, 0, allocator, &output_meta) != 0) {
    goto cleanup;
  }

  fprintf(stdout, "[INFO] input %s type=%s shape=", input_meta.name, tensor_type_name(input_meta.element_type));
  print_dims(stdout, input_meta.dims, input_meta.dim_count);
  fprintf(stdout, " element_count=%zu\n", input_meta.element_count);
  fprintf(stdout, "[INFO] first output %s type=%s shape=", output_meta.name, tensor_type_name(output_meta.element_type));
  print_dims(stdout, output_meta.dims, output_meta.dim_count);
  fprintf(stdout, " element_count=%zu\n", output_meta.element_count);

  status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
  if (status != NULL) {
    return fail_status(api, "CreateCpuMemoryInfo", status);
  }

  if (create_float_input_tensor(api, &input_meta, memory_info, &input_tensor, &input_buffer) != 0) {
    goto cleanup;
  }

  {
    const char* input_names[] = {input_meta.name};
    const OrtValue* input_values[] = {input_tensor};
    const char* output_names[] = {output_meta.name};

    status = api->Run(session, NULL, input_names, input_values, 1, output_names, 1, &output_tensor);
    if (status != NULL) {
      return fail_status(api, "Run", status);
    }
  }

  fprintf(stdout, "[OK] session created successfully\n");
  if (print_first_output_summary(api, output_tensor) != 0) {
    goto cleanup;
  }
  rc = 0;

cleanup:
  if (output_tensor != NULL && api != NULL) {
    api->ReleaseValue(output_tensor);
  }
  if (input_tensor != NULL && api != NULL) {
    api->ReleaseValue(input_tensor);
  }
  free(input_buffer);
  if (memory_info != NULL && api != NULL) {
    api->ReleaseMemoryInfo(memory_info);
  }
  if (session != NULL && api != NULL) {
    api->ReleaseSession(session);
  }
  if (session_options != NULL && api != NULL) {
    api->ReleaseSessionOptions(session_options);
  }
  if (env != NULL && api != NULL) {
    api->ReleaseEnv(env);
  }
  if (api != NULL && allocator != NULL) {
    free_tensor_meta(api, allocator, &input_meta);
    free_tensor_meta(api, allocator, &output_meta);
  }
  free(keys);
  free(values);
  return rc;
}
