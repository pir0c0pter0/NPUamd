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
    size_t i;

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
        fprintf(stderr, "[FAIL] only tensor inputs/outputs are supported\n");
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
        fprintf(stderr, "[FAIL] allocation failed for dims\n");
        api->ReleaseTypeInfo(type_info);
        return 1;
    }

    if (meta->dim_count > 0) {
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

static int read_exact_file(const char* path, void* buffer, size_t bytes) {
    FILE* fp = fopen(path, "rb");
    size_t got = 0;
    if (fp == NULL) {
        perror(path);
        return 1;
    }
    got = fread(buffer, 1, bytes, fp);
    fclose(fp);
    if (got != bytes) {
        fprintf(stderr, "[FAIL] expected %zu bytes in %s, got %zu\n", bytes, path, got);
        return 1;
    }
    return 0;
}

static int write_exact_file(const char* path, const void* buffer, size_t bytes) {
    FILE* fp = fopen(path, "wb");
    size_t wrote = 0;
    if (fp == NULL) {
        perror(path);
        return 1;
    }
    wrote = fwrite(buffer, 1, bytes, fp);
    fclose(fp);
    if (wrote != bytes) {
        fprintf(stderr, "[FAIL] expected to write %zu bytes to %s, wrote %zu\n", bytes, path, wrote);
        return 1;
    }
    return 0;
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
    TensorMeta input_meta;
    TensorMeta output_meta;
    float* input_buffer = NULL;
    float* output_buffer = NULL;
    const char* model_path;
    const char* input_path;
    const char* output_path;
    const char** keys = NULL;
    const char** values = NULL;
    size_t num_pairs = 0;
    size_t input_count = 0;
    size_t output_count = 0;
    int rc = 1;

    reset_tensor_meta(&input_meta);
    reset_tensor_meta(&output_meta);

    if (argc < 4) {
        fprintf(stderr, "usage: %s MODEL_PATH INPUT_RAW OUTPUT_RAW [key=value ...]\n", argv[0]);
        return 2;
    }

    model_path = argv[1];
    input_path = argv[2];
    output_path = argv[3];

    if (argc > 4) {
        int i;
        num_pairs = (size_t)(argc - 4);
        keys = (const char**)calloc(num_pairs, sizeof(*keys));
        values = (const char**)calloc(num_pairs, sizeof(*values));
        if (keys == NULL || values == NULL) {
            fprintf(stderr, "[FAIL] allocation failed for provider options\n");
            goto cleanup;
        }
        for (i = 4; i < argc; ++i) {
            char* pair = argv[i];
            char* eq = strchr(pair, '=');
            if (eq == NULL || eq == pair || eq[1] == '\0') {
                fprintf(stderr, "invalid provider option: %s\n", pair);
                goto cleanup;
            }
            *eq = '\0';
            keys[i - 4] = pair;
            values[i - 4] = eq + 1;
        }
    }

    api_base = OrtGetApiBase();
    api = api_base->GetApi(ORT_API_VERSION);

    status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "whisper-enc-dump", &env);
    if (status != NULL) {
        return fail_status(api, "CreateEnv", status);
    }

    status = api->GetAllocatorWithDefaultOptions(&allocator);
    if (status != NULL) {
        return fail_status(api, "GetAllocator", status);
    }

    status = api->CreateSessionOptions(&session_options);
    if (status != NULL) {
        return fail_status(api, "CreateSessionOptions", status);
    }

    fprintf(stdout, "[INFO] appending VitisAI EP with %zu option(s)\n", num_pairs);
    status = api->SessionOptionsAppendExecutionProvider_VitisAI(session_options, keys, values, num_pairs);
    if (status != NULL) {
        return fail_status(api, "AppendVitisAI", status);
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

    if (input_count != 1 || output_count == 0) {
        fprintf(stderr, "[FAIL] expected exactly 1 input and at least 1 output\n");
        goto cleanup;
    }

    if (load_tensor_meta(api, session, 0, 1, allocator, &input_meta) != 0) {
        goto cleanup;
    }
    if (load_tensor_meta(api, session, 0, 0, allocator, &output_meta) != 0) {
        goto cleanup;
    }

    if (input_meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT ||
        output_meta.element_type != ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT) {
        fprintf(stderr, "[FAIL] only float32 input/output are supported\n");
        goto cleanup;
    }

    input_buffer = (float*)malloc(input_meta.element_count * sizeof(*input_buffer));
    if (input_buffer == NULL) {
        fprintf(stderr, "[FAIL] allocation failed for input buffer\n");
        goto cleanup;
    }
    if (read_exact_file(input_path, input_buffer, input_meta.element_count * sizeof(*input_buffer)) != 0) {
        goto cleanup;
    }

    status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
    if (status != NULL) {
        return fail_status(api, "CreateCpuMemoryInfo", status);
    }

    status = api->CreateTensorWithDataAsOrtValue(memory_info,
                                                 input_buffer,
                                                 input_meta.element_count * sizeof(*input_buffer),
                                                 input_meta.dims,
                                                 input_meta.dim_count,
                                                 input_meta.element_type,
                                                 &input_tensor);
    if (status != NULL) {
        return fail_status(api, "CreateTensorWithDataAsOrtValue", status);
    }

    {
        const char* input_names[] = {input_meta.name};
        const OrtValue* input_values[] = {input_tensor};
        const char* output_names[] = {output_meta.name};
        fprintf(stdout, "[INFO] running encoder inference...\n");
        status = api->Run(session, NULL, input_names, input_values, 1, output_names, 1, &output_tensor);
        if (status != NULL) {
            return fail_status(api, "Run", status);
        }
    }

    status = api->GetTensorMutableData(output_tensor, (void**)&output_buffer);
    if (status != NULL) {
        return fail_status(api, "GetTensorMutableData", status);
    }

    if (write_exact_file(output_path,
                         output_buffer,
                         output_meta.element_count * sizeof(*output_buffer)) != 0) {
        goto cleanup;
    }

    fprintf(stdout, "[OK] wrote %s (%zu float32 values)\n", output_path, output_meta.element_count);
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
