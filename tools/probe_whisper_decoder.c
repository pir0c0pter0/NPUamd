/*
 * Probe Whisper decoder through VitisAI EP.
 * Handles 2 inputs: x (int64 [1,448]) + xa (float32 [1,1500,384])
 *
 * Usage: probe_whisper_decoder MODEL_PATH [key=value ...]
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <onnxruntime/core/session/onnxruntime_c_api.h>

static int fail_status(const OrtApi* api, const char* step, OrtStatus* status) {
    const char* message = api->GetErrorMessage(status);
    fprintf(stderr, "[FAIL] %s: %s\n", step, message);
    api->ReleaseStatus(status);
    return 1;
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
    OrtValue* input_x = NULL;
    OrtValue* input_xa = NULL;
    OrtValue* output_tensor = NULL;
    int64_t* x_buffer = NULL;
    float* xa_buffer = NULL;
    const char* model_path;
    const char** keys = NULL;
    const char** values = NULL;
    size_t num_pairs = 0;
    int rc = 1;

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
        for (i = 2; i < argc; ++i) {
            char* pair = argv[i];
            char* eq = strchr(pair, '=');
            if (!eq || eq == pair || eq[1] == '\0') {
                fprintf(stderr, "invalid option: %s\n", pair);
                goto cleanup;
            }
            *eq = '\0';
            keys[i - 2] = pair;
            values[i - 2] = eq + 1;
        }
    }

    api_base = OrtGetApiBase();
    api = api_base->GetApi(ORT_API_VERSION);

    status = api->CreateEnv(ORT_LOGGING_LEVEL_WARNING, "whisper-dec-probe", &env);
    if (status) return fail_status(api, "CreateEnv", status);

    status = api->GetAllocatorWithDefaultOptions(&allocator);
    if (status) return fail_status(api, "GetAllocator", status);

    status = api->CreateSessionOptions(&session_options);
    if (status) return fail_status(api, "CreateSessionOptions", status);

    fprintf(stdout, "[INFO] appending VitisAI EP with %zu option(s)\n", num_pairs);
    status = api->SessionOptionsAppendExecutionProvider_VitisAI(session_options, keys, values, num_pairs);
    if (status) return fail_status(api, "AppendVitisAI", status);

    fprintf(stdout, "[INFO] creating session for %s\n", model_path);
    status = api->CreateSession(env, model_path, session_options, &session);
    if (status) return fail_status(api, "CreateSession", status);

    /* Create x input: int64 [1, 448] */
    {
        int64_t x_shape[] = {1, 448};
        size_t x_count = 448;
        size_t i;

        status = api->CreateCpuMemoryInfo(OrtArenaAllocator, OrtMemTypeDefault, &memory_info);
        if (status) return fail_status(api, "CreateMemInfo", status);

        x_buffer = (int64_t*)calloc(x_count, sizeof(*x_buffer));
        /* Fill with EOT (50256) and SOT at position 0 */
        for (i = 0; i < x_count; ++i) x_buffer[i] = 50256;
        x_buffer[0] = 50257; /* SOT */
        x_buffer[1] = 50258; /* en */
        x_buffer[2] = 50358; /* transcribe */
        x_buffer[3] = 50362; /* notimestamps */

        status = api->CreateTensorWithDataAsOrtValue(
            memory_info, x_buffer, x_count * sizeof(*x_buffer),
            x_shape, 2, ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64, &input_x);
        if (status) return fail_status(api, "CreateTensor_x", status);
    }

    /* Create xa input: float32 [1, 1500, 384] */
    {
        int64_t xa_shape[] = {1, 1500, 384};
        size_t xa_count = 1 * 1500 * 384;
        size_t i;

        xa_buffer = (float*)malloc(xa_count * sizeof(*xa_buffer));
        for (i = 0; i < xa_count; ++i) {
            xa_buffer[i] = (float)((int)(i % 19) - 9) / 30.0f;
        }

        status = api->CreateTensorWithDataAsOrtValue(
            memory_info, xa_buffer, xa_count * sizeof(*xa_buffer),
            xa_shape, 3, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, &input_xa);
        if (status) return fail_status(api, "CreateTensor_xa", status);
    }

    /* Run inference */
    {
        const char* input_names[] = {"x", "xa"};
        const OrtValue* input_values[] = {input_x, input_xa};
        const char* output_names[] = {"matmul"};

        fprintf(stdout, "[INFO] running inference...\n");
        status = api->Run(session, NULL, input_names, input_values, 2, output_names, 1, &output_tensor);
        if (status) return fail_status(api, "Run", status);
    }

    /* Print output summary */
    {
        OrtTensorTypeAndShapeInfo* tensor_info = NULL;
        size_t dim_count = 0;
        int64_t dims[4];
        float* output_data = NULL;

        status = api->GetTensorTypeAndShape(output_tensor, &tensor_info);
        if (status) return fail_status(api, "GetTensorTypeAndShape", status);

        status = api->GetDimensionsCount(tensor_info, &dim_count);
        if (!status) {
            size_t i;
            api->GetDimensions(tensor_info, dims, dim_count);
            fprintf(stdout, "[OK] output shape=[");
            for (i = 0; i < dim_count; ++i) {
                if (i > 0) fprintf(stdout, ", ");
                fprintf(stdout, "%lld", (long long)dims[i]);
            }
            fprintf(stdout, "]\n");
        }
        api->ReleaseTensorTypeAndShapeInfo(tensor_info);

        status = api->GetTensorMutableData(output_tensor, (void**)&output_data);
        if (!status && output_data) {
            fprintf(stdout, "[OK] output[0,0,0]=%f\n", output_data[0]);
        }
    }

    fprintf(stdout, "[OK] decoder inference completed successfully\n");
    rc = 0;

cleanup:
    if (output_tensor) api->ReleaseValue(output_tensor);
    if (input_xa) api->ReleaseValue(input_xa);
    if (input_x) api->ReleaseValue(input_x);
    if (memory_info) api->ReleaseMemoryInfo(memory_info);
    if (session) api->ReleaseSession(session);
    if (session_options) api->ReleaseSessionOptions(session_options);
    if (env) api->ReleaseEnv(env);
    free(x_buffer);
    free(xa_buffer);
    free(keys);
    free(values);
    return rc;
}
