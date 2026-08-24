#ifndef SCAMGUARD_GGUF_H
#define SCAMGUARD_GGUF_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32) && defined(SCAMGUARD_GGUF_SHARED)
#if defined(SCAMGUARD_GGUF_BUILD)
#define SG_GGUF_API __declspec(dllexport)
#else
#define SG_GGUF_API __declspec(dllimport)
#endif
#elif defined(__GNUC__) && defined(SCAMGUARD_GGUF_SHARED)
#define SG_GGUF_API __attribute__((visibility("default")))
#else
#define SG_GGUF_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define SG_GGUF_ABI_VERSION 1u
#define SG_GGUF_PROTOCOL_VERSION 3u

typedef struct sg_gguf_runtime sg_gguf_runtime;

typedef enum sg_gguf_status {
    SG_GGUF_OK = 0,
    SG_GGUF_INVALID_ARGUMENT = 1,
    SG_GGUF_MODEL_LOAD_FAILED = 2,
    SG_GGUF_RUNTIME_ERROR = 3,
} sg_gguf_status;

typedef struct sg_gguf_config {
    uint32_t struct_size;
    uint32_t abi_version;
    int32_t context_size;
    int32_t batch_size;
    int32_t micro_batch_size;
    int32_t threads;
    int32_t gpu_layers;
} sg_gguf_config;

typedef struct sg_gguf_runtime_info {
    uint32_t struct_size;
    uint32_t abi_version;
    uint32_t protocol_version;
    uint64_t model_bytes;
    int32_t context_size;
    int32_t prefix_tokens;
} sg_gguf_runtime_info;

typedef struct sg_gguf_score_result {
    uint32_t struct_size;
    uint32_t abi_version;
    double safe_score;
    double uncertain_score;
    double scam_score;
    int64_t elapsed_microseconds;
    int32_t maximum_sequence_tokens;
    int32_t prefix_reused;
    int32_t prefix_tokens;
} sg_gguf_score_result;

SG_GGUF_API void sg_gguf_config_init(sg_gguf_config * config);

SG_GGUF_API void sg_gguf_runtime_info_init(sg_gguf_runtime_info * info);

SG_GGUF_API void sg_gguf_score_result_init(sg_gguf_score_result * result);

SG_GGUF_API sg_gguf_status sg_gguf_runtime_create(
    const char * model_path,
    const char * prefix_utf8,
    size_t prefix_bytes,
    const sg_gguf_config * config,
    sg_gguf_runtime ** runtime,
    char * error_message,
    size_t error_capacity);

SG_GGUF_API sg_gguf_status sg_gguf_runtime_get_info(
    const sg_gguf_runtime * runtime,
    sg_gguf_runtime_info * info,
    char * error_message,
    size_t error_capacity);

SG_GGUF_API sg_gguf_status sg_gguf_runtime_score(
    sg_gguf_runtime * runtime,
    const char * question_utf8,
    size_t question_bytes,
    sg_gguf_score_result * result,
    char * error_message,
    size_t error_capacity);

SG_GGUF_API void sg_gguf_runtime_destroy(sg_gguf_runtime * runtime);

#ifdef __cplusplus
}
#endif

#endif
