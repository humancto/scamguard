#include "scamguard_gguf.h"

#include <stddef.h>

int main(void) {
    sg_gguf_config config;
    sg_gguf_runtime_info info;
    sg_gguf_score_result score;

    sg_gguf_config_init(&config);
    sg_gguf_runtime_info_init(&info);
    sg_gguf_score_result_init(&score);

    if (config.struct_size != sizeof(config) || config.abi_version != SG_GGUF_ABI_VERSION) return 1;
    if (info.struct_size != sizeof(info) || info.abi_version != SG_GGUF_ABI_VERSION) return 2;
    if (score.struct_size != sizeof(score) || score.abi_version != SG_GGUF_ABI_VERSION) return 3;
    if (config.context_size != 640 || config.batch_size != 640) return 4;

    sg_gguf_runtime_destroy(NULL);
    return 0;
}
