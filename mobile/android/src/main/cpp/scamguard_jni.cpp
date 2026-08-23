#include "scamguard_gguf.h"

#include <jni.h>

#include <cstdint>
#include <string>

namespace {

void throw_java(JNIEnv * env, const char * class_name, const char * message) {
    jclass exception = env->FindClass(class_name);
    if (exception != nullptr) env->ThrowNew(exception, message);
}

std::string bytes(JNIEnv * env, jbyteArray value) {
    if (value == nullptr) return {};
    const jsize length = env->GetArrayLength(value);
    std::string result(static_cast<size_t>(length), '\0');
    if (length > 0) env->GetByteArrayRegion(value, 0, length, reinterpret_cast<jbyte *>(result.data()));
    return result;
}

sg_gguf_runtime * checked_runtime(JNIEnv * env, jlong handle) {
    if (handle == 0) {
        throw_java(env, "java/lang/IllegalStateException", "ScamGuard runtime is closed");
        return nullptr;
    }
    return reinterpret_cast<sg_gguf_runtime *>(static_cast<intptr_t>(handle));
}

}  // namespace

extern "C" JNIEXPORT jlong JNICALL
Java_com_scamguard_runtime_ScamGuardNative_nativeCreate(
    JNIEnv * env,
    jobject,
    jbyteArray model_path_utf8,
    jbyteArray prefix_utf8,
    jint context_size,
    jint batch_size,
    jint micro_batch_size,
    jint threads,
    jint gpu_layers) {
    const std::string model_path = bytes(env, model_path_utf8);
    const std::string prefix = bytes(env, prefix_utf8);
    sg_gguf_config config;
    sg_gguf_config_init(&config);
    config.context_size = context_size;
    config.batch_size = batch_size;
    config.micro_batch_size = micro_batch_size;
    config.threads = threads;
    config.gpu_layers = gpu_layers;
    char error[512] = {};
    sg_gguf_runtime * runtime = nullptr;
    const sg_gguf_status status = sg_gguf_runtime_create(
        model_path.c_str(), prefix.data(), prefix.size(), &config, &runtime, error, sizeof(error));
    if (status != SG_GGUF_OK) {
        throw_java(env, "java/lang/IllegalStateException", error);
        return 0;
    }
    return static_cast<jlong>(reinterpret_cast<intptr_t>(runtime));
}

extern "C" JNIEXPORT jlongArray JNICALL
Java_com_scamguard_runtime_ScamGuardNative_nativeInfo(JNIEnv * env, jobject, jlong handle) {
    sg_gguf_runtime * runtime = checked_runtime(env, handle);
    if (runtime == nullptr) return nullptr;
    sg_gguf_runtime_info info;
    sg_gguf_runtime_info_init(&info);
    char error[512] = {};
    if (sg_gguf_runtime_get_info(runtime, &info, error, sizeof(error)) != SG_GGUF_OK) {
        throw_java(env, "java/lang/IllegalStateException", error);
        return nullptr;
    }
    const jlong values[] = {
        static_cast<jlong>(info.protocol_version),
        static_cast<jlong>(info.model_bytes),
        static_cast<jlong>(info.context_size),
        static_cast<jlong>(info.prefix_tokens),
    };
    jlongArray result = env->NewLongArray(4);
    if (result != nullptr) env->SetLongArrayRegion(result, 0, 4, values);
    return result;
}

extern "C" JNIEXPORT jdoubleArray JNICALL
Java_com_scamguard_runtime_ScamGuardNative_nativeScore(
    JNIEnv * env, jobject, jlong handle, jbyteArray question_utf8) {
    sg_gguf_runtime * runtime = checked_runtime(env, handle);
    if (runtime == nullptr) return nullptr;
    const std::string question = bytes(env, question_utf8);
    sg_gguf_score_result score;
    sg_gguf_score_result_init(&score);
    char error[512] = {};
    if (sg_gguf_runtime_score(
            runtime, question.data(), question.size(), &score, error, sizeof(error)) != SG_GGUF_OK) {
        throw_java(env, "java/lang/IllegalArgumentException", error);
        return nullptr;
    }
    const jdouble values[] = {
        score.safe_score,
        score.uncertain_score,
        score.scam_score,
        static_cast<jdouble>(score.elapsed_microseconds),
        static_cast<jdouble>(score.maximum_sequence_tokens),
        static_cast<jdouble>(score.prefix_reused),
        static_cast<jdouble>(score.prefix_tokens),
    };
    jdoubleArray result = env->NewDoubleArray(7);
    if (result != nullptr) env->SetDoubleArrayRegion(result, 0, 7, values);
    return result;
}

extern "C" JNIEXPORT void JNICALL
Java_com_scamguard_runtime_ScamGuardNative_nativeDestroy(JNIEnv *, jobject, jlong handle) {
    if (handle != 0) {
        sg_gguf_runtime_destroy(
            reinterpret_cast<sg_gguf_runtime *>(static_cast<intptr_t>(handle)));
    }
}
