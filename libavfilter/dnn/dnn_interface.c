/*
 * Copyright (c) 2018 Sergey Lavrushkin
 *
 * This file is part of FFmpeg.
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with FFmpeg; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Implements DNN module initialization with specified backend.
 */

#include "../dnn_interface.h"
#include "dnn_backend_native.h"
#include "dnn_backend_tf.h"
#include "dnn_backend_openvino.h"
#include "libavutil/mem.h"
#include "libavutil/avassert.h"
#include "libavutil/imgutils.h"
#include "libswscale/swscale.h"
#include <pthread.h>

DNNModule *ff_get_dnn_module(DNNBackendType backend_type)
{
    DNNModule *dnn_module;

    dnn_module = av_malloc(sizeof(DNNModule));
    if(!dnn_module){
        return NULL;
    }

    switch(backend_type){
    case DNN_NATIVE:
        dnn_module->load_model = &ff_dnn_load_model_native;
        dnn_module->execute_model = &ff_dnn_execute_model_native;
        dnn_module->free_model = &ff_dnn_free_model_native;
        break;
    case DNN_TF:
    #if (CONFIG_LIBTENSORFLOW == 1)
        dnn_module->load_model = &ff_dnn_load_model_tf;
        dnn_module->execute_model = &ff_dnn_execute_model_tf;
        dnn_module->free_model = &ff_dnn_free_model_tf;
    #else
        av_freep(&dnn_module);
        return NULL;
    #endif
        break;
    case DNN_OV:
    #if (CONFIG_LIBOPENVINO == 1)
        dnn_module->load_model = &ff_dnn_load_model_ov;
        dnn_module->execute_model = &ff_dnn_execute_model_ov;
        dnn_module->execute_model_async = &ff_dnn_execute_model_async_ov;
        dnn_module->free_model = &ff_dnn_free_model_ov;
    #else
        av_freep(&dnn_module);
        return NULL;
    #endif
        break;
    default:
        av_log(NULL, AV_LOG_ERROR, "Module backend_type is not native or tensorflow\n");
        av_freep(&dnn_module);
        return NULL;
    }

    return dnn_module;
}

FFBaseInference *ff_dnn_interface_create(const char *inference_id, FFInferenceParam *param, AVFilterContext *filter_ctx) {

    if (!param)
        return NULL;

    FFBaseInference *base_inference = (FFBaseInference *)av_mallocz(sizeof(*base_inference));
    if (base_inference == NULL)
        return NULL;

    base_inference->param = *param;
    base_inference->dnn_module = ff_get_dnn_module(base_inference->param.backend_type);

    if (!base_inference->dnn_module) {
        av_log(filter_ctx, AV_LOG_ERROR, "could not create DNN module for requested backend\n");
        goto err;
    }

    if (!base_inference->dnn_module->load_model) {
        av_log(filter_ctx, AV_LOG_ERROR, "load_model for network is not specified\n");
        goto err;
    }

    base_inference->model = (base_inference->dnn_module->load_model)(param->model_filename);
    if (!base_inference->model) {
        av_log(filter_ctx, AV_LOG_ERROR, "could not load DNN model\n");
        goto err;
    }

    base_inference->inference_id = inference_id ? av_strdup(inference_id) : NULL;
    base_inference->processing_frames = ff_list_alloc();
    av_assert0(base_inference->processing_frames);
    base_inference->processed_frames = ff_list_alloc();
    av_assert0(base_inference->processed_frames);
    base_inference->filter_ctx = filter_ctx;
    pthread_mutex_init(&base_inference->output_frames_mutex, NULL);

    return base_inference;

err:
    av_freep(&base_inference->dnn_module);
    av_free(base_inference);

    return NULL;
}

void ff_dnn_interface_set_pre_proc(FFBaseInference *base, DNNPreProc pre_proc)
{
    if (!base)
        return;

    base->pre_proc = pre_proc;

    return;
}

void ff_dnn_interface_set_post_proc(FFBaseInference *base, DNNPostProc post_proc)
{
    if (!base)
        return;

    base->post_proc = post_proc;

    return;
}

void ff_dnn_interface_release(FFBaseInference *base) {
    if (!base)
        return;

    //if (base->inference) {
    //    FFInferenceImplRelease((FFInferenceImpl *)base->inference);
    //    base->inference = NULL;
    //}
    ff_list_free(base->processing_frames);
    ff_list_free(base->processed_frames);
    pthread_mutex_destroy(&base->output_frames_mutex);

    if (base->dnn_module)
        (base->dnn_module->free_model)(&base->model);

    av_freep(&base->dnn_module);

    if (base->inference_id) {
        av_free((void *)base->inference_id);
        base->inference_id = NULL;
    }

    av_free(base);
}

static inline void PushOutput(FFBaseInference *base) {
    ff_list_t *processing_frames = base->processing_frames;
    ff_list_t *processed = base->processed_frames;

    while (!processing_frames->empty(processing_frames)) {
        ProcessingFrame *front = (ProcessingFrame *)processing_frames->front(processing_frames);
        if (!front->inference_done) {
            break; // inference not completed yet
        }
        processed->push_back(processed, front->frame_out);
        processing_frames->pop_front(processing_frames);
        av_free(front);
    }
}

static void InferenceCompletionCallback(DNNData *model_output, ProcessingFrame *processing_frame, FFBaseInference *base) {

    if (!base->post_proc || !processing_frame || !base) {
        av_log(NULL, AV_LOG_ERROR, "invalid parameter\n");
        return;
    }

    // post proc: DNNData to AVFrame, 
    if (((DNNPostProc)base->post_proc)(model_output, processing_frame->frame_in, &processing_frame->frame_out, base) != 0) {
       av_log(NULL, AV_LOG_ERROR, "post_proc failed\n");
       return;
    }
    processing_frame->inference_done = 1;

    pthread_mutex_lock(&base->output_frames_mutex);
    PushOutput(base);
    pthread_mutex_unlock(&base->output_frames_mutex);

    return;
}

int ff_dnn_interface_send_frame(FFBaseInference *base, AVFrame *frame_in) {

    DNNReturnType dnn_result;

    if (!base || !frame_in)
        return AVERROR(EINVAL);

    // preproc
    DNNData input_blob;
    int ret = (base->model->get_input_blob)(base->model->model, &input_blob, base->param.model_inputname);

    if(!base->pre_proc) {
        av_log(NULL, AV_LOG_ERROR, "pre_proc function not specified\n");
        return AVERROR(EINVAL);
    }

    ((DNNPreProc)(base->pre_proc))(frame_in, &input_blob, base);

    // inference
    if (base->dnn_module->execute_model_async) {

       // push into processing_frames queue
       pthread_mutex_lock(&base->output_frames_mutex);
       ProcessingFrame *processing_frame = (ProcessingFrame *)av_malloc(sizeof(ProcessingFrame));
       if (processing_frame == NULL) {
          pthread_mutex_unlock(&base->output_frames_mutex);
          return AVERROR(EINVAL);
       }
       processing_frame->frame_in = frame_in;
       processing_frame->frame_out = NULL;
       processing_frame->inference_done = 0;
       base->processing_frames->push_back(base->processing_frames, processing_frame);
       pthread_mutex_unlock(&base->output_frames_mutex);

       InferenceContext *inference_ctx = (InferenceContext *)malloc(sizeof(InferenceContext));
       inference_ctx->processing_frame = processing_frame;
       inference_ctx->cb = InferenceCompletionCallback;
       inference_ctx->base = base;

       dnn_result = (base->dnn_module->execute_model_async)(base->model, inference_ctx, 0 /*FIXME: specify by filter option */);

    } else {
       dnn_result = (base->dnn_module->execute_model)(base->model, &base->output, 1);
    }

    if (dnn_result != DNN_SUCCESS){
        av_log(NULL, AV_LOG_ERROR, "failed to execute model asynchronously\n");
        av_frame_free(&frame_in); // FIXME: do we need to free the input frame?
        return AVERROR(EIO);
    }

    // post proc for sync inference
    if (!base->dnn_module->execute_model_async) {

       AVFrame *frame_out = NULL;
       if (((DNNPostProc)base->post_proc)(&base->output, frame_in, &frame_out, base) != 0) {
          return -1;
       }
       base->processed_frames->push_back(base->processed_frames, frame_out);
    }

    return 0;
}

int ff_dnn_interface_get_frame(FFBaseInference *base, AVFrame **frame_out) {
    ff_list_t *l = base->processed_frames;

    if (l->empty(l) || !frame_out)
        return AVERROR(EAGAIN);

    pthread_mutex_lock(&base->output_frames_mutex);
    *frame_out = (AVFrame *)l->front(l);
    l->pop_front(l);
    pthread_mutex_unlock(&base->output_frames_mutex);

    return 0;
}

int ff_dnn_interface_frame_queue_empty(FFBaseInference *base) {
    if (!base)
        return AVERROR(EINVAL);

    ff_list_t *pro = base->processed_frames;

    if (base->dnn_module->execute_model_async) {
       ff_list_t *out = base->processing_frames;
       av_log(NULL, AV_LOG_INFO, "output:%zu processed:%zu\n", out->size(out), pro->size(pro));
       return out->size(out) + pro->size(pro) == 0 ? 1 : 0;
    } else {
       av_log(NULL, AV_LOG_INFO, "processed:%zu\n", pro->size(pro));
       return pro->size(pro) == 0 ? 1 : 0;
    }
}
