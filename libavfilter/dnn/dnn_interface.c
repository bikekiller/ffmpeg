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

DnnInterface *dnn_interface_create(const char *inference_id, InferenceParam *param, AVFilterContext *filter_ctx) {

    if (!param)
        return NULL;

    DnnInterface *dnn_interface = (DnnInterface *)av_mallocz(sizeof(*dnn_interface));
    if (dnn_interface == NULL)
        return NULL;

    dnn_interface->param = *param;

    dnn_interface->dnn_module = ff_get_dnn_module(dnn_interface->param.backend_type);
    if (!dnn_interface->dnn_module) {
        av_log(filter_ctx, AV_LOG_ERROR, "could not create DNN module for requested backend\n");
        goto err;
    }

    if (!dnn_interface->dnn_module->load_model) {
        av_log(filter_ctx, AV_LOG_ERROR, "load_model for network is not specified\n");
        goto err;
    }

    dnn_interface->model = (dnn_interface->dnn_module->load_model)(param->model_filename);
    if (!dnn_interface->model) {
        av_log(filter_ctx, AV_LOG_ERROR, "could not load DNN model\n");
        goto err;
    }

    dnn_interface->async_run = dnn_interface->dnn_module->execute_model_async && dnn_interface->param.async;
    dnn_interface->inference_id = inference_id ? av_strdup(inference_id) : NULL;
    dnn_interface->processing_frames = ff_list_alloc();
    av_assert0(dnn_interface->processing_frames);
    dnn_interface->processed_frames = ff_list_alloc();
    av_assert0(dnn_interface->processed_frames);
    dnn_interface->filter_ctx = filter_ctx;
    pthread_mutex_init(&dnn_interface->frame_q_mutex, NULL);

    return dnn_interface;

err:
    av_freep(&dnn_interface->dnn_module);
    av_free(dnn_interface);

    return NULL;
}

void dnn_interface_set_pre_proc(DnnInterface *dnn_interface, DNNPreProc pre_proc)
{
    if (!dnn_interface)
        return;

    dnn_interface->pre_proc = pre_proc;

    return;
}

void dnn_interface_set_post_proc(DnnInterface *dnn_interface, DNNPostProc post_proc)
{
    if (!dnn_interface)
        return;

    dnn_interface->post_proc = post_proc;

    return;
}

void dnn_interface_release(DnnInterface *dnn_interface) {
    if (!dnn_interface)
        return;

    ff_list_free(dnn_interface->processing_frames);
    ff_list_free(dnn_interface->processed_frames);
    pthread_mutex_destroy(&dnn_interface->frame_q_mutex);

    if (dnn_interface->dnn_module)
        (dnn_interface->dnn_module->free_model)(&dnn_interface->model);

    av_freep(&dnn_interface->dnn_module);

    if (dnn_interface->inference_id) {
        av_free((void *)dnn_interface->inference_id);
        dnn_interface->inference_id = NULL;
    }

    av_free(dnn_interface);
}

static inline void PushOutput(DnnInterface *dnn_interface) {
    ff_list_t *processing_frames = dnn_interface->processing_frames;
    ff_list_t *processed = dnn_interface->processed_frames;

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

static void InferenceCompletionCallback(DNNData *model_output, ProcessingFrame *processing_frame, DnnInterface *dnn_interface) {

    if (!dnn_interface->post_proc || !processing_frame || !dnn_interface) {
        av_log(NULL, AV_LOG_ERROR, "invalid parameter\n");
        return;
    }

    // post proc: DNNData to AVFrame, 
    if (((DNNPostProc)dnn_interface->post_proc)(model_output, processing_frame->frame_in, &processing_frame->frame_out, dnn_interface) != 0) {
       av_log(NULL, AV_LOG_ERROR, "post_proc failed\n");
       return;
    }

    processing_frame->inference_done = 1;

    pthread_mutex_lock(&dnn_interface->frame_q_mutex);
    PushOutput(dnn_interface);
    pthread_mutex_unlock(&dnn_interface->frame_q_mutex);

    return;
}

int dnn_interface_send_frame(DnnInterface *dnn_interface, AVFrame *frame_in) {

    DNNReturnType dnn_result;
    DNNData model_output;

    if (!dnn_interface || !frame_in)
        return AVERROR(EINVAL);

    // preproc
    DNNData input_blob;

    (dnn_interface->model->get_input_blob)(dnn_interface->model->model, &input_blob, dnn_interface->param.model_inputname);

    if(!dnn_interface->pre_proc) {
        av_log(NULL, AV_LOG_ERROR, "pre_proc function not specified\n");
        return AVERROR(EINVAL);
    }

    ((DNNPreProc)(dnn_interface->pre_proc))(frame_in, &input_blob, dnn_interface);

    // inference
    if (dnn_interface->async_run) {

       // push into processing_frames queue
       pthread_mutex_lock(&dnn_interface->frame_q_mutex);
       ProcessingFrame *processing_frame = (ProcessingFrame *)av_malloc(sizeof(ProcessingFrame)); // release in PushOutput()
       if (processing_frame == NULL) {
          pthread_mutex_unlock(&dnn_interface->frame_q_mutex);
          return AVERROR(EINVAL);
       }
       processing_frame->frame_in = frame_in;
       processing_frame->frame_out = NULL;
       processing_frame->inference_done = 0;
       dnn_interface->processing_frames->push_back(dnn_interface->processing_frames, processing_frame);
       pthread_mutex_unlock(&dnn_interface->frame_q_mutex);

       InferenceContext *inference_ctx = (InferenceContext *)av_malloc(sizeof(InferenceContext)); // release in model callback
       inference_ctx->processing_frame = processing_frame;
       inference_ctx->cb = InferenceCompletionCallback;
       inference_ctx->dnn_interface = dnn_interface;

       dnn_result = (dnn_interface->dnn_module->execute_model_async)(dnn_interface->model, inference_ctx, dnn_interface->param.model_outputname);

    } else {
       dnn_result = (dnn_interface->dnn_module->execute_model)(dnn_interface->model, &model_output, 1);
    }

    if (dnn_result != DNN_SUCCESS){
        av_log(NULL, AV_LOG_ERROR, "failed to execute model\n");
        return AVERROR(EIO);
    }

    if (!dnn_interface->async_run) {
       AVFrame *frame_out;

       if (((DNNPostProc)dnn_interface->post_proc)(&model_output, frame_in, &frame_out, dnn_interface) != 0) {
          return AVERROR(EIO);
       }

       dnn_interface->processed_frames->push_back(dnn_interface->processed_frames, frame_out);
    }

    return 0;
}

int dnn_interface_get_frame(DnnInterface *dnn_interface, AVFrame **frame_out) {
    ff_list_t *l = dnn_interface->processed_frames;

    if (l->empty(l) || !frame_out)
        return AVERROR(EAGAIN);

    pthread_mutex_lock(&dnn_interface->frame_q_mutex);
    *frame_out = (AVFrame *)l->front(l);
    l->pop_front(l);
    pthread_mutex_unlock(&dnn_interface->frame_q_mutex);

    return 0;
}

int dnn_interface_frame_queue_empty(DnnInterface *dnn_interface) {
    if (!dnn_interface)
        return AVERROR(EINVAL);

    ff_list_t *pro = dnn_interface->processed_frames;

    if (dnn_interface->async_run) {
       ff_list_t *out = dnn_interface->processing_frames;
       av_log(NULL, AV_LOG_INFO, "output:%zu processed:%zu\n", out->size(out), pro->size(pro));
       return out->size(out) + pro->size(pro) == 0 ? 1 : 0;
    } else {
       av_log(NULL, AV_LOG_INFO, "processed:%zu\n", pro->size(pro));
       return pro->size(pro) == 0 ? 1 : 0;
    }
}
