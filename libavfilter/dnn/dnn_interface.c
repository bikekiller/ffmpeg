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

#if 0
static int copy_from_frame_to_dnn(FFBaseInference *ctx, const AVFrame *frame)
{
    int bytewidth = av_image_get_linesize(frame->format, frame->width, 0);
    DNNData *dnn_input = &ctx->input;

    switch (frame->format) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
        if (dnn_input->dt == DNN_FLOAT) {
            sws_scale(ctx->sws_gray8_to_grayf32, (const uint8_t **)frame->data, frame->linesize,
                      0, frame->height, (uint8_t * const*)(&dnn_input->data),
                      (const int [4]){frame->width * 3 * sizeof(float), 0, 0, 0});
        } else {
            av_assert0(dnn_input->dt == DNN_UINT8);
            av_image_copy_plane(dnn_input->data, bytewidth,
                                frame->data[0], frame->linesize[0],
                                bytewidth, frame->height);
        }
        return 0;
    case AV_PIX_FMT_GRAY8:
    case AV_PIX_FMT_GRAYF32:
        av_image_copy_plane(dnn_input->data, bytewidth,
                            frame->data[0], frame->linesize[0],
                            bytewidth, frame->height);
        return 0;
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUV410P:
    case AV_PIX_FMT_YUV411P:
        sws_scale(ctx->sws_gray8_to_grayf32, (const uint8_t **)frame->data, frame->linesize,
                  0, frame->height, (uint8_t * const*)(&dnn_input->data),
                  (const int [4]){frame->width * sizeof(float), 0, 0, 0});
        return 0;
    default:
        return AVERROR(EIO);
    }

    return 0;
}
#endif

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

FFBaseInference *ff_dnn_interface_create(const char *inference_id, FFInferenceParam *param) {
    if (!param)
        return NULL;

    FFBaseInference *base_inference = (FFBaseInference *)av_mallocz(sizeof(*base_inference));
    if (base_inference == NULL)
        return NULL;

    //set_log_function(ff_log_function);
    //set_trace_function(ff_trace_function);

    base_inference->inference_id = inference_id ? av_strdup(inference_id) : NULL;
    base_inference->param = *param;

    // TODO: create image inference backend and stuff for async inference
    //base->inference = (void *)FFInferenceImplCreate(base);

    return base_inference;
}

void ff_dnn_interface_release(FFBaseInference *base) {
    if (!base)
        return;

    //if (base->inference) {
    //    FFInferenceImplRelease((FFInferenceImpl *)base->inference);
    //    base->inference = NULL;
    //}

    if (base->dnn_module)
        (base->dnn_module->free_model)(&base->model);

    av_freep(&base->dnn_module);

    if (base->inference_id) {
        av_free((void *)base->inference_id);
        base->inference_id = NULL;
    }

    av_free(base);
}

int ff_dnn_interface_send_frame(FFBaseInference *base, AVFrame *frame_in) {
    if (!base || !frame_in)
        return AVERROR(EINVAL);

    //return FFInferenceImplAddFrame(ctx, (FFInferenceImpl *)base->inference, frame_in);

    // per-proc and set input blob

#if 0
    copy_from_frame_to_dnn(base, in);
#endif

    // TODO: submit inference request

#if 0
    dnn_result = (ctx->dnn_module->execute_model)(ctx->model, &ctx->output, 1);
    if (dnn_result != DNN_SUCCESS){
        av_log(ctx, AV_LOG_ERROR, "failed to execute model\n");
        av_frame_free(&in);
        return AVERROR(EIO);
    }
#endif

    return 0;
}

int ff_dnn_interface_get_frame(FFBaseInference *base, AVFrame **frame_out) {
    if (!base || !frame_out)
        return AVERROR(EINVAL);

    // TODO:
    //return FFInferenceImplGetFrame(ctx, (FFInferenceImpl *)base->inference, frame_out);
    return 0;
}

int ff_dnn_interface_frame_queue_empty(FFBaseInference *base) {
    if (!base)
        return AVERROR(EINVAL);

    // TODO:
    //return FFInferenceImplGetQueueSize(ctx, (FFInferenceImpl *)base->inference) == 0 ? TRUE : FALSE;
    return 0;
}

int ff_dnn_interface_resource_status(FFBaseInference *base) {
    if (!base)
        return AVERROR(EINVAL);

    // TODO:
    //return FFInferenceImplResourceStatus(ctx, (FFInferenceImpl *)base->inference);
    return 0;
}

void ff_dnn_interface_send_event(FFBaseInference *base, FF_INFERENCE_EVENT event) {
    if (!base)
        return;

    // TODO:
    //FFInferenceImplSinkEvent(ctx, (FFInferenceImpl *)base->inference, event);
    return 0;
}
