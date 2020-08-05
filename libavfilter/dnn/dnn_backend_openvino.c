/*
 * Copyright (c) 2020
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
 * DNN OpenVINO backend implementation.
 */

#include "dnn_backend_openvino.h"
#include "safe_queue.h"
#include "libavformat/avio.h"
#include "libavutil/avassert.h"
#include <c_api/ie_c_api.h>
#include <pthread.h>

typedef struct RequestContext {
    int out_blob_id;
    ie_infer_request_t *infer_request;
    ie_complete_call_back_t callback;
    InferenceContext *inference_ctx;
} RequestContext;

typedef struct OVModel{
    ie_core_t *core;
    ie_network_t *network;
    ie_executable_network_t *exe_network;
    ie_infer_request_t *infer_request;
    ie_blob_t *input_blob;
    ie_blob_t **output_blobs;
    uint32_t nb_output;

    // async support
    int num_reqs;
    SafeQueueT *request_ctx_q; // queue to hold request context
    ie_infer_request_t **infer_requests;
    pthread_mutex_t callback_mutex;

} OVModel;

static DNNDataType precision_to_datatype(precision_e precision)
{
    switch (precision)
    {
    case FP32:
        return DNN_FLOAT;
    default:
        av_assert0(!"not supported yet.");
        return DNN_FLOAT;
    }
}

static DNNReturnType get_input_blob_ov(void *model, DNNData *input, const char *input_name)
{
    OVModel *ov_model = (OVModel *)model;
    RequestContext *request_ctx = NULL;
    ie_infer_request_t *infer_request = NULL;
    ie_blob_t *input_blob = NULL;
    IEStatusCode status;
    dimensions_t dims;
    precision_e precision;
    ie_blob_buffer_t blob_buffer;

    request_ctx = (RequestContext *)SafeQueuePop(ov_model->request_ctx_q);
    infer_request = request_ctx->infer_request;

    status = ie_infer_request_get_blob(infer_request, input_name, &input_blob);
    if (status != OK)
       goto err;

    status |= ie_blob_get_dims(input_blob, &dims);
    status |= ie_blob_get_precision(input_blob, &precision);
    if (status != OK)
       goto err;

    input->channels = dims.dims[1];
    input->height   = dims.dims[2];
    input->width    = dims.dims[3];
    input->dt       = precision_to_datatype(precision);

    status = ie_blob_get_buffer(input_blob, &blob_buffer);
    if (status != OK)
       goto err;
    input->data = blob_buffer.buffer;
    SafeQueuePushFront(ov_model->request_ctx_q, request_ctx);
    return DNN_SUCCESS;

err:
    SafeQueuePushFront(ov_model->request_ctx_q, request_ctx);
    if (input_blob)
        ie_blob_free(&input_blob);
    if (infer_request)
        ie_infer_request_free(&infer_request);
    return DNN_ERROR;
}

static DNNReturnType get_output_ov(void *model, DNNData *output, const char *output_name)
{
   DNNReturnType ret = DNN_SUCCESS;
   OVModel *ov_model = (OVModel *)model;
   ie_blob_t *output_blob = NULL;
   ie_blob_buffer_t blob_buffer;
   dimensions_t dims;
   precision_e precision;

   IEStatusCode status = ie_infer_request_infer(ov_model->infer_request);
   if (status != OK)
      return DNN_ERROR;

   status = ie_infer_request_get_blob(ov_model->infer_request, output_name, &output_blob);
   if (status != OK) {
      return DNN_ERROR;
   }

   status = ie_blob_get_buffer(output_blob, &blob_buffer);
   if (status != OK) {
      ret = DNN_ERROR;
      goto out;
   }

   status |= ie_blob_get_dims(output_blob, &dims);
   status |= ie_blob_get_precision(output_blob, &precision);
   if (status != OK) {
      ret = DNN_ERROR;
      goto out;
   }

   output->channels = dims.dims[1];
   output->height   = dims.dims[2];
   output->width    = dims.dims[3];
   output->dt       = precision_to_datatype(precision);
   output->data     = blob_buffer.buffer;

out:
   ie_blob_free(&output_blob);
   return ret;
}

static DNNReturnType get_input_ov(void *model, DNNData *input, const char *input_name)
{
    OVModel *ov_model = (OVModel *)model;
    char *model_input_name = NULL;
    IEStatusCode status;
    size_t model_input_count = 0;
    dimensions_t dims;
    precision_e precision;

    status = ie_network_get_inputs_number(ov_model->network, &model_input_count);
    if (status != OK)
        return DNN_ERROR;

    for (size_t i = 0; i < model_input_count; i++) {
        status = ie_network_get_input_name(ov_model->network, i, &model_input_name);
        if (status != OK)
            return DNN_ERROR;
        if (strcmp(model_input_name, input_name) == 0) {
            ie_network_name_free(&model_input_name);
            status |= ie_network_get_input_dims(ov_model->network, input_name, &dims);
            status |= ie_network_get_input_precision(ov_model->network, input_name, &precision);
            if (status != OK)
                return DNN_ERROR;

            // The order of dims in the openvino is fixed and it is always NCHW for 4-D data.
            // while we pass NHWC data from FFmpeg to openvino
            status = ie_network_set_input_layout(ov_model->network, input_name, NHWC);
            if (status != OK)
                return DNN_ERROR;

            input->channels = dims.dims[1];
            input->height   = dims.dims[2];
            input->width    = dims.dims[3];
            input->dt       = precision_to_datatype(precision);
            return DNN_SUCCESS;
        }

        ie_network_name_free(&model_input_name);
    }

    return DNN_ERROR;
}

static DNNReturnType set_input_output_ov(void *model, DNNData *input, const char *input_name, const char **output_names, uint32_t nb_output)
{
    OVModel *ov_model = (OVModel *)model;
    IEStatusCode status;
    dimensions_t dims;
    precision_e precision;
    ie_blob_buffer_t blob_buffer;

    status = ie_exec_network_create_infer_request(ov_model->exe_network, &ov_model->infer_request);
    if (status != OK)
        goto err;

    status = ie_infer_request_get_blob(ov_model->infer_request, input_name, &ov_model->input_blob);
    if (status != OK)
        goto err;

    status |= ie_blob_get_dims(ov_model->input_blob, &dims);
    status |= ie_blob_get_precision(ov_model->input_blob, &precision);
    if (status != OK)
        goto err;

    av_assert0(input->channels == dims.dims[1]);
    av_assert0(input->height   == dims.dims[2]);
    av_assert0(input->width    == dims.dims[3]);
    av_assert0(input->dt       == precision_to_datatype(precision));

    status = ie_blob_get_buffer(ov_model->input_blob, &blob_buffer);
    if (status != OK)
        goto err;
    input->data = blob_buffer.buffer;

    // outputs
    ov_model->nb_output = 0;
    av_freep(&ov_model->output_blobs);
    ov_model->output_blobs = av_mallocz_array(nb_output, sizeof(*ov_model->output_blobs));
    if (!ov_model->output_blobs)
        goto err;

    for (int i = 0; i < nb_output; i++) {
        const char *output_name = output_names[i];
        status = ie_infer_request_get_blob(ov_model->infer_request, output_name, &(ov_model->output_blobs[i]));
        if (status != OK)
            goto err;
        ov_model->nb_output++;
    }

    return DNN_SUCCESS;

err:
    if (ov_model->output_blobs) {
        for (uint32_t i = 0; i < ov_model->nb_output; i++) {
            ie_blob_free(&(ov_model->output_blobs[i]));
        }
        av_freep(&ov_model->output_blobs);
    }
    if (ov_model->input_blob)
        ie_blob_free(&ov_model->input_blob);
    if (ov_model->infer_request)
        ie_infer_request_free(&ov_model->infer_request);
    return DNN_ERROR;
}

DNNModel *ff_dnn_load_model_ov(const char *model_filename, const char *options)
{
    DNNModel *model = NULL;
    OVModel *ov_model = NULL;
    IEStatusCode status;
    ie_config_t config = {NULL, NULL, NULL};

    model = av_malloc(sizeof(DNNModel));
    if (!model){
        return NULL;
    }

    ov_model = av_mallocz(sizeof(OVModel));
    if (!ov_model)
        goto err;

    status = ie_core_create("", &ov_model->core);
    if (status != OK)
        goto err;

    status = ie_core_read_network(ov_model->core, model_filename, NULL, &ov_model->network);
    if (status != OK)
        goto err;

    status = ie_core_load_network(ov_model->core, ov_model->network, "CPU", &config, &ov_model->exe_network);
    if (status != OK)
        goto err;


    int nireq = 8; // TODO: pass in as parameter
    ov_model->infer_requests = (ie_infer_request_t **)malloc(nireq * sizeof(ie_infer_request_t *));
    if (!ov_model->infer_requests) {
        goto err;
    }
    ov_model->num_reqs = nireq;
    for (size_t i = 0; i < ov_model->num_reqs; ++i) {
        ie_exec_network_create_infer_request(ov_model->exe_network, &ov_model->infer_requests[i]);
        if (!ov_model->infer_requests[i]) {
            goto err;
        }
    }

    ov_model->request_ctx_q = SafeQueueCreate();
    if (!ov_model->request_ctx_q) {
        goto err;
    }

    for (size_t n = 0; n < ov_model->num_reqs; ++n) {
        RequestContext *request_ctx = (RequestContext *)malloc(sizeof(*request_ctx));
        if (!request_ctx)
            goto err;
        memset(request_ctx, 0, sizeof(*request_ctx));
        request_ctx->infer_request = ov_model->infer_requests[n];
        SafeQueuePush(ov_model->request_ctx_q, request_ctx);
    }

    pthread_mutex_init(&ov_model->callback_mutex, NULL);

    model->model = (void *)ov_model;
    model->options = options;
    model->set_input_output = &set_input_output_ov;
    model->get_input = &get_input_ov;
    model->get_output = &get_output_ov;
    model->get_input_blob = &get_input_blob_ov;

    return model;

err:
    if (model)
        av_freep(&model);
    if (ov_model) {
        if (ov_model->infer_requests) {
           for (size_t i = 0; i < ov_model->num_reqs; ++i)
              if (ov_model->infer_requests[i])
                 ie_infer_request_free(&ov_model->infer_requests[i]);
           free(ov_model->infer_requests);
           ov_model->num_reqs = 0;
        }
        if (ov_model->request_ctx_q)
           SafeQueueDestroy(ov_model->request_ctx_q);
        if (ov_model->exe_network)
            ie_exec_network_free(&ov_model->exe_network);
        if (ov_model->network)
            ie_network_free(&ov_model->network);
        if (ov_model->core)
            ie_core_free(&ov_model->core);
        av_freep(&ov_model);
    }
    return NULL;
}

static void completion_callback(void *args) {

    RequestContext *request = (RequestContext *)args;
    uint32_t out_blob_id = request->out_blob_id;
    InferenceContext *inference_ctx = request->inference_ctx;
    ProcessingFrame *processing_frame = inference_ctx->processing_frame;
    FFBaseInference *base = inference_ctx->base;
    DNNModel *model = (DNNModel*)base->model;
    OVModel *ov_model = (OVModel*)model->model;
    dimensions_t dims;
    precision_e precision;
    size_t num_outputs;
    IEStatusCode status;


    //VAII_DEBUG(__FUNCTION__);

    pthread_mutex_lock(&ov_model->callback_mutex);

    // output blob to DNNData 
    ie_network_get_outputs_number(ov_model->network, &num_outputs);

    ie_blob_t *out_blob = NULL;
    char *output_name = NULL;
    ie_network_get_output_name(ov_model->network, out_blob_id, &output_name);
    ie_infer_request_get_blob(request->infer_request, output_name, &out_blob);
    free(output_name);

    if (!out_blob) {
       av_log(NULL, AV_LOG_ERROR, "failed to get out blob\n");
       goto out;
    }

    ie_blob_buffer_t blob_buffer;
    status = ie_blob_get_buffer(out_blob, &blob_buffer);
    if (status != OK) {
        av_log(NULL, AV_LOG_ERROR, "failed to get buffer\n");
        goto out;
    }

    status |= ie_blob_get_dims(out_blob, &dims);
    status |= ie_blob_get_precision(out_blob, &precision);
    if (status != OK) {
        av_log(NULL, AV_LOG_ERROR, "failed to get precision\n");
    }

    DNNData output;
    output.channels = dims.dims[1];
    output.height   = dims.dims[2];
    output.width    = dims.dims[3];
    output.dt       = precision_to_datatype(precision);
    output.data     = blob_buffer.buffer;

    ((InferCallback)inference_ctx->cb)(&output, processing_frame, base);

    SafeQueuePush(ov_model->request_ctx_q, request);

out:
    if (out_blob)
        ie_blob_free(&out_blob);

    pthread_mutex_unlock(&ov_model->callback_mutex);

    //VAII_DEBUG("EXIT");
}

DNNReturnType ff_dnn_execute_model_async_ov(const DNNModel *model, InferenceContext *inference_ctx, int out_blob_id)
{
    OVModel *ov_model = (OVModel *)model->model;
    RequestContext *request_ctx = (RequestContext *)SafeQueuePop(ov_model->request_ctx_q);

    request_ctx->callback.completeCallBackFunc = completion_callback;
    request_ctx->callback.args = request_ctx;
    request_ctx->inference_ctx = inference_ctx;
    request_ctx->out_blob_id = out_blob_id;

    ie_infer_set_completion_callback(request_ctx->infer_request, &request_ctx->callback);
    ie_infer_request_infer_async(request_ctx->infer_request);

    return DNN_SUCCESS;
}

DNNReturnType ff_dnn_execute_model_ov(const DNNModel *model, DNNData *outputs, uint32_t nb_output)
{
    dimensions_t dims;
    precision_e precision;
    ie_blob_buffer_t blob_buffer;
    OVModel *ov_model = (OVModel *)model->model;
    uint32_t nb = FFMIN(nb_output, ov_model->nb_output);
    IEStatusCode status = ie_infer_request_infer(ov_model->infer_request);
    if (status != OK)
        return DNN_ERROR;

    for (uint32_t i = 0; i < nb; ++i) {
        status = ie_blob_get_buffer(ov_model->output_blobs[i], &blob_buffer);
        if (status != OK)
            return DNN_ERROR;

        status |= ie_blob_get_dims(ov_model->output_blobs[i], &dims);
        status |= ie_blob_get_precision(ov_model->output_blobs[i], &precision);
        if (status != OK)
            return DNN_ERROR;

        outputs[i].channels = dims.dims[1];
        outputs[i].height   = dims.dims[2];
        outputs[i].width    = dims.dims[3];
        outputs[i].dt       = precision_to_datatype(precision);
        outputs[i].data     = blob_buffer.buffer;
    }

    return DNN_SUCCESS;
}

void ff_dnn_free_model_ov(DNNModel **model)
{
    if (*model){
        OVModel *ov_model = (OVModel *)(*model)->model;
        if (ov_model->output_blobs) {
            for (uint32_t i = 0; i < ov_model->nb_output; i++) {
                ie_blob_free(&(ov_model->output_blobs[i]));
            }
            av_freep(&ov_model->output_blobs);
        }

        if (ov_model->input_blob)
            ie_blob_free(&ov_model->input_blob);
        if (ov_model->infer_request)
            ie_infer_request_free(&ov_model->infer_request);
        if (ov_model->infer_requests) {
           for (size_t i = 0; i < ov_model->num_reqs; ++i)
              if (ov_model->infer_requests[i])
                 ie_infer_request_free(&ov_model->infer_requests[i]);
           free(ov_model->infer_requests);
           ov_model->num_reqs = 0;
        }
        if (ov_model->request_ctx_q)
           SafeQueueDestroy(ov_model->request_ctx_q);
        pthread_mutex_destroy(&ov_model->callback_mutex);
        if (ov_model->exe_network)
            ie_exec_network_free(&ov_model->exe_network);
        if (ov_model->network)
            ie_network_free(&ov_model->network);
        if (ov_model->core)
            ie_core_free(&ov_model->core);
        av_freep(&ov_model);
        av_freep(model);
    }
}
