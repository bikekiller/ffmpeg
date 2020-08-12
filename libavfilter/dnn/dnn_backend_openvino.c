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

#define DEFAULT_BATCH_SIZE (4)
typedef struct RequestContext {
    char *blob_name;
    ie_infer_request_t *infer_request;
    ie_complete_call_back_t callback;
    InferenceContext *inference_ctx;
    ProcessingFrame *processing_frame;
    ProcessingFrame **processing_frame_array;
    int num_processing_frames;
    const DNNModel *model;
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
    pthread_mutex_t frame_q_mutex;
    ff_list_t *processing_frames;
    ff_list_t *processed_frames;

    int num_reqs;
    SafeQueueT *request_ctx_q; // queue to hold request context
    ie_infer_request_t **infer_requests;
    pthread_mutex_t callback_mutex;

    void *user_data;

    int batch_size;
} OVModel;

static DNNReturnType _ff_dnn_execute_model_ov(OVModel *model, DNNData *output, const char *output_name);
static DNNReturnType get_output_ov(void *model, DNNData *output, const char *output_name);
static void _new_blob_by_batch_idx(OVModel *ov_model, DNNData *input_blob, DNNData *new_blob, const int batch_idx);

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

    input->batch_size = dims.dims[0];
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

static DNNReturnType _get_input_blob_ov(void *model, DNNData *input, const char *input_name)
{
   OVModel *ov_model = (OVModel *)model;
   IEStatusCode status;
   dimensions_t dims;
   precision_e precision;
   ie_blob_buffer_t blob_buffer;

   av_assert0(ov_model->infer_request);

   status = ie_infer_request_get_blob(ov_model->infer_request, input_name, &ov_model->input_blob);
   if (status != OK)
      goto err;

   status |= ie_blob_get_dims(ov_model->input_blob, &dims);
   status |= ie_blob_get_precision(ov_model->input_blob, &precision);
   if (status != OK)
      goto err;

   input->channels = dims.dims[1];
   input->height   = dims.dims[2];
   input->width    = dims.dims[3];
   input->dt       = precision_to_datatype(precision);

   status = ie_blob_get_buffer(ov_model->input_blob, &blob_buffer);
   if (status != OK)
      goto err;

   input->data = blob_buffer.buffer;

   return DNN_SUCCESS;

err:
   if (ov_model->infer_request)
      ie_infer_request_free(&ov_model->infer_request);

   return DNN_ERROR;
}


static DNNReturnType set_input_output_ov(void *model, DNNData *input, const char *input_name, const char **output_names, uint32_t nb_output)
{
    OVModel *ov_model = (OVModel *)model;
    IEStatusCode status;
    dimensions_t dims;
    precision_e precision;
    ie_blob_buffer_t blob_buffer;

    av_assert0(ov_model->infer_request);

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
    input_shapes_t network_input_shapes;
    int batch_size;
    int nireq;

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

    batch_size = DEFAULT_BATCH_SIZE; // FIXME: get the batch size from user option
    status = ie_network_get_input_shapes(ov_model->network, &network_input_shapes);
    if (status != OK)
        goto err;

    network_input_shapes.shapes[0].shape.dims[0] = batch_size;
    status = ie_network_reshape(ov_model->network, network_input_shapes);
    if (status != OK)
        goto err;

    //if (batch_size > 1 && network_input_shapes.shapes) {
    //    for (int i = 0; i < network_input_shapes.shape_num; i++)
    //        network_input_shapes.shapes[i].shape.dims[0] = batch_size;
    //    ie_network_reshape(ov_model->network, network_input_shapes);
    //}

    ie_network_input_shapes_free(&network_input_shapes);
    //network_input_shapes.shape_num = 0;

    status = ie_core_load_network(ov_model->core, ov_model->network, "CPU", &config, &ov_model->exe_network);
    if (status != OK)
        goto err;

    // create infer request for sync mode
    status = ie_exec_network_create_infer_request(ov_model->exe_network, &ov_model->infer_request);
    if (status != OK)
       goto err;

    nireq = 8; // TODO: pass in as parameter
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
        request_ctx->processing_frame_array = (ProcessingFrame **)malloc(batch_size * sizeof(ProcessingFrame *));
        request_ctx->num_processing_frames = 0;
        SafeQueuePush(ov_model->request_ctx_q, request_ctx);
    }

    pthread_mutex_init(&ov_model->callback_mutex, NULL);

    model->model = (void *)ov_model;
    model->options = options;
    model->set_input_output = &set_input_output_ov;
    model->get_input = &get_input_ov;
    model->get_input_blob = &get_input_blob_ov;
    model->get_output = &get_output_ov;

    return model;

err:
    if (model)
        av_freep(&model);
    if (ov_model) {
        if (ov_model->infer_request)
            ie_infer_request_free(&ov_model->infer_request);
        if (ov_model->infer_requests) {
           for (size_t i = 0; i < ov_model->num_reqs; ++i)
              if (ov_model->infer_requests[i])
                 ie_infer_request_free(&ov_model->infer_requests[i]);
           free(ov_model->infer_requests);
        }

        if (ov_model->request_ctx_q) {
           for (size_t i = 0; i < ov_model->num_reqs; ++i) {
              RequestContext *request_ctx = (RequestContext *)SafeQueuePop(ov_model->request_ctx_q);
              if (request_ctx->blob_name)
                 free(request_ctx->blob_name);
              if (request_ctx->processing_frame_array)
                 free(request_ctx->processing_frame_array);
           }
           SafeQueueDestroy(ov_model->request_ctx_q);
        }

        if (ov_model->exe_network)
            ie_exec_network_free(&ov_model->exe_network);
        if (ov_model->network)
            ie_network_free(&ov_model->network);
        if (ov_model->core)
            ie_core_free(&ov_model->core);
        av_freep(&ov_model);
    }

    av_log(NULL, AV_LOG_ERROR, "failed to load model\n");

    return NULL;
}

static void completion_callback_batch_infer(void *args) {

    RequestContext *request = (RequestContext *)args;
    DNNModel *model = (DNNModel*)request->model;
    OVModel *ov_model = (OVModel*)model->model;
    dimensions_t dims;
    precision_e precision;
    IEStatusCode status;
    char *blob_name;
    ie_blob_t *out_blob;
    DNNData output, new_blob;
    ProcessingFrame *processing_frame;
    ie_blob_buffer_t blob_buffer;
    ff_list_t *processing_frames;
    ff_list_t *processed;

    pthread_mutex_lock(&ov_model->callback_mutex);

    av_log(NULL, AV_LOG_INFO, "~~~~~~~~~~~~~ completion_callback_batch_infer\n");
    // get output blob
    if (request->blob_name)
       blob_name = av_strdup(request->blob_name);
    else {
       // defaultly return the first output blob
       ie_network_get_output_name(ov_model->network, 0, &blob_name);
    }

    ie_infer_request_get_blob(request->infer_request, blob_name, &out_blob);
    if (!out_blob) {
       av_log(NULL, AV_LOG_ERROR, "failed to get out blob\n");
       goto out;
    }

    status = ie_blob_get_buffer(out_blob, &blob_buffer);
    if (status != OK) {
        av_log(NULL, AV_LOG_ERROR, "failed to get buffer\n");
        goto out;
    }

    status |= ie_blob_get_dims(out_blob, &dims);
    status |= ie_blob_get_precision(out_blob, &precision);
    if (status != OK) {
        av_log(NULL, AV_LOG_ERROR, "failed to get precision or dims\n");
        goto out;
    }

    // output blob to DNNData 
    output.batch_size = dims.dims[0];
    output.channels = dims.dims[1];
    output.height   = dims.dims[2];
    output.width    = dims.dims[3];
    output.dt       = precision_to_datatype(precision);
    output.data     = blob_buffer.buffer;

    // model-specific post proc: DNNData to AVFrame, 
    for (int b = 0; b < request->num_processing_frames; b++) {
       _new_blob_by_batch_idx(ov_model, &output, &new_blob, b);
       processing_frame = request->processing_frame_array[b];
       if (((DNNPostProc2)model->post_proc)(&new_blob, processing_frame->frame_in, &processing_frame->frame_out, ov_model->user_data) != 0) {
          av_log(NULL, AV_LOG_ERROR, "post_proc failed\n");
          goto out;
       }
       // mark done
       processing_frame->inference_done = 1;
    }

    // enqueue processed frame queue
    pthread_mutex_lock(&ov_model->frame_q_mutex);
    processing_frames = ov_model->processing_frames;
    processed = ov_model->processed_frames;

    while (!processing_frames->empty(processing_frames)) {
        ProcessingFrame *front = (ProcessingFrame *)processing_frames->front(processing_frames);
        if (!front->inference_done) {
            break; // inference not completed yet
        }
        processed->push_back(processed, front->frame_out);
        processing_frames->pop_front(processing_frames);
        av_free(front);
    }
    av_log(NULL, AV_LOG_INFO, "processing frames (%ld), processed frames (%ld)\n", processing_frames->size(processing_frames), processed->size(processed));
    pthread_mutex_unlock(&ov_model->frame_q_mutex);

    if (request->blob_name) {
       av_free(request->blob_name);
       request->blob_name = NULL;
    }
    request->num_processing_frames = 0;
    // give back the request resource
    SafeQueuePush(ov_model->request_ctx_q, request);

out:
    av_free(blob_name);
    if (out_blob)
        ie_blob_free(&out_blob);
    pthread_mutex_unlock(&ov_model->callback_mutex);
}

static void completion_callback2(void *args) {

    RequestContext *request = (RequestContext *)args;
    ProcessingFrame *processing_frame = request->processing_frame;
    DNNModel *model = (DNNModel*)request->model;
    OVModel *ov_model = (OVModel*)model->model;
    dimensions_t dims;
    precision_e precision;
    IEStatusCode status;
    char *blob_name;
    ie_blob_t *out_blob;
    ie_blob_buffer_t blob_buffer;
    DNNData output;
    ff_list_t *processing_frames;
    ff_list_t *processed;

    pthread_mutex_lock(&ov_model->callback_mutex);

    // get output blob
    if (request->blob_name)
       blob_name = av_strdup(request->blob_name);
    else {
       // defaultly return the first output blob
       ie_network_get_output_name(ov_model->network, 0, &blob_name);
    }

    ie_infer_request_get_blob(request->infer_request, blob_name, &out_blob);
    if (!out_blob) {
       av_log(NULL, AV_LOG_ERROR, "failed to get out blob\n");
       goto out;
    }

    status = ie_blob_get_buffer(out_blob, &blob_buffer);
    if (status != OK) {
        av_log(NULL, AV_LOG_ERROR, "failed to get buffer\n");
        goto out;
    }

    status |= ie_blob_get_dims(out_blob, &dims);
    status |= ie_blob_get_precision(out_blob, &precision);
    if (status != OK) {
        av_log(NULL, AV_LOG_ERROR, "failed to get precision or dims\n");
        goto out;
    }

    // output blob to DNNData 
    output.channels = dims.dims[1];
    output.height   = dims.dims[2];
    output.width    = dims.dims[3];
    output.dt       = precision_to_datatype(precision);
    output.data     = blob_buffer.buffer;

    // model-specific post proc: DNNData to AVFrame, 
    if (((DNNPostProc2)model->post_proc)(&output, processing_frame->frame_in, &processing_frame->frame_out, ov_model->user_data) != 0) {
       av_log(NULL, AV_LOG_ERROR, "post_proc failed\n");
       goto out;
    }

    // mark done
    processing_frame->inference_done = 1;

    // enqueue processed frame queue
    pthread_mutex_lock(&ov_model->frame_q_mutex);
    processing_frames = ov_model->processing_frames;
    processed = ov_model->processed_frames;

    while (!processing_frames->empty(processing_frames)) {
        ProcessingFrame *front = (ProcessingFrame *)processing_frames->front(processing_frames);
        if (!front->inference_done) {
            break; // inference not completed yet
        }
        processed->push_back(processed, front->frame_out);
        processing_frames->pop_front(processing_frames);
        av_free(front);
    }
    pthread_mutex_unlock(&ov_model->frame_q_mutex);

    // give back the request resource
    if (request->blob_name) {
       av_free(request->blob_name);
       request->blob_name = NULL;
    }
    SafeQueuePush(ov_model->request_ctx_q, request);

out:
    av_free(blob_name);
    if (out_blob)
        ie_blob_free(&out_blob);
    pthread_mutex_unlock(&ov_model->callback_mutex);
}

static void completion_callback(void *args) {

    RequestContext *request = (RequestContext *)args;
    InferenceContext *inference_ctx = request->inference_ctx;
    ProcessingFrame *processing_frame = inference_ctx->processing_frame;
    DnnInterface *dnn_interface = inference_ctx->dnn_interface;
    DNNModel *model = (DNNModel*)dnn_interface->model;
    OVModel *ov_model = (OVModel*)model->model;
    dimensions_t dims;
    precision_e precision;
    IEStatusCode status;
    char *blob_name;
    ie_blob_t *out_blob;
    ie_blob_buffer_t blob_buffer;
    DNNData output;

    pthread_mutex_lock(&ov_model->callback_mutex);

    if (request->blob_name)
       blob_name = av_strdup(request->blob_name);
    else {
       // defaultly return the first output blob
       ie_network_get_output_name(ov_model->network, 0, &blob_name);
    }

    ie_infer_request_get_blob(request->infer_request, blob_name, &out_blob);
    if (!out_blob) {
       av_log(NULL, AV_LOG_ERROR, "failed to get out blob\n");
       goto out;
    }

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

    // output blob to DNNData 
    output.channels = dims.dims[1];
    output.height   = dims.dims[2];
    output.width    = dims.dims[3];
    output.dt       = precision_to_datatype(precision);
    output.data     = blob_buffer.buffer;

    // DNNData to AVFrame
    ((InferCallback)inference_ctx->cb)(&output, processing_frame, dnn_interface);

    av_free(inference_ctx);

    SafeQueuePush(ov_model->request_ctx_q, request);

out:
    av_free(blob_name);
    if (out_blob)
        ie_blob_free(&out_blob);
    pthread_mutex_unlock(&ov_model->callback_mutex);
}

DNNReturnType ff_dnn_execute_model_async_ov(const DNNModel *model, InferenceContext *inference_ctx, const char *blob_name)
{
    OVModel *ov_model = (OVModel *)model->model;
    RequestContext *request_ctx = (RequestContext *)SafeQueuePop(ov_model->request_ctx_q);

    request_ctx->callback.completeCallBackFunc = completion_callback;
    request_ctx->callback.args = request_ctx;
    request_ctx->inference_ctx = inference_ctx;
    request_ctx->blob_name = blob_name ? av_strdup(blob_name) : NULL;

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
        
        ff_list_free(ov_model->processing_frames);
        ff_list_free(ov_model->processed_frames);
        pthread_mutex_destroy(&ov_model->frame_q_mutex);

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

DNNModel *ff_dnn_load_model_2_ov(const char *model_filename, const char *options, void *user_data)
{
    DNNModel *dnn_model = ff_dnn_load_model_ov(model_filename);
    OVModel *ov_model = dnn_model->model;

    // TODO: parse options
    ov_model->batch_size = DEFAULT_BATCH_SIZE; // FIXME: get the batch size from user option
    ov_model->user_data = user_data;

    ov_model->processing_frames = ff_list_alloc();
    av_assert0(ov_model->processing_frames);
    ov_model->processed_frames = ff_list_alloc();
    av_assert0(ov_model->processed_frames);

    pthread_mutex_init(&ov_model->frame_q_mutex, NULL);

    return dnn_model;
}

static DNNReturnType _ff_dnn_execute_model_ov(OVModel *ov_model, DNNData *output, const char *output_name)
{
    dimensions_t dims;
    precision_e precision;
    ie_blob_t *out_blob;
    ie_blob_buffer_t blob_buffer;
    IEStatusCode status;

    av_assert0(ov_model->infer_request);
    status = ie_infer_request_infer(ov_model->infer_request);
    if (status != OK)
        return DNN_ERROR;

    status = ie_infer_request_get_blob(ov_model->infer_request, output_name, &out_blob);
    if (status != OK)
       return DNN_ERROR;
    status |= ie_blob_get_buffer(out_blob, &blob_buffer);
    status |= ie_blob_get_dims(out_blob, &dims);
    status |= ie_blob_get_precision(out_blob, &precision);
    if (status != OK)
       return DNN_ERROR;

    output->channels = dims.dims[1];
    output->height   = dims.dims[2];
    output->width    = dims.dims[3];
    output->dt       = precision_to_datatype(precision);
    output->data     = blob_buffer.buffer;

    return DNN_SUCCESS;
}

DNNReturnType ff_dnn_execute_model_2_ov(const DNNModel *model, AVFrame *in, const char *model_input_name,
                                        AVFrame **out, const char **output_names, uint32_t nb_output)
{
   DNNData input_blob, output;
   OVModel *ov_model;

   // FIXME: to support more than 1 output?
   av_assert0(nb_output == 1);

   if (!model)
      return AVERROR(EINVAL);

   ov_model = (OVModel *)model->model;

   // get input blob
   if (DNN_SUCCESS != _get_input_blob_ov(ov_model, &input_blob, model_input_name))
      return AVERROR(EINVAL);

   // preproc
   if(!model->pre_proc) {
      av_log(NULL, AV_LOG_ERROR, "pre_proc function not specified\n");
      return AVERROR(EINVAL);
   }
   ((DNNPreProc2)(model->pre_proc))(in, &input_blob, ov_model->user_data);

   if (DNN_SUCCESS != _ff_dnn_execute_model_ov(ov_model, &output, output_names[0]))
      return AVERROR(EINVAL);

   return ((DNNPostProc2)model->post_proc)(&output, in, out, ov_model->user_data);
}

void _new_blob_by_batch_idx(OVModel *ov_model, DNNData *input_blob, DNNData *new_blob, const int batch_idx)
{
   int ele_size;
   av_assert0(ov_model->batch_size == input_blob->batch_size);
   av_assert0(batch_idx < input_blob->batch_size);

   *new_blob = *input_blob;
   new_blob->batch_size = 1;

   ele_size = input_blob->dt == DNN_FLOAT ? sizeof(float) : 1;
   new_blob->data = (void *)((uint8_t *)input_blob->data + batch_idx * input_blob->channels * input_blob->height * input_blob->width * ele_size);

   return;
}

static int _create_and_enqueue_processing_frame(OVModel *ov_model, AVFrame *in, RequestContext *request_ctx)
{
    ProcessingFrame *processing_frame;
   // create a ProcessingFrame instance and push it into processing_frames queue
    pthread_mutex_lock(&ov_model->frame_q_mutex);
    processing_frame = (ProcessingFrame *)av_malloc(sizeof(ProcessingFrame)); // release in PushOutput()
    if (processing_frame == NULL) {
       pthread_mutex_unlock(&ov_model->frame_q_mutex);
       return AVERROR(EINVAL);
    }
    processing_frame->frame_in = in;
    processing_frame->frame_out = NULL;
    processing_frame->inference_done = 0;
    ov_model->processing_frames->push_back(ov_model->processing_frames, processing_frame);
    pthread_mutex_unlock(&ov_model->frame_q_mutex);

    request_ctx->processing_frame_array[request_ctx->num_processing_frames++] = processing_frame;

    return 0;
}


void ff_dnn_flush_ov(const DNNModel *model)
{
   OVModel *ov_model = (OVModel *)model->model;

   RequestContext *request_ctx = (RequestContext *)SafeQueuePop(ov_model->request_ctx_q);
   
   av_log(NULL, AV_LOG_INFO, "flush %d cached frames, batch_size: %d\n", request_ctx->num_processing_frames, ov_model->batch_size);

   request_ctx->callback.completeCallBackFunc = completion_callback_batch_infer;
   request_ctx->callback.args = request_ctx;
   request_ctx->model = model;

   ie_infer_set_completion_callback(request_ctx->infer_request, &request_ctx->callback);
   ie_infer_request_infer_async(request_ctx->infer_request);
}

DNNReturnType ff_dnn_execute_model_async_batch_ov(const DNNModel *model, AVFrame *in, const char *model_input_name,
                                                  const char **output_names, uint32_t nb_output)
{
    DNNData input_blob, new_blob;
    OVModel *ov_model;
    RequestContext *request_ctx;

    // FIXME: to support more than 1 output?
    av_assert0(nb_output == 1);

    if (!model || !in)
        return AVERROR(EINVAL);

    ov_model = (OVModel *)model->model;


    (model->get_input_blob)(model->model, &input_blob, model_input_name);

    request_ctx = (RequestContext *)SafeQueuePop(ov_model->request_ctx_q);

    _new_blob_by_batch_idx(ov_model, &input_blob, &new_blob, request_ctx->num_processing_frames);

    // preproc
    if(!model->pre_proc) {
        av_log(NULL, AV_LOG_ERROR, "pre_proc function not specified\n");
        return AVERROR(EINVAL);
    }

    ((DNNPreProc2)(model->pre_proc))(in, &new_blob, ov_model->user_data);

    _create_and_enqueue_processing_frame(ov_model, in, request_ctx);

    if (request_ctx->blob_name == NULL)
       request_ctx->blob_name = output_names[0] ? av_strdup(output_names[0]) : NULL;

    if (request_ctx->num_processing_frames == ov_model->batch_size) {
       request_ctx->callback.completeCallBackFunc = completion_callback_batch_infer;
       request_ctx->callback.args = request_ctx;
       request_ctx->model = model;

       av_log(NULL, AV_LOG_INFO, "~~~~~~~~~~~~~ infer async\n");

       ie_infer_set_completion_callback(request_ctx->infer_request, &request_ctx->callback);
       ie_infer_request_infer_async(request_ctx->infer_request);

    } else {
       SafeQueuePushFront(ov_model->request_ctx_q, request_ctx);
    }

    return DNN_SUCCESS;
}

DNNReturnType ff_dnn_execute_model_async_2_ov(const DNNModel *model, AVFrame *in, const char *model_input_name,
                                              const char **output_names, uint32_t nb_output)
{
    DNNData input_blob;
    OVModel *ov_model;
    ProcessingFrame *processing_frame;
    RequestContext *request_ctx;

    // FIXME: to support more than 1 output?
    av_assert0(nb_output == 1);

    if (!model || !in)
        return AVERROR(EINVAL);

    ov_model = (OVModel *)model->model;

    // preproc
    (model->get_input_blob)(model->model, &input_blob, model_input_name);

    if(!model->pre_proc) {
        av_log(NULL, AV_LOG_ERROR, "pre_proc function not specified\n");
        return AVERROR(EINVAL);
    }

    ((DNNPreProc2)(model->pre_proc))(in, &input_blob, ov_model->user_data);

    // create a ProcessingFrame instance and push it into processing_frames queue
    pthread_mutex_lock(&ov_model->frame_q_mutex);
    processing_frame = (ProcessingFrame *)av_malloc(sizeof(ProcessingFrame)); // release in PushOutput()
    if (processing_frame == NULL) {
       pthread_mutex_unlock(&ov_model->frame_q_mutex);
       return AVERROR(EINVAL);
    }
    processing_frame->frame_in = in;
    processing_frame->frame_out = NULL;
    processing_frame->inference_done = 0;
    ov_model->processing_frames->push_back(ov_model->processing_frames, processing_frame);
    pthread_mutex_unlock(&ov_model->frame_q_mutex);

    // async inference
    request_ctx = (RequestContext *)SafeQueuePop(ov_model->request_ctx_q);
    request_ctx->processing_frame = processing_frame;
    request_ctx->blob_name = output_names[0] ? av_strdup(output_names[0]) : NULL;
    request_ctx->model = model;
    request_ctx->callback.completeCallBackFunc = completion_callback2;
    request_ctx->callback.args = request_ctx;

    ie_infer_set_completion_callback(request_ctx->infer_request, &request_ctx->callback);
    ie_infer_request_infer_async(request_ctx->infer_request);

    return DNN_SUCCESS;
}

DNNAsyncStatusType ff_dnn_get_async_result_ov(const DNNModel *model, AVFrame **out)
{
    OVModel *ov_model;
    ff_list_t *processing_frames;
    ff_list_t *processed_frames;

    if (!model || !out)
       return DAST_FAIL;

    ov_model = (OVModel *)model->model;

    processing_frames = ov_model->processing_frames;
    processed_frames = ov_model->processed_frames;

    if (processed_frames->empty(processed_frames)) {
       if (!processing_frames->empty(processing_frames))
        return DAST_NOT_READY;
       else
        return DAST_EMPTY_QUEUE;
    }

    pthread_mutex_lock(&ov_model->frame_q_mutex);
    *out = (AVFrame *)processed_frames->front(processed_frames);
    processed_frames->pop_front(processed_frames);
    pthread_mutex_unlock(&ov_model->frame_q_mutex);

    return DAST_SUCCESS;
}

static DNNReturnType get_output_ov(void *model, DNNData *output, const char *output_name)
{
   return _ff_dnn_execute_model_ov((OVModel *)model, output, output_name);
}
