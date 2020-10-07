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
#include "dnn_io_proc.h"
#include "dnn_safe_queue.h"
#include "dnn_ff_list.h"
#include "libavformat/avio.h"
#include "libavutil/avassert.h"
#include "libavutil/opt.h"
#include "libavutil/avstring.h"
#include "../internal.h"
#include <c_api/ie_c_api.h>
#include <pthread.h>

#define DEFAULT_BATCH_SIZE (1)
#define DEFAULT_MAX_REQUEST (8)

typedef struct OVOptions{
    char *device_type;
    int async;
    int nireq;
    int batch_size;
} OVOptions;

typedef struct OVContext {
    const AVClass *class;
    OVOptions options;
} OVContext;

typedef struct ProcessingFrame {
    AVFrame *frame_in;
    AVFrame *frame_out;
    int inference_done;
} ProcessingFrame;

typedef struct RequestContext {
    char *blob_name;
    ie_infer_request_t *infer_request;
    ie_complete_call_back_t callback;
    ProcessingFrame **processing_frame_array;
    int num_processing_frames;
    const DNNModel *model;
} RequestContext;

typedef struct OVModel{
    OVContext ctx;
    DNNModel *model;
    ie_core_t *core;
    ie_network_t *network;
    ie_executable_network_t *exe_network;
    ie_infer_request_t *infer_request;

    int async;
    int batch_size;
    int num_reqs;
    pthread_mutex_t frame_q_mutex;
    ff_list_t *processing_frames;
    ff_list_t *processed_frames;
    SafeQueueT *request_ctx_q; // queue to hold request context
    ie_infer_request_t **infer_requests;
    pthread_mutex_t callback_mutex;
    void *user_data;
} OVModel;

#define APPEND_STRING(generated_string, iterate_string)                                            \
    generated_string = generated_string ? av_asprintf("%s %s", generated_string, iterate_string) : \
                                          av_asprintf("%s", iterate_string);

#define OFFSET(x) offsetof(OVContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM
static const AVOption dnn_openvino_options[] = {
    { "device", "device to run model",      OFFSET(options.device_type), AV_OPT_TYPE_STRING, { .str = "CPU" }, 0, 0, FLAGS },
    { "async", "enable async inference",    OFFSET(options.async),       AV_OPT_TYPE_BOOL,   { .i64 = 1}, 0, 1, FLAGS},
    { "nireq", "inference request number",  OFFSET(options.nireq),       AV_OPT_TYPE_INT,    { .i64 = 8 }, 1, 128, FLAGS},
    { "batch_size", "batch size per infer", OFFSET(options.batch_size),  AV_OPT_TYPE_INT,    { .i64 = 4 }, 1, 1024, FLAGS},
    { NULL }
};

AVFILTER_DEFINE_CLASS(dnn_openvino);

static DNNReturnType execute_model_ov(const DNNModel *model, const char *input_name, AVFrame *in_frame,
                                      const char **output_names, uint32_t nb_output, AVFrame *out_frame,
                                      int do_ioproc);
static DNNReturnType get_input_blob_common(ie_infer_request_t *infer_request, DNNData *input, const char *input_name, ie_blob_t **input_blob_p);
static void new_blob_by_batch_idx(OVModel *ov_model, DNNData *input_blob, DNNData *new_blob, const int batch_idx);
static int create_and_enqueue_processing_frame(OVModel *ov_model, AVFrame *in, AVFrame *out, RequestContext *request_ctx);
static void completion_callback_batch_infer(void *args);

static void q_log(OVModel *ov_model, const char *msg_prefix, RequestContext *rc)
{
    av_log(NULL, AV_LOG_INFO, "q_log: %s, processing_frames(%ld), processed_frames(%ld), request_q(%d), batch_idx(%d)\n",
           msg_prefix,
           ov_model->processing_frames->size(ov_model->processing_frames),
           ov_model->processed_frames->size(ov_model->processed_frames),
           SafeQueueSize(ov_model->request_ctx_q),
           rc ? rc->num_processing_frames : -111);
}

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

static DNNReturnType get_input_ov(void *model, DNNData *input, const char *input_name)
{
    OVModel *ov_model = (OVModel *)model;
    OVContext *ctx = &ov_model->ctx;
    char *model_input_name = NULL;
    char *all_input_names = NULL;
    IEStatusCode status;
    size_t model_input_count = 0;
    dimensions_t dims;
    precision_e precision;

    status = ie_network_get_inputs_number(ov_model->network, &model_input_count);
    if (status != OK) {
        av_log(ctx, AV_LOG_ERROR, "Failed to get input count\n");
        return DNN_ERROR;
    }

    for (size_t i = 0; i < model_input_count; i++) {
        status = ie_network_get_input_name(ov_model->network, i, &model_input_name);
        if (status != OK) {
            av_log(ctx, AV_LOG_ERROR, "Failed to get No.%d input's name\n", (int)i);
            return DNN_ERROR;
        }
        if (strcmp(model_input_name, input_name) == 0) {
            ie_network_name_free(&model_input_name);
            status |= ie_network_get_input_dims(ov_model->network, input_name, &dims);
            status |= ie_network_get_input_precision(ov_model->network, input_name, &precision);
            if (status != OK) {
                av_log(ctx, AV_LOG_ERROR, "Failed to get No.%d input's dims or precision\n", (int)i);
                return DNN_ERROR;
            }

            // The order of dims in the openvino is fixed and it is always NCHW for 4-D data.
            // while we pass NHWC data from FFmpeg to openvino
            status = ie_network_set_input_layout(ov_model->network, input_name, NHWC);
            if (status != OK) {
                av_log(ctx, AV_LOG_ERROR, "Input \"%s\" does not match layout NHWC\n", input_name);
                return DNN_ERROR;
            }

            input->channels = dims.dims[1];
            input->height   = dims.dims[2];
            input->width    = dims.dims[3];
            input->dt       = precision_to_datatype(precision);
            return DNN_SUCCESS;
        } else {
            //incorrect input name
            APPEND_STRING(all_input_names, model_input_name)
        }

        ie_network_name_free(&model_input_name);
    }

    av_log(ctx, AV_LOG_ERROR, "Could not find \"%s\" in model, all input(s) are: \"%s\"\n", input_name, all_input_names);
    return DNN_ERROR;
}

static DNNReturnType get_output_ov(void *model, const char *input_name, int input_width, int input_height,
                                   const char *output_name, int *output_width, int *output_height)
{
    DNNReturnType ret;
    OVModel *ov_model = (OVModel *)model;
    AVFrame *in_frame = av_frame_alloc();
    AVFrame *out_frame = av_frame_alloc();
    in_frame->width = input_width;
    in_frame->height = input_height;

    ret = execute_model_ov(ov_model->model, input_name, in_frame, &output_name, 1, out_frame, 0);
    *output_width = out_frame->width;
    *output_height = out_frame->height;

    av_frame_free(&out_frame);
    av_frame_free(&in_frame);
    return ret;
}

DNNModel *ff_dnn_load_model_ov(const char *model_filename, const char *options, void *userdata)
{
    char *all_dev_names = NULL;
    DNNModel *model = NULL;
    OVModel *ov_model = NULL;
    OVContext *ctx = NULL;
    IEStatusCode status;
    ie_config_t config = {NULL, NULL, NULL};
    ie_available_devices_t a_dev;
    input_shapes_t network_input_shapes;
    int batch_size;

    model = av_mallocz(sizeof(DNNModel));
    if (!model){
        return NULL;
    }

    ov_model = av_mallocz(sizeof(OVModel));
    if (!ov_model)
        goto err;
    ov_model->model = model;
    ov_model->ctx.class = &dnn_openvino_class;
    ctx = &ov_model->ctx;

    //parse options
    av_opt_set_defaults(ctx);
    if (av_opt_set_from_string(ctx, options, NULL, "=", "&") < 0) {
        av_log(ctx, AV_LOG_ERROR, "Failed to parse options \"%s\"\n", options);
        goto err;
    }

    status = ie_core_create("", &ov_model->core);
    if (status != OK)
        goto err;

    status = ie_core_read_network(ov_model->core, model_filename, NULL, &ov_model->network);
    if (status != OK)
        goto err;

    ov_model->batch_size = ctx->options.batch_size;

    // reshape input to support batch mode
    status = ie_network_get_input_shapes(ov_model->network, &network_input_shapes);
    if (status != OK)
        goto err;

    if (ov_model->batch_size > 1 && network_input_shapes.shapes) {
        for (int i = 0; i < network_input_shapes.shape_num; i++)
            network_input_shapes.shapes[i].shape.dims[0] = ov_model->batch_size;
        ie_network_reshape(ov_model->network, network_input_shapes);
    }
    ie_network_input_shapes_free(&network_input_shapes);

    status = ie_core_load_network(ov_model->core, ov_model->network, ctx->options.device_type, &config, &ov_model->exe_network);
    if (status != OK) {
        av_log(ctx, AV_LOG_ERROR, "Failed to init OpenVINO model\n");
        status = ie_core_get_available_devices(ov_model->core, &a_dev);
        if (status != OK) {
            av_log(ctx, AV_LOG_ERROR, "Failed to get available devices\n");
            goto err;
        }
        for (int i = 0; i < a_dev.num_devices; i++) {
            APPEND_STRING(all_dev_names, a_dev.devices[i])
        }
        av_log(ctx, AV_LOG_ERROR,"device %s may not be supported, all available devices are: \"%s\"\n",
               ctx->options.device_type, all_dev_names);
        goto err;
    }

    status = ie_exec_network_create_infer_request(ov_model->exe_network, &ov_model->infer_request);
    if (status != OK)
        goto err;

    if (ctx->options.async) {
        ov_model->num_reqs = ctx->options.nireq;
        ov_model->infer_requests = (ie_infer_request_t **)malloc(ov_model->num_reqs * sizeof(ie_infer_request_t *));
        if (!ov_model->infer_requests) {
            goto err;
        }
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
            request_ctx->processing_frame_array = (ProcessingFrame **)malloc(ov_model->batch_size * sizeof(ProcessingFrame *));
            request_ctx->num_processing_frames = 0;
            SafeQueuePush(ov_model->request_ctx_q, request_ctx);
        }

        pthread_mutex_init(&ov_model->callback_mutex, NULL);

        // inner queue initialization
        ov_model->processing_frames = ff_list_alloc();
        av_assert0(ov_model->processing_frames);
        ov_model->processed_frames = ff_list_alloc();
        av_assert0(ov_model->processed_frames);
        pthread_mutex_init(&ov_model->frame_q_mutex, NULL);
    }


    model->model = (void *)ov_model;
    model->get_input = &get_input_ov;
    model->get_output = &get_output_ov;
    model->options = options;
    model->userdata = userdata;

    return model;

err:
    if (model)
        av_freep(&model);
    if (ov_model) {
        if (ov_model->infer_request)
            ie_infer_request_free(&ov_model->infer_request);
        if (ov_model->async) {
            pthread_mutex_destroy(&ov_model->callback_mutex);
            ff_list_free(ov_model->processing_frames);
            ff_list_free(ov_model->processed_frames);
            pthread_mutex_destroy(&ov_model->frame_q_mutex);
            if (ov_model->request_ctx_q) {
                for (size_t i = 0; i < ov_model->num_reqs; ++i) {
                    RequestContext *request_ctx = (RequestContext *)SafeQueuePop(ov_model->request_ctx_q);
                    if (request_ctx->blob_name)
                        free(request_ctx->blob_name);
                    if (request_ctx->processing_frame_array)
                        free(request_ctx->processing_frame_array);
                    free(request_ctx);
                }
                SafeQueueDestroy(ov_model->request_ctx_q);
            }
            if (ov_model->infer_requests) {
                for (size_t i = 0; i < ov_model->num_reqs; ++i)
                    if (ov_model->infer_requests[i])
                        ie_infer_request_free(&ov_model->infer_requests[i]);
                free(ov_model->infer_requests);
            }
        }
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

static DNNReturnType execute_model_ov(const DNNModel *model, const char *input_name, AVFrame *in_frame,
                                      const char **output_names, uint32_t nb_output, AVFrame *out_frame,
                                      int do_ioproc)
{
    char *model_output_name = NULL;
    char *all_output_names = NULL;
    dimensions_t dims;
    precision_e precision;
    ie_blob_buffer_t blob_buffer;
    OVModel *ov_model = (OVModel *)model->model;
    OVContext *ctx = &ov_model->ctx;
    IEStatusCode status;
    size_t model_output_count = 0;
    DNNData input, output;
    ie_blob_t *input_blob = NULL;

    status = ie_infer_request_get_blob(ov_model->infer_request, input_name, &input_blob);
    if (status != OK) {
        av_log(ctx, AV_LOG_ERROR, "Failed to get input blob\n");
        return DNN_ERROR;
    }

    status |= ie_blob_get_dims(input_blob, &dims);
    status |= ie_blob_get_precision(input_blob, &precision);
    if (status != OK) {
        av_log(ctx, AV_LOG_ERROR, "Failed to get input blob dims/precision\n");
        return DNN_ERROR;
    }

    status = ie_blob_get_buffer(input_blob, &blob_buffer);
    if (status != OK) {
        av_log(ctx, AV_LOG_ERROR, "Failed to get input blob buffer\n");
        return DNN_ERROR;
    }

    input.height = dims.dims[2];
    input.width = dims.dims[3];
    input.channels = dims.dims[1];
    input.data = blob_buffer.buffer;
    input.dt = precision_to_datatype(precision);
    if (do_ioproc) {
        if (ov_model->model->pre_proc != NULL) {
            ov_model->model->pre_proc(in_frame, &input, ov_model->model->userdata);
        } else {
            proc_from_frame_to_dnn(in_frame, &input, ctx);
        }
    }
    ie_blob_free(&input_blob);

    if (nb_output != 1) {
        // currently, the filter does not need multiple outputs,
        // so we just pending the support until we really need it.
        av_log(ctx, AV_LOG_ERROR, "do not support multiple outputs\n");
        return DNN_ERROR;
    }

    status = ie_infer_request_infer(ov_model->infer_request);
    if (status != OK) {
        av_log(ctx, AV_LOG_ERROR, "Failed to start synchronous model inference\n");
        return DNN_ERROR;
    }

    for (uint32_t i = 0; i < nb_output; ++i) {
        const char *output_name = output_names[i];
        ie_blob_t *output_blob = NULL;
        status = ie_infer_request_get_blob(ov_model->infer_request, output_name, &output_blob);
        if (status != OK) {
            //incorrect output name
            av_log(ctx, AV_LOG_ERROR, "Failed to get model output data\n");
            status = ie_network_get_outputs_number(ov_model->network, &model_output_count);
            for (size_t i = 0; i < model_output_count; i++) {
                status = ie_network_get_output_name(ov_model->network, i, &model_output_name);
                APPEND_STRING(all_output_names, model_output_name)
            }
            av_log(ctx, AV_LOG_ERROR,
                   "output \"%s\" may not correct, all output(s) are: \"%s\"\n",
                   output_name, all_output_names);
            return DNN_ERROR;
        }

        status = ie_blob_get_buffer(output_blob, &blob_buffer);
        if (status != OK) {
            av_log(ctx, AV_LOG_ERROR, "Failed to access output memory\n");
            return DNN_ERROR;
        }

        status |= ie_blob_get_dims(output_blob, &dims);
        status |= ie_blob_get_precision(output_blob, &precision);
        if (status != OK) {
            av_log(ctx, AV_LOG_ERROR, "Failed to get dims or precision of output\n");
            return DNN_ERROR;
        }

        output.channels = dims.dims[1];
        output.height   = dims.dims[2];
        output.width    = dims.dims[3];
        output.dt       = precision_to_datatype(precision);
        output.data     = blob_buffer.buffer;
        if (do_ioproc) {
            if (ov_model->model->post_proc != NULL) {
                ov_model->model->post_proc(out_frame, &output, ov_model->model->userdata);
            } else {
                proc_from_dnn_to_frame(out_frame, &output, ctx);
            }
        } else {
            out_frame->width = output.width;
            out_frame->height = output.height;
        }
        ie_blob_free(&output_blob);
    }

    return DNN_SUCCESS;
}

DNNReturnType ff_dnn_execute_model_ov(const DNNModel *model, const char *input_name, AVFrame *in_frame,
                                      const char **output_names, uint32_t nb_output, AVFrame *out_frame)
{
    OVModel *ov_model = (OVModel *)model->model;
    OVContext *ctx = &ov_model->ctx;

    if (!in_frame) {
        av_log(ctx, AV_LOG_ERROR, "in frame is NULL when execute model.\n");
        return DNN_ERROR;
    }

    if (!out_frame) {
        av_log(ctx, AV_LOG_ERROR, "out frame is NULL when execute model.\n");
        return DNN_ERROR;
    }

    return execute_model_ov(model, input_name, in_frame, output_names, nb_output, out_frame, 1);
}

DNNReturnType ff_dnn_execute_model_async_ov(const DNNModel *model, const char *input_name, AVFrame *in_frame,
                                            const char **output_names, uint32_t nb_output, AVFrame *out_frame)
{
    DNNData input_blob, new_blob;
    OVModel *ov_model;
    RequestContext *request_ctx;
    ie_blob_t *ie_blob = NULL;

    // FIXME: to support more than 1 output?
    av_assert0(nb_output == 1);

    if (!model || !in_frame)
        return AVERROR(EINVAL);

    ov_model = (OVModel *)model->model;

    request_ctx = (RequestContext *)SafeQueuePop(ov_model->request_ctx_q);


    if (DNN_SUCCESS != get_input_blob_common(request_ctx->infer_request, &input_blob, input_name, &ie_blob)) {
       SafeQueuePushFront(ov_model->request_ctx_q, request_ctx);
       return DNN_ERROR;
    }

    new_blob_by_batch_idx(ov_model, &input_blob, &new_blob, request_ctx->num_processing_frames);

    // preproc
    if (ov_model->model->pre_proc != NULL) {
        ov_model->model->pre_proc(in_frame, &new_blob, ov_model->model->userdata);
    } else {
        proc_from_frame_to_dnn(in_frame, &new_blob, ov_model->model->userdata);
    }
    
    ie_blob_free(&ie_blob);

    create_and_enqueue_processing_frame(ov_model, in_frame, out_frame, request_ctx);

    if (request_ctx->blob_name == NULL)
       request_ctx->blob_name = output_names[0] ? av_strdup(output_names[0]) : NULL;

    if (request_ctx->num_processing_frames == ov_model->batch_size) {
       request_ctx->callback.completeCallBackFunc = completion_callback_batch_infer;
       request_ctx->callback.args = request_ctx;
       request_ctx->model = model;

       ie_infer_set_completion_callback(request_ctx->infer_request, &request_ctx->callback);
       ie_infer_request_infer_async(request_ctx->infer_request);

    } else {
       SafeQueuePushFront(ov_model->request_ctx_q, request_ctx);
    }

    return DNN_SUCCESS;
}

void ff_dnn_flush_ov(const DNNModel *model)
{
   OVModel *ov_model = (OVModel *)model->model;

   RequestContext *request_ctx = (RequestContext *)SafeQueuePop(ov_model->request_ctx_q);
   
   //av_log(NULL, AV_LOG_INFO, "flush %d cached frames, batch_size: %d\n", request_ctx->num_processing_frames, ov_model->batch_size);

   request_ctx->callback.completeCallBackFunc = completion_callback_batch_infer;
   request_ctx->callback.args = request_ctx;
   request_ctx->model = model;

   ie_infer_set_completion_callback(request_ctx->infer_request, &request_ctx->callback);
   ie_infer_request_infer_async(request_ctx->infer_request);
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

static DNNReturnType get_input_blob_common(ie_infer_request_t *infer_request, DNNData *input, const char *input_name, ie_blob_t **input_blob_p)
{
    ie_blob_t *input_blob;
    IEStatusCode status;
    dimensions_t dims;
    precision_e precision;
    ie_blob_buffer_t blob_buffer;

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
    *input_blob_p = input_blob;

    return DNN_SUCCESS;

err:
    ie_blob_free(&input_blob);
    return DNN_ERROR;
}

static void new_blob_by_batch_idx(OVModel *ov_model, DNNData *input_blob, DNNData *new_blob, const int batch_idx)
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

static int create_and_enqueue_processing_frame(OVModel *ov_model, AVFrame *in, AVFrame *out, RequestContext *request_ctx)
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
    processing_frame->frame_out = out;
    processing_frame->inference_done = 0;
    ov_model->processing_frames->push_back(ov_model->processing_frames, processing_frame);
    pthread_mutex_unlock(&ov_model->frame_q_mutex);

    request_ctx->processing_frame_array[request_ctx->num_processing_frames++] = processing_frame;

    return 0;
}

static void completion_callback_batch_infer(void *args)
{
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
       new_blob_by_batch_idx(ov_model, &output, &new_blob, b);
       processing_frame = request->processing_frame_array[b];

       if (ov_model->model->post_proc != NULL) {
          ov_model->model->post_proc(processing_frame->frame_out, &new_blob, ov_model->model->userdata);
       } else {
          proc_from_dnn_to_frame(processing_frame->frame_out, &new_blob, ov_model->model->userdata);
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
    ie_blob_free(&out_blob);
    pthread_mutex_unlock(&ov_model->callback_mutex);
}

void ff_dnn_free_model_ov(DNNModel **model)
{
    if (*model) {
        OVModel *ov_model = (OVModel *)(*model)->model;
        if (ov_model->infer_request)
            ie_infer_request_free(&ov_model->infer_request);
        if (ov_model->async) {
            pthread_mutex_destroy(&ov_model->callback_mutex);
            ff_list_free(ov_model->processing_frames);
            ff_list_free(ov_model->processed_frames);
            pthread_mutex_destroy(&ov_model->frame_q_mutex);
            if (ov_model->request_ctx_q) {
                for (size_t i = 0; i < ov_model->num_reqs; ++i) {
                    RequestContext *request_ctx = (RequestContext *)SafeQueuePop(ov_model->request_ctx_q);
                    if (request_ctx->blob_name)
                        free(request_ctx->blob_name);
                    if (request_ctx->processing_frame_array)
                        free(request_ctx->processing_frame_array);
                    free(request_ctx);
                }
                SafeQueueDestroy(ov_model->request_ctx_q);
            }
            if (ov_model->infer_requests) {
                for (size_t i = 0; i < ov_model->num_reqs; ++i)
                    if (ov_model->infer_requests[i])
                        ie_infer_request_free(&ov_model->infer_requests[i]);
                free(ov_model->infer_requests);
            }
        }
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
