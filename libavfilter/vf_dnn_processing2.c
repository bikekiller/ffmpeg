/*
 * Copyright (c) 2019 Guo Yejun
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
 * implementing a async generic image processing filter using deep learning networks.
 */

#include "libavformat/avio.h"
#include "libavutil/opt.h"
#include "libavutil/pixdesc.h"
#include "libavutil/avassert.h"
#include "libavutil/imgutils.h"
#include "avfilter.h"
#include "dnn_interface.h"
#include "formats.h"
#include "internal.h"
#include "libswscale/swscale.h"

typedef struct DnnProcessing2Context {
    const AVClass *class;

    char *model_filename;
    DNNBackendType backend_type;
    char *model_inputname;
    char *model_outputname;
    int  batch_size;
    FFBaseInference *dnn_interface;
    struct SwsContext *sws_resize;
    struct SwsContext *sws_gray8_to_grayf32;
    struct SwsContext *sws_grayf32_to_gray8;
    struct SwsContext *sws_uv_scale;
    int sws_uv_height;
} DnnProcessing2Context;

#define OFFSET(x) offsetof(DnnProcessing2Context, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM
static const AVOption dnn_processing2_options[] = {
    { "dnn_backend", "DNN backend",                OFFSET(backend_type),     AV_OPT_TYPE_INT,       { .i64 = 0 },    INT_MIN, INT_MAX, FLAGS, "backend" },
    { "native",      "native backend flag",        0,                        AV_OPT_TYPE_CONST,     { .i64 = 0 },    0, 0, FLAGS, "backend" },
#if (CONFIG_LIBTENSORFLOW == 1)
    { "tensorflow",  "tensorflow backend flag",    0,                        AV_OPT_TYPE_CONST,     { .i64 = 1 },    0, 0, FLAGS, "backend" },
#endif
#if (CONFIG_LIBOPENVINO == 1)
    { "openvino",    "openvino backend flag",      0,                        AV_OPT_TYPE_CONST,     { .i64 = 2 },    0, 0, FLAGS, "backend" },
#endif
    { "model",       "path to model file",         OFFSET(model_filename),   AV_OPT_TYPE_STRING,    { .str = NULL }, 0, 0, FLAGS },
    { "input",       "input name of the model",    OFFSET(model_inputname),  AV_OPT_TYPE_STRING,    { .str = NULL }, 0, 0, FLAGS },
    { "output",      "output name of the model",   OFFSET(model_outputname), AV_OPT_TYPE_STRING,    { .str = NULL }, 0, 0, FLAGS },
    { "batch_size",  "batch size per infer",       OFFSET(batch_size),       AV_OPT_TYPE_INT,       { .i64 = 1 },    1, 1000, FLAGS},
    { NULL }
};

AVFILTER_DEFINE_CLASS(dnn_processing2);

static int copy_from_dnn_to_frame(DnnProcessing2Context *ctx, DNNData *dnn_output, AVFrame *frame)
{
    int bytewidth = av_image_get_linesize(frame->format, frame->width, 0);

    switch (frame->format) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
        if (dnn_output->dt == DNN_FLOAT) {
            sws_scale(ctx->sws_grayf32_to_gray8, (const uint8_t *[4]){(const uint8_t *)dnn_output->data, 0, 0, 0},
                      (const int[4]){frame->width * 3 * sizeof(float), 0, 0, 0},
                      0, frame->height, (uint8_t * const*)frame->data, frame->linesize);

        } else {
            av_assert0(dnn_output->dt == DNN_UINT8);
            av_image_copy_plane(frame->data[0], frame->linesize[0],
                                dnn_output->data, bytewidth,
                                bytewidth, frame->height);
        }
        return 0;
    case AV_PIX_FMT_GRAY8:
        // it is possible that data type of dnn output is float32,
        // need to add support for such case when needed.
        av_assert0(dnn_output->dt == DNN_UINT8);
        av_image_copy_plane(frame->data[0], frame->linesize[0],
                            dnn_output->data, bytewidth,
                            bytewidth, frame->height);
        return 0;
    case AV_PIX_FMT_GRAYF32:
        av_assert0(dnn_output->dt == DNN_FLOAT);
        av_image_copy_plane(frame->data[0], frame->linesize[0],
                            dnn_output->data, bytewidth,
                            bytewidth, frame->height);
        return 0;
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUV410P:
    case AV_PIX_FMT_YUV411P:
        sws_scale(ctx->sws_grayf32_to_gray8, (const uint8_t *[4]){(const uint8_t *)dnn_output->data, 0, 0, 0},
                  (const int[4]){frame->width * sizeof(float), 0, 0, 0},
                  0, frame->height, (uint8_t * const*)frame->data, frame->linesize);
        return 0;
    default:
        return AVERROR(EIO);
    }

    return 0;
}

static int copy_from_frame_to_dnn(DnnProcessing2Context *ctx, const AVFrame *frame, DNNData *dnn_input)
{
    int bytewidth = av_image_get_linesize(frame->format, frame->width, 0);

    if ((dnn_input->width != frame->width || dnn_input->height != frame->height) && !ctx->resized_frame_allocated) {
       // alloc frame with the same resolution with model input
       uint8_t *planes[4];
       int stride[4] = {}
       av_image_alloc(planes, );
    
    
    }
    sws_scale(ctx->sws_resize, (const uint8_t **)frame->data, frame->linesize)

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

static int prepare_sws_context(AVFilterLink *outlink)
{
    AVFilterContext *context = outlink->src;
    DnnProcessing2Context *ctx = context->priv;
    AVFilterLink *inlink = context->inputs[0];
    enum AVPixelFormat fmt = inlink->format;
    DNNDataType input_dt  = ctx->input.dt;
    DNNDataType output_dt = ctx->output.dt;

    switch (fmt) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
        if (input_dt == DNN_FLOAT) {
            ctx->sws_gray8_to_grayf32 = sws_getContext(inlink->w * 3,
                                                       inlink->h,
                                                       AV_PIX_FMT_GRAY8,
                                                       inlink->w * 3,
                                                       inlink->h,
                                                       AV_PIX_FMT_GRAYF32,
                                                       0, NULL, NULL, NULL);
        }
        if (output_dt == DNN_FLOAT) {
            ctx->sws_grayf32_to_gray8 = sws_getContext(outlink->w * 3,
                                                       outlink->h,
                                                       AV_PIX_FMT_GRAYF32,
                                                       outlink->w * 3,
                                                       outlink->h,
                                                       AV_PIX_FMT_GRAY8,
                                                       0, NULL, NULL, NULL);
        }
        return 0;
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUV410P:
    case AV_PIX_FMT_YUV411P:
        av_assert0(input_dt == DNN_FLOAT);
        av_assert0(output_dt == DNN_FLOAT);
        ctx->sws_gray8_to_grayf32 = sws_getContext(inlink->w,
                                                   inlink->h,
                                                   AV_PIX_FMT_GRAY8,
                                                   inlink->w,
                                                   inlink->h,
                                                   AV_PIX_FMT_GRAYF32,
                                                   0, NULL, NULL, NULL);
        ctx->sws_grayf32_to_gray8 = sws_getContext(outlink->w,
                                                   outlink->h,
                                                   AV_PIX_FMT_GRAYF32,
                                                   outlink->w,
                                                   outlink->h,
                                                   AV_PIX_FMT_GRAY8,
                                                   0, NULL, NULL, NULL);

        if (inlink->w != outlink->w || inlink->h != outlink->h) {
            const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(fmt);
            int sws_src_h = AV_CEIL_RSHIFT(inlink->h, desc->log2_chroma_h);
            int sws_src_w = AV_CEIL_RSHIFT(inlink->w, desc->log2_chroma_w);
            int sws_dst_h = AV_CEIL_RSHIFT(outlink->h, desc->log2_chroma_h);
            int sws_dst_w = AV_CEIL_RSHIFT(outlink->w, desc->log2_chroma_w);
            ctx->sws_uv_scale = sws_getContext(sws_src_w, sws_src_h, AV_PIX_FMT_GRAY8,
                                               sws_dst_w, sws_dst_h, AV_PIX_FMT_GRAY8,
                                               SWS_BICUBIC, NULL, NULL, NULL);
            ctx->sws_uv_height = sws_src_h;
        }
        return 0;
    default:
        //do nothing
        break;
    }

    return 0;
}

static int config_output(AVFilterLink *outlink)
{
    AVFilterContext *context = outlink->src;
    DnnProcessing2Context *ctx = context->priv;
    DNNReturnType result;

    // have a try run in case that the dnn model resize the frame
    result = (ctx->dnn_interface->model->get_output)(ctx->dnn_interface->model, &ctx->output, ctx->model_outputname);

    if (result != DNN_SUCCESS){
        av_log(ctx, AV_LOG_ERROR, "failed to execute model\n");
        return AVERROR(EIO);
    }

    outlink->w = ctx->output.width;
    outlink->h = ctx->output.height;

    prepare_sws_context(outlink);

    return 0;
}

// AVFrame to model input
static int pre_proc(AVFrame *frame_in, DNNData *model_input, FFBaseInference *base)
{
   if (!frame_in || !base)
      return -1;

   DnnProcessing2Context *ctx = (DnnProcessing2Context *)base->filter_ctx->priv;
_type   if (copy_from_frame_to_dnn(ctx, frame_in, model_input) != 0) {
      av_log(ctx, AV_LOG_ERROR, "copy_from_frame_to_dnn failed\n");
      return -1;
   }

   return 0;
}

// model output to AVFrame
static int post_proc(DNNData *model_output, AVFrame *frame_in, AVFrame **frame_out_p, FFBaseInference *base)
{
   if (!model_output || !frame_in || !frame_out_p || !base)
      return -1; 

   AVFilterContext *filter_ctx = base->filter_ctx;
   AVFilterLink *inlink = ctx->inputs[0];
   AVFilterLink *outlink = ctx->outputs[0];

   AVFrame *frame_out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
   if (!output) {
      av_log(NULL, AV_LOG_ERROR, "can't get video buffer from outlink\n");
      return AVERROR(EINVAL);
   }

   av_frame_copy_props(frame_out, frame_in);

   if (copy_from_dnn_to_frame(base, model_output, frame_out) != 0) {
        av_log(NULL, AV_LOG_ERROR, "copy_from_dnn_to_frame failed\n");
        av_frame_free(&frame_out);
        return AVERROR(EINVAL);
   }

   *frame_out_p = frame_out;

   return 0;
}

static av_cold int init(AVFilterContext *context)
{
    DnnProcessing2Context *s = context->priv;

    if (!s->model_filename) {
        av_log(s, AV_LOG_ERROR, "model file for network is not specified\n");
        return AVERROR(EINVAL);
    }
    if (!s->model_inputname) {
        av_log(s, AV_LOG_ERROR, "input name of the model network is not specified\n");
        return AVERROR(EINVAL);
    }
    if (!s->model_outputname) {
        av_log(s, AV_LOG_ERROR, "output name of the model network is not specified\n");
        return AVERROR(EINVAL);
    }

    FFInferenceParam param = { };

    param.model_filename  = s->model_filename;
    param.model_inputname = s->model_inputname;
    param.model_outputname = s->model_outputname;
    param.batch_size      = s->batch_size;
    param.backend_type    = DNN_OV; // FIXME: temp for testing

    s->dnn_interface = ff_dnn_interface_create(context->filter->name, &param, context);
    if (!s->dnn_interface) {
        av_log(s, AV_LOG_ERROR, "Could not create inference.\n");
        return AVERROR(EINVAL);
    }

    ff_dnn_interface_set_pre_proc(s->dnn_interface, (DNNPreProc)pre_proc);
    ff_dnn_interface_set_post_proc(s->dnn_interface, (DNNPostProc)post_proc);

    return 0;
}

static int query_formats(AVFilterContext *context)
{
    static const enum AVPixelFormat pix_fmts[] = {
        AV_PIX_FMT_RGB24, AV_PIX_FMT_BGR24,
        AV_PIX_FMT_GRAY8, AV_PIX_FMT_GRAYF32,
        AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV422P,
        AV_PIX_FMT_YUV444P, AV_PIX_FMT_YUV410P, AV_PIX_FMT_YUV411P,
        AV_PIX_FMT_NONE
    };
    AVFilterFormats *fmts_list = ff_make_format_list(pix_fmts);
    return ff_set_common_formats(context, fmts_list);
}

#define LOG_FORMAT_CHANNEL_MISMATCH()                       \
    av_log(ctx, AV_LOG_ERROR,                               \
           "the frame's format %s does not match "          \
           "the model input channel %d\n",                  \
           av_get_pix_fmt_name(fmt),                        \
           model_input->channels);

static int check_modelinput_inlink(const DNNData *model_input, const AVFilterLink *inlink)
{
    AVFilterContext *ctx   = inlink->dst;
    enum AVPixelFormat fmt = inlink->format;

    // the design is to add explicit scale filter before this filter
    if (model_input->height != -1 && model_input->height != inlink->h) {
        av_log(ctx, AV_LOG_ERROR, "the model requires frame height %d but got %d\n",
                                   model_input->height, inlink->h);
        return AVERROR(EIO);
    }
    if (model_input->width != -1 && model_input->width != inlink->w) {
        av_log(ctx, AV_LOG_ERROR, "the model requires frame width %d but got %d\n",
                                   model_input->width, inlink->w);
        return AVERROR(EIO);
    }

    switch (fmt) {
    case AV_PIX_FMT_RGB24:
    case AV_PIX_FMT_BGR24:
        if (model_input->channels != 3) {
            LOG_FORMAT_CHANNEL_MISMATCH();
            return AVERROR(EIO);
        }
        if (model_input->dt != DNN_FLOAT && model_input->dt != DNN_UINT8) {
            av_log(ctx, AV_LOG_ERROR, "only support dnn models with input data type as float32 and uint8.\n");
            return AVERROR(EIO);
        }
        return 0;
    case AV_PIX_FMT_GRAY8:
        if (model_input->channels != 1) {
            LOG_FORMAT_CHANNEL_MISMATCH();
            return AVERROR(EIO);
        }
        if (model_input->dt != DNN_UINT8) {
            av_log(ctx, AV_LOG_ERROR, "only support dnn models with input data type uint8.\n");
            return AVERROR(EIO);
        }
        return 0;
    case AV_PIX_FMT_GRAYF32:
    case AV_PIX_FMT_YUV420P:
    case AV_PIX_FMT_YUV422P:
    case AV_PIX_FMT_YUV444P:
    case AV_PIX_FMT_YUV410P:
    case AV_PIX_FMT_YUV411P:
        if (model_input->channels != 1) {
            LOG_FORMAT_CHANNEL_MISMATCH();
            return AVERROR(EIO);
        }
        if (model_input->dt != DNN_FLOAT) {
            av_log(ctx, AV_LOG_ERROR, "only support dnn models with input data type float32.\n");
            return AVERROR(EIO);
        }
        return 0;
    default:
        av_log(ctx, AV_LOG_ERROR, "%s not supported.\n", av_get_pix_fmt_name(fmt));
        return AVERROR(EIO);
    }

    return 0;
}

static int flush_frame(AVFilterContext *ctx, AVFilterLink *outlink, int64_t pts, int64_t *out_pts)
{
    int ret = 0;
    DnnProcessing2Context *s = ctx->priv;

    if (s->already_flushed)
        return ret;

    while (!ff_dnn_interface_frame_queue_empty(s->dnn_interface)) {
        AVFrame *output = NULL;
        ff_dnn_interface_get_frame(s->dnn_interface, &output);
        if (output) {
            if (outlink) {
                ret = ff_filter_frame(outlink, output);
                if (out_pts)
                    *out_pts = output->pts + pts;
            } else {
                av_frame_free(&output);
            }
        }

        av_usleep(5000);
    }

    s->already_flushed = 1;
    return ret;
}

static int activate(AVFilterContext *ctx)
{
    AVFilterLink *inlink = ctx->inputs[0];
    AVFilterLink *outlink = ctx->outputs[0];
    DnnProcessing2Context *s = ctx->priv;
    AVFrame *in = NULL, *output = NULL;
    int64_t pts;
    int ret, status;
    int got_frame = 0;

    FF_FILTER_FORWARD_STATUS_BACK(outlink, inlink);

    do {
        int get_frame_status;
        // drain all input frames
        ret = ff_inlink_consume_frame(inlink, &in);
        if (ret < 0)
            return ret;
        if (ret > 0)
           ff_dnn_interface_send_frame(s->dnn_interface, in);

        // drain all processed frames
        do {
            get_frame_status = ff_dnn_interface_get_frame(s->dnn_interface, &output);
            if (output) {
                int ret_val = ff_filter_frame(outlink, output);
                if (ret_val < 0)
                    return ret_val;

                got_frame = 1;
                output = NULL;
            }
        } while (get_frame_status == 0);
    } while (ret > 0);

    // if frame got, schedule to next filter
    if (got_frame)
        return 0;

    if (ff_inlink_acknowledge_status(inlink, &status, &pts)) {
        if (status == AVERROR_EOF) {
            int64_t out_pts = pts;

            av_log(ctx, AV_LOG_INFO, "Get EOS.\n");
            ret = flush_frame(ctx, outlink, pts, &out_pts);
            ff_outlink_set_status(outlink, status, out_pts);
            return ret;
        }
    }

    FF_FILTER_FORWARD_WANTED(outlink, inlink);

    return FFERROR_NOT_READY;
}

static av_cold void uninit(AVFilterContext *ctx)
{
    DnnProcessing2Context *s = ctx->priv;

    sws_freeContext(s->sws_resize);
    sws_freeContext(s->sws_gray8_to_grayf32);
    sws_freeContext(s->sws_grayf32_to_gray8);
    sws_freeContext(s->sws_uv_scale);

    ff_dnn_interface_release(s->dnn_interface);
}

static int config_input(AVFilterLink *inlink)
{
    AVFilterContext *context     = inlink->dst;
    DnnProcessingsContext *ctx = context->priv;
    DNNReturnType result;
    DNNData model_input;
    int check;

    result = ctx->dnn_interface->model->get_input(ctx->dnn_interface->model->model, &model_input, ctx->model_inputname);
    if (result != DNN_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "could not get input from the model\n");
        return AVERROR(EIO);
    }

    check = check_modelinput_inlink(&model_input, inlink);
    if (check != 0) {
        return check;
    }

    // FIXME: do we need this for sync mode?
    /*
    ctx->input.width    = inlink->w;
    ctx->input.height   = inlink->h;
    ctx->input.channels = model_input.channels;
    ctx->input.dt = model_input.dt;

    result = (ctx->model->set_input_output)(ctx->model->model,
                                        &ctx->input, ctx->model_inputname,
                                        (const char **)&ctx->model_outputname, 1);
    if (result != DNN_SUCCESS) {
        av_log(ctx, AV_LOG_ERROR, "could not set input and output for the model\n");
        return AVERROR(EIO);
    }
    */

    return 0;
}

static const AVFilterPad dnn_processing2_inputs[] = {
    {
        .name         = "default",
        .type         = AVMEDIA_TYPE_VIDEO,
        .config_props = config_input,
    },
    { NULL }
};

static const AVFilterPad dnn_processing2_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props  = config_output,
    },
    { NULL }
};

AVFilter ff_vf_dnn_processing2 = {
    .name          = "dnn_processing2",
    .description   = NULL_IF_CONFIG_SMALL("Apply DNN processing filter to the input."),
    .priv_size     = sizeof(DnnProcessing2Context),
    .init          = init,
    .uninit        = uninit,
    .query_formats = query_formats,
    .activate      = activate,
    .inputs        = dnn_processing2_inputs,
    .outputs       = dnn_processing2_outputs,
    .priv_class    = &dnn_processing2_class,
    .flags_internal = FF_FILTER_FLAG_HWFRAME_AWARE, // TODO: What is this for?
};
