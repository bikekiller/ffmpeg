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
 * DNN inference engine interface.
 */

#ifndef AVFILTER_DNN_INTERFACE_H
#define AVFILTER_DNN_INTERFACE_H

#include <stdint.h>
#include <libavutil/frame.h>
#include "avfilter.h"
#include "dnn/ff_list.h"

typedef enum {DNN_SUCCESS, DNN_ERROR} DNNReturnType;

typedef enum {DNN_NATIVE, DNN_TF, DNN_OV} DNNBackendType;

typedef enum {DNN_FLOAT = 1, DNN_UINT8 = 4} DNNDataType;

typedef struct DNNData{
    void *data;
    DNNDataType dt;
    int width, height, channels;
} DNNData;

typedef struct __DnnInterface DnnInterface;
typedef struct __InferenceParam InferenceParam;
typedef struct __InferenceContext InferenceContext;
typedef struct __ProcessingFrame ProcessingFrame;

// callback to call model-specific post proc and update the internal frame queue
typedef void (*InferCallback)(DNNData *out_blob, ProcessingFrame *processing_frame, DnnInterface *dnn_interface);

// model-specific post proc function.
// Parse inference result and store the result in frame_in and frame_out_p
// For dnn processing filter, it may generate a new frame and return it using frame_out_p.
// For analytic filter, store the inference result as side data of frame_in and make the *frame_out_p to ref the frame_in.
typedef int (*DNNPostProc)(DNNData *model_output, AVFrame *frame_in, AVFrame **frame_out_p, DnnInterface *dnn_interface);

// model-specific pre proc function.
// Convert and then copy the data in frame_in to model_input
typedef int (*DNNPreProc)(AVFrame *frame_in, DNNData *model_input, DnnInterface *dnn_interface);

typedef struct DNNModel{
    // Stores model that can be different for different backends.
    void *model;
    // Stores options when the model is executed by the backend
    const char *options;
    // Gets model input information
    // Just reuse struct DNNData here, actually the DNNData.data field is not needed.
    DNNReturnType (*get_input)(void *model, DNNData *input, const char *input_name);
    // Gets model input blob, DNNData.data is needed to copy input data into model
    DNNReturnType (*get_input_blob)(void *model, DNNData *input, const char *input_name);
    // Sets model input and output.
    // Should be called at least once before model execution.
    DNNReturnType (*set_input_output)(void *model, DNNData *input, const char *input_name, const char **output_names, uint32_t nb_output);
} DNNModel;

// Stores pointers to functions for loading, executing, freeing DNN models for one of the backends.
typedef struct DNNModule{
    // Loads model and parameters from given file. Returns NULL if it is not possible.
    DNNModel *(*load_model)(const char *model_filename, const char *options);
    // Executes model with specified input and output. Returns DNN_ERROR otherwise.
    DNNReturnType (*execute_model)(const DNNModel *model, DNNData *outputs, uint32_t nb_output);
    // Executes model asynchronously. Release inference_ctx when execution done
    DNNReturnType (*execute_model_async)(const DNNModel *model, InferenceContext *inference_ctx, const char *output_name);
    // Frees memory allocated for model.
    void (*free_model)(DNNModel **model);
} DNNModule;

// Initializes DNNModule depending on chosen backend.
DNNModule *ff_get_dnn_module(DNNBackendType backend_type);

struct __ProcessingFrame {
    AVFrame *frame_in;
    AVFrame *frame_out;
    int inference_done;
};

struct __InferenceContext{
   ProcessingFrame *processing_frame;
   InferCallback cb;
   DnnInterface *dnn_interface;
};

struct __InferenceParam {
    char *model_filename;                                                                                                       
    char *model_inputname;
    char *model_outputname;
    int async;
    int nireq;
    int batch_size;                                                                                                    
    int backend_type;                                                                                                    
};

struct __DnnInterface {
    AVFilterContext *filter_ctx;
    // unique infer string id
    const char *inference_id;
    DNNModule *dnn_module;
    DNNModel *model;

    InferenceParam param;
    int async_run;
    DNNPreProc pre_proc;         // type: PreProcFunction
    DNNPostProc post_proc;        // type: PostProcFunction

    pthread_mutex_t frame_q_mutex;
    ff_list_t *processing_frames;
    ff_list_t *processed_frames;
};

DnnInterface *dnn_interface_create(const char *inference_id, InferenceParam *param, AVFilterContext *filter_ctx);

void dnn_interface_set_pre_proc(DnnInterface *dnn_interface, DNNPreProc pre_proc);

void dnn_interface_set_post_proc(DnnInterface *dnn_interface, DNNPostProc post_proc);

void dnn_interface_release(DnnInterface *dnn_interface);

int dnn_interface_send_frame(DnnInterface *dnn_interface, AVFrame *frame_in);

int dnn_interface_get_frame(DnnInterface *dnn_interface, AVFrame **frame_out);

int dnn_interface_frame_queue_empty(DnnInterface *dnn_interface);

#endif
