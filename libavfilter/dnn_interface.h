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

typedef enum {
   DAST_FAIL = -2,        // something wrong
   DAST_EMPTY_QUEUE = -1, // no more inference result to get
   DAST_NOT_READY,        // all inferences are not finished
   DAST_SUCCESS           // got a result frame successfully
} DNNAsyncStatusType;

typedef enum {DNN_SUCCESS, DNN_ERROR} DNNReturnType;

typedef enum {DNN_NATIVE, DNN_TF, DNN_OV} DNNBackendType;

typedef enum {DNN_FLOAT = 1, DNN_UINT8 = 4} DNNDataType;

typedef struct DNNData{
    void *data;
    DNNDataType dt;
    int width, height, channels, batch_size;
} DNNData;

typedef struct __ProcessingFrame ProcessingFrame;

// model-specific post proc function.
// Parse inference result and store the result in frame_in and frame_out_p
// For dnn processing filter, it may generate a new frame and return it using frame_out_p.
// For analytic filter, store the inference result as side data of frame_in and make the *frame_out_p to ref the frame_in.
typedef int (*DNNPostProc2)(DNNData *model_output, AVFrame *frame_in, AVFrame **frame_out_p, void *user_data);

// model-specific pre proc function.
// Convert and then copy the data in frame_in to model_input
typedef int (*DNNPreProc2)(AVFrame *frame_in, DNNData *model_input, void *user_data);

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

    DNNReturnType (*get_output)(void *model, DNNData *output, const char *output_name);
    DNNPreProc2 pre_proc;
    DNNPostProc2 post_proc;
    AVFilterContext *filter_ctx;
} DNNModel;

// Stores pointers to functions for loading, executing, freeing DNN models for one of the backends.
typedef struct DNNModule{
    // Loads model and parameters from given file. Returns NULL if it is not possible.
    DNNModel *(*load_model)(const char *model_filename, const char *options);
    // Executes model with specified input and output. Returns DNN_ERROR otherwise.
    DNNReturnType (*execute_model)(const DNNModel *model, DNNData *outputs, uint32_t nb_output);
    // Frees memory allocated for model.
    void (*free_model)(DNNModel **model);

    DNNModel *(*load_model2)(const char *model_filename, const char *options, void *user_data);
    DNNReturnType (*execute_model2)(const DNNModel *model, AVFrame *in, const char *model_input_name, AVFrame **out, const char **output_names, uint32_t nb_output);
    // Executes model asynchronously. Release inference_ctx when execution done
    DNNReturnType (*execute_model_async2)(const DNNModel *model, AVFrame *in, const char *model_input_name, const char **output_names, uint32_t nb_output);
    DNNReturnType (*execute_model_async_batch)(const DNNModel *model, AVFrame *in, const char *model_input_name,
                                               const char **output_names, uint32_t nb_output);
    void (*flush)(const DNNModel *model);
    DNNAsyncStatusType (*get_async_result)(const DNNModel *model, AVFrame **out);

} DNNModule;

// Initializes DNNModule depending on chosen backend.
DNNModule *ff_get_dnn_module(DNNBackendType backend_type);

struct __ProcessingFrame {
    AVFrame *frame_in;
    AVFrame *frame_out;
    int inference_done;
};

#endif
