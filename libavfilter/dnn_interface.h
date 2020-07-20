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

typedef enum {
    INFERENCE_EVENT_NONE,
    INFERENCE_EVENT_EOS,
} FF_INFERENCE_EVENT;

typedef enum {DNN_SUCCESS, DNN_ERROR} DNNReturnType;

typedef enum {DNN_NATIVE, DNN_TF, DNN_OV} DNNBackendType;

typedef enum {DNN_FLOAT = 1, DNN_UINT8 = 4} DNNDataType;

typedef struct DNNData{
    void *data;
    DNNDataType dt;
    int width, height, channels;
} DNNData;

typedef struct DNNModel{
    // Stores model that can be different for different backends.
    void *model;
    // Stores options when the model is executed by the backend
    const char *options;
    // Gets model input information
    // Just reuse struct DNNData here, actually the DNNData.data field is not needed.
    DNNReturnType (*get_input)(void *model, DNNData *input, const char *input_name);
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
    // Frees memory allocated for model.
    void (*free_model)(DNNModel **model);
} DNNModule;

// Initializes DNNModule depending on chosen backend.
DNNModule *ff_get_dnn_module(DNNBackendType backend_type);


typedef struct __FFBaseInference FFBaseInference;
typedef struct __FFInferenceParam FFInferenceParam;

struct __FFInferenceParam {
    char *model;                                                                                                       
    char *model_inputname;
    char *model_outputname;
    int batch_size;                                                                                                    
    int backend;                                                                                                    
};

struct __FFBaseInference {
    // unique infer string id
    const char *inference_id;
    FFInferenceParam param;
    void *pre_proc;         // type: PreProcFunction
    void *post_proc;        // type: PostProcFunction

    DNNModule *dnn_module;
    DNNModel *model;

    // input & output of the model at execution time
    DNNData input;
    DNNData output;

    struct SwsContext *sws_gray8_to_grayf32;
    struct SwsContext *sws_grayf32_to_gray8;
    struct SwsContext *sws_uv_scale;
    int sws_uv_height;
};

FFBaseInference *ff_dnn_interface_create(const char *inference_id, FFInferenceParam *param);

void ff_dnn_interface_release(FFBaseInference *base);

int ff_dnn_interface_send_frame(FFBaseInference *base, AVFrame *frame);

int ff_dnn_interface_get_frame(FFBaseInference *base, AVFrame **frame_out);

int ff_dnn_interface_frame_queue_empty(FFBaseInference *base);

int ff_dnn_interface_resource_status(FFBaseInference *base);

void ff_dnn_interface_send_event(FFBaseInference *base, FF_INFERENCE_EVENT event);
#endif
