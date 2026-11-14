#include "esp_timer.h"
#include <stdio.h>
#include <stdint.h>
#include "esp_log.h"
#include "dl_model_base.hpp"

static const char *TAG = "FALL_MODEL";

extern const uint8_t model_espdl[] asm("_binary_model_espdl_start");

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting CSI Fall Detection inference...");

    // 1. Load the model
    dl::Model *model = new dl::Model(
        (const char *)model_espdl,
        fbs::MODEL_LOCATION_IN_FLASH_RODATA
    );

    if (model == nullptr) {
        ESP_LOGE(TAG, "Model allocation failed");
        return;
    }

    ESP_LOGI(TAG, "Model loaded successfully");

    // 2. Get model input and output tensors
    auto &inputs = model->get_inputs();
    auto &outputs = model->get_outputs();

    if (inputs.empty() || outputs.empty()) {
        ESP_LOGE(TAG, "Model input/output tensors not available");
        delete model;
        return;
    }

    dl::TensorBase *model_input = inputs.begin()->second;
    dl::TensorBase *model_output = outputs.begin()->second;

    // 3. Validate tensor metadata
    ESP_LOGI(TAG, "Input shape elements: %d", model_input->get_size());
    ESP_LOGI(TAG, "Input exponent: %d", model_input->get_exponent());
    ESP_LOGI(TAG, "Input dtype: %s", model_input->get_dtype_string());

    ESP_LOGI(TAG, "Output shape elements: %d", model_output->get_size());
    ESP_LOGI(TAG, "Output exponent: %d", model_output->get_exponent());
    ESP_LOGI(TAG, "Output dtype: %s", model_output->get_dtype_string());

    if (model_input->get_size() != 5200 ||
        model_output->get_size() != 2 ||
        model_input->get_dtype() != dl::DATA_TYPE_INT8 ||
        model_output->get_dtype() != dl::DATA_TYPE_INT8) {

        ESP_LOGE(TAG, "Unexpected model tensor configuration");
        delete model;
        return;
    }

    // 4. Initialize synthetic input
    // Zero represents normalized input equal to the training mean.
    int8_t *input_ptr = model_input->get_element_ptr<int8_t>();

    for (int i = 0; i < model_input->get_size(); i++) {
        input_ptr[i] = 0;
    }

    ESP_LOGI(TAG, "Input tensor initialized with 5200 zero values");

    // 5. Execute inference
    ESP_LOGI(TAG, "Running model inference...");

    int64_t start_time = esp_timer_get_time();

    model->run();

    int64_t inference_time = esp_timer_get_time() - start_time;

    ESP_LOGI(TAG, "Inference completed in %lld ms",
             (long long)(inference_time / 1000));

    // 6. Read and dequantize output
    int8_t *output_ptr = model_output->get_element_ptr<int8_t>();

    float fall_score = dl::dequantize(
        output_ptr[0],
        DL_SCALE(model_output->get_exponent())
    );

    float nonfall_score = dl::dequantize(
        output_ptr[1],
        DL_SCALE(model_output->get_exponent())
    );

    ESP_LOGI(TAG, "Fall score: %.5f", fall_score);
    ESP_LOGI(TAG, "Non-Fall score: %.5f", nonfall_score);

    // 7. Classify
    if (fall_score > nonfall_score) {
        ESP_LOGI(TAG, "Prediction: FALL");
    } else {
        ESP_LOGI(TAG, "Prediction: NON-FALL");
    }

    ESP_LOGI(TAG, "Static inference test completed");

    // 8. Release model memory
    delete model;
}
