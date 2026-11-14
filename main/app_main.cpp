#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <math.h>

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "freertos/queue.h"
#include "freertos/semphr.h"

#include "esp_log.h"
#include "esp_err.h"
#include "esp_timer.h"
#include "nvs_flash.h"
#include "esp_netif.h"
#include "esp_event.h"
#include "esp_wifi.h"

#include "lwip/sockets.h"
#include "lwip/inet.h"

#include "dl_model_base.hpp"
#include "fbs_loader.hpp"

#include "wifi_credentials.h"

static const char *TAG = "CSI_FALL_DET";

// --------------------------------------------------
// EMBEDDED ESP-DL MODEL BINARY SYMBOLS
// --------------------------------------------------
extern const char model_espdl_start[] asm("_binary_model_espdl_start");
extern const char model_espdl_end[]   asm("_binary_model_espdl_end");

// --------------------------------------------------
// MODEL & PREPROCESSING CONSTANTS (FROM TRAINING)
// --------------------------------------------------
#define NUM_SUBCARRIERS   52
#define WINDOW_SIZE       100
#define STRIDE_SIZE       50
#define INPUT_ELEMENTS    (NUM_SUBCARRIERS * WINDOW_SIZE)

// Calibration packets collected at startup (200 pkts = ~2.0 s)
#define CALIB_PKT_COUNT   200

#define TRAIN_MEAN        8.26974868774414f
#define TRAIN_STD         3.6846582889556885f

// Calibrated thresholds with Per-Packet AGC Normalization active:
#define FALL_LOGIT_THRESHOLD       3.50f
#define FALL_MOTION_STD_THRESHOLD  0.62f
#define FALL_P2P_THRESHOLD         6.20f

#define CSI_QUEUE_LENGTH  64
#define CSI_BUFFER_SIZE   128

typedef struct {
    int8_t rssi;
    int8_t noise_floor;
    uint8_t sig_mode;
    uint16_t original_len;
    uint16_t copied_len;
    int8_t data[CSI_BUFFER_SIZE];
} csi_packet_t;

static QueueHandle_t csi_queue = NULL;
static SemaphoreHandle_t window_mutex = NULL;
static TaskHandle_t inference_task_handle = NULL;

static volatile uint32_t g_gw_ip_addr = 0;
static volatile bool g_wifi_connected = false;

// Per-subcarrier adaptive static baseline
static float g_sc_static_mean[NUM_SUBCARRIERS];
static volatile bool g_calibrated = false;

// Sliding window buffers in [52][100] layout
static float s_OrderWindow[NUM_SUBCARRIERS][WINDOW_SIZE];
static float s_ReadyWindow[NUM_SUBCARRIERS][WINDOW_SIZE];
static volatile uint32_t s_LatestWindowId = 0;
static volatile int8_t s_LatestRssi = 0;
static volatile float s_ReadyWindowPeakStd = -1.0f;

// --------------------------------------------------
// BACKGROUND TRAFFIC GENERATOR TASK (~100 Hz)
// --------------------------------------------------

static void csi_traffic_generator_task(void *arg)
{
    int sock = -1;

    const uint8_t dns_query[] = {
        0x12, 0x34, 0x01, 0x00, 0x00, 0x01, 0x00, 0x00,
        0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x01
    };

    while (true) {
        if (!g_wifi_connected || g_gw_ip_addr == 0) {
            vTaskDelay(pdMS_TO_TICKS(200));
            continue;
        }

        if (sock < 0) {
            sock = socket(AF_INET, SOCK_DGRAM, IPPROTO_UDP);
            if (sock < 0) {
                vTaskDelay(pdMS_TO_TICKS(500));
                continue;
            }
            int flags = fcntl(sock, F_GETFL, 0);
            fcntl(sock, F_SETFL, flags | O_NONBLOCK);
        }

        struct sockaddr_in dest_addr = {};
        dest_addr.sin_family = AF_INET;
        dest_addr.sin_port = htons(53);
        dest_addr.sin_addr.s_addr = g_gw_ip_addr;

        sendto(sock, dns_query, sizeof(dns_query), 0,
               (struct sockaddr *)&dest_addr, sizeof(dest_addr));

        uint8_t rx_trash[128];
        while (recv(sock, rx_trash, sizeof(rx_trash), 0) > 0) {
            // Drain incoming UDP responses
        }

        vTaskDelay(pdMS_TO_TICKS(15));
    }
}

// --------------------------------------------------
// CSI CALLBACK
// --------------------------------------------------

static void wifi_csi_callback(void *ctx, wifi_csi_info_t *info)
{
    if (info == NULL || info->buf == NULL || info->len < 128) {
        return;
    }

    csi_packet_t packet;
    packet.rssi = info->rx_ctrl.rssi;
    packet.noise_floor = info->rx_ctrl.noise_floor;
    packet.sig_mode = info->rx_ctrl.sig_mode;
    packet.original_len = info->len;
    packet.copied_len = 128;

    memcpy(packet.data, info->buf, 128);

    xQueueSend(csi_queue, &packet, 0);
}

// --------------------------------------------------
// HELPER: EXTRACT 52 ACTIVE SUBCARRIERS IN CSI-BENCH ORDER
// --------------------------------------------------

static inline void extract_52_subcarriers(const int8_t *raw_data, float *out_subcarriers)
{
    int sc_idx = 0;

    // Lower 26 active subcarriers: DMA indices 38..63 -> CSI-Bench indices 6..31 (-26..-1)
    for (int i = 38; i <= 63; i++) {
        float imag = (float)raw_data[2 * i];
        float real = (float)raw_data[2 * i + 1];
        out_subcarriers[sc_idx++] = sqrtf(real * real + imag * imag);
    }

    // Upper 26 active subcarriers: DMA indices 1..26 -> CSI-Bench indices 33..58 (+1..+26)
    for (int i = 1; i <= 26; i++) {
        float imag = (float)raw_data[2 * i];
        float real = (float)raw_data[2 * i + 1];
        out_subcarriers[sc_idx++] = sqrtf(real * real + imag * imag);
    }
}

// --------------------------------------------------
// HELPER: COMPUTE TEMPORAL STD OF CURRENT WINDOW
// --------------------------------------------------

static float compute_window_temporal_std(const float win[NUM_SUBCARRIERS][WINDOW_SIZE])
{
    float total_std = 0.0f;
    for (int s = 0; s < NUM_SUBCARRIERS; s++) {
        float sum = 0.0f;
        float sq_sum = 0.0f;
        for (int t = 0; t < WINDOW_SIZE; t++) {
            float v = win[s][t];
            sum += v;
            sq_sum += v * v;
        }
        float mean = sum / (float)WINDOW_SIZE;
        float var = (sq_sum / (float)WINDOW_SIZE) - (mean * mean);
        total_std += sqrtf(fmaxf(var, 0.0f));
    }
    return total_std / (float)NUM_SUBCARRIERS;
}

// --------------------------------------------------
// CSI PROCESSING & SLIDING WINDOW TASK (CORE 0)
// --------------------------------------------------

static void csi_processing_task(void *arg)
{
    csi_packet_t packet;
    uint32_t calib_count = 0;
    static double calib_sc_sum[NUM_SUBCARRIERS] = {0};

    uint32_t packet_count = 0;
    uint32_t stride_counter = 0;
    uint32_t window_id = 0;

    float current_subcarriers[NUM_SUBCARRIERS];

    int64_t last_rate_time_us = esp_timer_get_time();
    uint32_t last_rate_count = 0;

    ESP_LOGI(TAG, "Step 1: Calibrating room & fan background for 2 seconds (200 packets)...");

    while (true) {
        if (xQueueReceive(csi_queue, &packet, portMAX_DELAY) == pdTRUE) {
            extract_52_subcarriers(packet.data, current_subcarriers);

            // Step A: Strict Per-Packet AGC Power Normalization
            float pkt_mean = 0.0f;
            for (int s = 0; s < NUM_SUBCARRIERS; s++) {
                pkt_mean += current_subcarriers[s];
            }
            pkt_mean /= (float)NUM_SUBCARRIERS;

            float pkt_gain = (pkt_mean > 0.5f) ? (TRAIN_MEAN / pkt_mean) : 1.0f;
            for (int s = 0; s < NUM_SUBCARRIERS; s++) {
                current_subcarriers[s] *= pkt_gain;
            }

            // Phase 1: Initial 2-Second Static Baseline Calibration
            if (!g_calibrated) {
                for (int s = 0; s < NUM_SUBCARRIERS; s++) {
                    calib_sc_sum[s] += (double)current_subcarriers[s];
                }
                calib_count++;

                if (calib_count >= CALIB_PKT_COUNT) {
                    for (int s = 0; s < NUM_SUBCARRIERS; s++) {
                        g_sc_static_mean[s] = (float)(calib_sc_sum[s] / (double)CALIB_PKT_COUNT);
                    }

                    g_calibrated = true;
                    last_rate_time_us = esp_timer_get_time();

                    ESP_LOGI(TAG, "============================================");
                    ESP_LOGI(TAG, "Room & Fan Baseline Calibration Complete!");
                    ESP_LOGI(TAG, "Target Mean: %.3f | RSSI: %d dBm", TRAIN_MEAN, packet.rssi);
                    ESP_LOGI(TAG, "============================================");
                }
                continue;
            }

            // Step B: Adaptive Static Background Subtraction (alpha = 0.025)
            for (int s = 0; s < NUM_SUBCARRIERS; s++) {
                float val = current_subcarriers[s];
                float diff = val - g_sc_static_mean[s];

                g_sc_static_mean[s] += 0.025f * diff;

                current_subcarriers[s] = TRAIN_MEAN + diff;
            }

            // Emit 8 evenly-spaced subcarrier waves at 25 Hz (every 4th packet) for Live Plotter
            if (packet_count % 4 == 0) {
                printf("CSI_WAVE:%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f\n",
                       current_subcarriers[0],
                       current_subcarriers[7],
                       current_subcarriers[14],
                       current_subcarriers[21],
                       current_subcarriers[28],
                       current_subcarriers[35],
                       current_subcarriers[42],
                       current_subcarriers[49]);
            }

            // Phase 2: Sliding Window Buffering (Window = 100, Stride = 50)
            if (packet_count < WINDOW_SIZE) {
                for (int s = 0; s < NUM_SUBCARRIERS; s++) {
                    s_OrderWindow[s][packet_count] = current_subcarriers[s];
                }
                packet_count++;

                if (packet_count == WINDOW_SIZE) {
                    window_id++;
                    float cand_std = compute_window_temporal_std(s_OrderWindow);
                    if (xSemaphoreTake(window_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
                        memcpy(s_ReadyWindow, s_OrderWindow, sizeof(s_OrderWindow));
                        s_LatestWindowId = window_id;
                        s_LatestRssi = packet.rssi;
                        s_ReadyWindowPeakStd = cand_std;
                        xSemaphoreGive(window_mutex);
                    }
                    if (inference_task_handle != NULL) {
                        xTaskNotifyGive(inference_task_handle);
                    }
                }
            } else {
                for (int s = 0; s < NUM_SUBCARRIERS; s++) {
                    memmove(&s_OrderWindow[s][0],
                            &s_OrderWindow[s][1],
                            (WINDOW_SIZE - 1) * sizeof(float));
                    s_OrderWindow[s][WINDOW_SIZE - 1] = current_subcarriers[s];
                }

                packet_count++;
                stride_counter++;

                if (stride_counter >= STRIDE_SIZE) {
                    stride_counter = 0;
                    window_id++;

                    float cand_std = compute_window_temporal_std(s_OrderWindow);

                    // Peak-Hold Latch while Core 1 is busy running inference
                    if (xSemaphoreTake(window_mutex, pdMS_TO_TICKS(5)) == pdTRUE) {
                        if (s_ReadyWindowPeakStd < 0.0f || cand_std >= s_ReadyWindowPeakStd) {
                            memcpy(s_ReadyWindow, s_OrderWindow, sizeof(s_OrderWindow));
                            s_LatestWindowId = window_id;
                            s_LatestRssi = packet.rssi;
                            s_ReadyWindowPeakStd = cand_std;
                        }
                        xSemaphoreGive(window_mutex);
                    }
                    if (inference_task_handle != NULL) {
                        xTaskNotifyGive(inference_task_handle);
                    }
                }
            }

            if (packet_count % 200 == 0) {
                int64_t now_us = esp_timer_get_time();
                float elapsed_s = (float)(now_us - last_rate_time_us) / 1000000.0f;
                float rate_hz = (elapsed_s > 0.0f)
                    ? ((float)(packet_count - last_rate_count) / elapsed_s)
                    : 0.0f;
                last_rate_time_us = now_us;
                last_rate_count = packet_count;

                ESP_LOGI(TAG, "[CSI RX] Pkts: %lu | Rate: %.1f Hz | RSSI: %d dBm",
                         (unsigned long)packet_count, rate_hz, packet.rssi);
            }
        }
    }
}

// --------------------------------------------------
// ESP-DL REAL-TIME INFERENCE TASK (PINNED TO CORE 1)
// --------------------------------------------------

static void model_inference_task(void *arg)
{
    ESP_LOGI(TAG, "Loading ESP-DL model on Core %d...", xPortGetCoreID());

    dl::Model *model = new dl::Model(
        model_espdl_start,
        fbs::MODEL_LOCATION_IN_FLASH_RODATA
    );

    if (model == NULL) {
        ESP_LOGE(TAG, "Failed to load ESP-DL model!");
        vTaskDelete(NULL);
        return;
    }

    std::map<std::string, dl::TensorBase *> inputs = model->get_inputs();
    std::map<std::string, dl::TensorBase *> outputs = model->get_outputs();

    dl::TensorBase *input_tensor = inputs.begin()->second;
    dl::TensorBase *output_tensor = outputs.begin()->second;

    int input_exponent = input_tensor->exponent;
    int output_exponent = output_tensor->exponent;

    float input_scale = powf(2.0f, (float)input_exponent);   // 2^-5 = 0.03125
    float inv_input_scale = 1.0f / input_scale;              // 32.0
    float output_scale = powf(2.0f, (float)output_exponent); // 2^-4 = 0.0625

    ESP_LOGI(TAG, "Model loaded successfully!");
    if (input_tensor->shape.size() == 4) {
        ESP_LOGI(TAG, "Input shape    : [%d, %d, %d, %d] (%d elements) | exp: %d",
                 input_tensor->shape[0], input_tensor->shape[1],
                 input_tensor->shape[2], input_tensor->shape[3],
                 input_tensor->get_size(), input_exponent);
    }
    ESP_LOGI(TAG, "Output elements: %d | exp: %d (scale=%.5f)",
             output_tensor->get_size(), output_exponent, output_scale);

    static float local_window[NUM_SUBCARRIERS][WINDOW_SIZE];
    bool fall_cooldown_active = false;

    while (true) {
        // Wait for a completed 100-sample sliding window
        ulTaskNotifyTake(pdTRUE, portMAX_DELAY);

        uint32_t eval_window_id = 0;
        int8_t eval_rssi = 0;

        if (xSemaphoreTake(window_mutex, portMAX_DELAY) == pdTRUE) {
            memcpy(local_window, s_ReadyWindow, sizeof(local_window));
            eval_window_id = s_LatestWindowId;
            eval_rssi = s_LatestRssi;
            s_ReadyWindowPeakStd = -1.0f;
            xSemaphoreGive(window_mutex);
        }

        int8_t *in_ptr = (int8_t *)input_tensor->get_element_ptr();

        float win_sum = 0.0f;
        float win_sq_sum = 0.0f;
        float win_min = 1000000.0f;
        float win_max = 0.0f;

        float temporal_std_sum = 0.0f;

        int idx = 0;
        for (int s = 0; s < NUM_SUBCARRIERS; s++) {
            float sc_sum = 0.0f;
            float sc_sq_sum = 0.0f;

            for (int t = 0; t < WINDOW_SIZE; t++) {
                // 7-tap temporal low-pass filter
                int t0 = (t >= 3) ? (t - 3) : 0;
                int t1 = (t <= WINDOW_SIZE - 4) ? (t + 3) : (WINDOW_SIZE - 1);
                float sum_tap = 0.0f;
                int cnt_tap = 0;
                for (int k = t0; k <= t1; k++) {
                    sum_tap += local_window[s][k];
                    cnt_tap++;
                }
                float amp = sum_tap / (float)cnt_tap;

                win_sum += amp;
                win_sq_sum += amp * amp;
                sc_sum += amp;
                sc_sq_sum += amp * amp;

                if (amp < win_min) win_min = amp;
                if (amp > win_max) win_max = amp;

                // Exact training normalization: (X - train_mean) / train_std
                float x_norm = (amp - TRAIN_MEAN) / TRAIN_STD;

                // Quantize to INT8 using input_exponent (-5)
                float q_float = roundf(x_norm * inv_input_scale);
                if (q_float > 127.0f) {
                    q_float = 127.0f;
                } else if (q_float < -128.0f) {
                    q_float = -128.0f;
                }

                in_ptr[idx++] = (int8_t)q_float;
            }

            float sc_mean = sc_sum / (float)WINDOW_SIZE;
            float sc_var = (sc_sq_sum / (float)WINDOW_SIZE) - (sc_mean * sc_mean);
            temporal_std_sum += sqrtf(fmaxf(sc_var, 0.0f));
        }

        float win_mean = win_sum / (float)INPUT_ELEMENTS;
        float win_var = (win_sq_sum / (float)INPUT_ELEMENTS) - (win_mean * win_mean);
        float win_std = sqrtf(fmaxf(win_var, 0.0f));
        float motion_temporal_std = temporal_std_sum / (float)NUM_SUBCARRIERS;
        float p2p_swing = win_max - win_min;

        // Run ESP-DL neural network inference
        int64_t t_start = esp_timer_get_time();
        model->run();
        int64_t t_end = esp_timer_get_time();
        uint32_t infer_ms = (uint32_t)((t_end - t_start) / 1000);

        // Read and dequantize INT8 output logits: [0] = Fall, [1] = Nonfall
        int8_t *out_ptr = (int8_t *)output_tensor->get_element_ptr();
        int8_t raw_fall_int8 = out_ptr[0];
        int8_t raw_nonfall_int8 = out_ptr[1];

        float fall_score = (float)raw_fall_int8 * output_scale;
        float nonfall_score = (float)raw_nonfall_int8 * output_scale;

        // Softmax probability for Fall (Class 0)
        float max_logit = fmaxf(fall_score, nonfall_score);
        float exp_fall = expf(fall_score - max_logit);
        float exp_nonfall = expf(nonfall_score - max_logit);
        float fall_prob = exp_fall / (exp_fall + exp_nonfall);

        // Check if current window meets all neural + physical burst criteria
        bool burst_detected = (fall_score >= FALL_LOGIT_THRESHOLD) &&
                              (motion_temporal_std >= FALL_MOTION_STD_THRESHOLD) &&
                              (p2p_swing >= FALL_P2P_THRESHOLD);

        bool is_fall = false;
        if (burst_detected && !fall_cooldown_active) {
            is_fall = true;
            fall_cooldown_active = true;
        } else if (!burst_detected) {
            fall_cooldown_active = false;
        }

        const char *prediction = is_fall
            ? ">>> FALL DETECTED! <<<"
            : "NON-FALL";

        ESP_LOGI(TAG, "============================================");
        ESP_LOGI(TAG, "[Window #%lu] Inference time: %lu ms | RSSI: %d dBm",
                 (unsigned long)eval_window_id, (unsigned long)infer_ms, eval_rssi);
        ESP_LOGI(TAG, "Window CSI : Mean=%.2f, SpatialStd=%.2f, MotionStd=%.3f, P2P=%.2f (Min=%.2f, Max=%.2f)",
                 win_mean, win_std, motion_temporal_std, p2p_swing, win_min, win_max);
        ESP_LOGI(TAG, "Logits     : Fall=%.4f (q=%d) | Non-Fall=%.4f (q=%d)",
                 fall_score, (int)raw_fall_int8, nonfall_score, (int)raw_nonfall_int8);
        ESP_LOGI(TAG, "Result     : %s (Model Fall Prob: %.1f%%)",
                 prediction, fall_prob * 100.0f);
        ESP_LOGI(TAG, "============================================");

        // Emit structured status line for Python visualizer banner
        printf("CSI_PRED:%s,%.1f,%.3f,%.2f,%d\n",
               is_fall ? "FALL DETECTED!" : "NON-FALL",
               fall_prob * 100.0f,
               motion_temporal_std,
               p2p_swing,
               (int)eval_rssi);

        // Yield 20 ms to allow IDLE1 task on Core 1 to reset Task Watchdog Timer
        vTaskDelay(pdMS_TO_TICKS(20));
    }
}

// --------------------------------------------------
// WIFI EVENT HANDLER
// --------------------------------------------------

static void wifi_event_handler(
    void *arg,
    esp_event_base_t event_base,
    int32_t event_id,
    void *event_data)
{
    if (event_base == WIFI_EVENT &&
        event_id == WIFI_EVENT_STA_START) {

        ESP_LOGI(TAG, "Wi-Fi started. Connecting...");
        ESP_ERROR_CHECK(esp_wifi_connect());
    }

    else if (event_base == WIFI_EVENT &&
             event_id == WIFI_EVENT_STA_DISCONNECTED) {

        g_wifi_connected = false;

        wifi_event_sta_disconnected_t *event =
            (wifi_event_sta_disconnected_t *)event_data;

        ESP_LOGW(TAG,
                 "Wi-Fi disconnected! Reason: %d, RSSI: %d",
                 event->reason,
                 event->rssi);

        esp_wifi_connect();
    }

    else if (event_base == IP_EVENT &&
             event_id == IP_EVENT_STA_GOT_IP) {

        ip_event_got_ip_t *event =
            (ip_event_got_ip_t *)event_data;

        g_gw_ip_addr = event->ip_info.gw.addr;
        g_wifi_connected = true;

        ESP_LOGI(TAG, "Connected to hotspot!");
        ESP_LOGI(TAG, "IP address: " IPSTR " | Gateway: " IPSTR,
                 IP2STR(&event->ip_info.ip),
                 IP2STR(&event->ip_info.gw));
    }
}

// --------------------------------------------------
// WIFI INITIALIZATION
// --------------------------------------------------

static void wifi_init_sta(void)
{
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default());
    esp_netif_create_default_wifi_sta();

    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));

    ESP_ERROR_CHECK(
        esp_event_handler_register(
            WIFI_EVENT,
            ESP_EVENT_ANY_ID,
            &wifi_event_handler,
            NULL
        )
    );

    ESP_ERROR_CHECK(
        esp_event_handler_register(
            IP_EVENT,
            IP_EVENT_STA_GOT_IP,
            &wifi_event_handler,
            NULL
        )
    );

    wifi_config_t wifi_config = {};

    strncpy(
        (char *)wifi_config.sta.ssid,
        WIFI_SSID,
        sizeof(wifi_config.sta.ssid) - 1
    );

    strncpy(
        (char *)wifi_config.sta.password,
        WIFI_PASSWORD,
        sizeof(wifi_config.sta.password) - 1
    );

    wifi_config.sta.threshold.authmode = WIFI_AUTH_WPA2_PSK;

    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_set_config(WIFI_IF_STA, &wifi_config));
    ESP_ERROR_CHECK(esp_wifi_start());

    // Disable Wi-Fi Modem Sleep so packet reception is not throttled
    ESP_ERROR_CHECK(esp_wifi_set_ps(WIFI_PS_NONE));
}

// --------------------------------------------------
// CSI INITIALIZATION
// --------------------------------------------------

static void csi_init(void)
{
    wifi_csi_config_t csi_config = {};

    // Extract 52-subcarrier Legacy LTF with hardware channel filter enabled
    csi_config.lltf_en = true;
    csi_config.htltf_en = false;
    csi_config.stbc_htltf2_en = false;

    csi_config.ltf_merge_en = false;
    csi_config.channel_filter_en = true;

    csi_config.manu_scale = false;
    csi_config.shift = 0;

    csi_config.dump_ack_en = false;

    ESP_ERROR_CHECK(esp_wifi_set_csi_rx_cb(wifi_csi_callback, NULL));
    ESP_ERROR_CHECK(esp_wifi_set_csi_config(&csi_config));
    ESP_ERROR_CHECK(esp_wifi_set_csi(true));

    ESP_LOGI(TAG, "CSI reception enabled (LLTF=1, HTLTF=0, FILTER=1)!");
}

// --------------------------------------------------
// APPLICATION ENTRY
// --------------------------------------------------

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Starting ESP32-S3 Real-Time CSI Fall Detection System...");

    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES ||
        ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    ESP_ERROR_CHECK(ret);

    window_mutex = xSemaphoreCreateMutex();
    csi_queue = xQueueCreate(CSI_QUEUE_LENGTH, sizeof(csi_packet_t));

    if (csi_queue == NULL || window_mutex == NULL) {
        ESP_LOGE(TAG, "Failed to create RTOS primitives");
        return;
    }

    // Pin ESP-DL Inference Task to Core 1 so it never blocks Wi-Fi / CSI on Core 0
    BaseType_t infer_res = xTaskCreatePinnedToCore(
        model_inference_task,
        "model_infer",
        16384,
        NULL,
        5,
        &inference_task_handle,
        1
    );

    if (infer_res != pdPASS) {
        ESP_LOGE(TAG, "Inference task creation failed");
        return;
    }

    // Pin CSI Processing Task to Core 0 (higher priority than inference)
    BaseType_t csi_res = xTaskCreatePinnedToCore(
        csi_processing_task,
        "csi_processing",
        8192,
        NULL,
        6,
        NULL,
        0
    );

    if (csi_res != pdPASS) {
        ESP_LOGE(TAG, "CSI processing task creation failed");
        return;
    }

    wifi_init_sta();
    csi_init();

    // Start background UDP traffic generator on Core 0
    xTaskCreatePinnedToCore(
        csi_traffic_generator_task,
        "csi_traffic",
        4096,
        NULL,
        4,
        NULL,
        0
    );

    ESP_LOGI(TAG, "System running! Waiting for first 100-packet CSI window...");
}