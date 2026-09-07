# EfficientNetV2-B2 Quantization for Mobile Deployment

## Problem definition
We fine-tuned EfficientNetV2-B2 for facial emotion classification (7 classes) on the CK+ dataset, then explored whether quantization could shrink the model and speed up inference for mobile deployment without meaningfully hurting accuracy. The core question: does quantization actually deliver on its promise on real hardware, or only on paper?

## Model and tech stack
Model: EfficientNetV2-B2 (ImageNet-pretrained, fine-tuned on CK+). Framework: TensorFlow/Keras with TensorFlow Lite for conversion, and `tensorflow_model_optimization` for quantization-aware training. Techniques applied: QAT, dynamic-range PTQ, and full-integer PTQ. On-device benchmarking used adb and Google's official TFLite `benchmark_model` APK on a physical Android phone.

## Results: Colab (desktop CPU) vs Android (on-device)

| Model | Colab latency (per image) | Android latency (per image) |
|---|---|---|
| FP32 | 23.3 ms | 149.8 ms |
| QAT | 52-97 ms | 62.6 ms |
| Dynamic-range PTQ | 56-63 ms | 145.3 ms |
| Full-integer PTQ | 67.4 ms | 41.3 ms |

On desktop, FP32 was fastest and quantized models were slower, since TFLite's int8 kernels aren't hardware-accelerated on x86. On the actual phone the ranking flipped: FP32 and dynamic-range PTQ were slowest, while QAT and full-integer PTQ were 2.4x-3.6x faster. This confirmed desktop-only profiling was misleading for mobile deployment decisions.

## How evaluation was done on Colab

| Model | Accuracy | Model size |
|---|---|---|
| FP32 | 97.37% | 101.4 MB |
| QAT | 96.71% | 21.0 MB (4.82x smaller) |
| Dynamic-range PTQ | 97.37% | 33.1 MB |
| Full-integer PTQ | 21.71% (collapsed) | -- |

Accuracy was computed using the TFLite Python `Interpreter` (and `model.evaluate()` for the FP32 Keras model) against a held-out `test_generator` of 152 CK+ images. Full-integer PTQ's collapse was traced to Squeeze-and-Excitation layers being quantization-sensitive under full-integer conversion.

## How evaluation was done on Android
Each `.tflite` model was pushed to `/data/local/tmp/` on the device via `adb push`. Google's prebuilt TFLite benchmark APK was installed and launched via `adb shell am start`, passing the model path, thread count, and run count as arguments. Per-image inference latency (`Inference (avg)` in microseconds) was read from `adb logcat`, with multiple launches used to get a median where run-to-run variance was observed. Accuracy was not measured on-device -- only latency -- since int8 arithmetic is deterministic and the Colab accuracy figures apply regardless of the device running inference.