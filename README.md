# EfficientNetV2-B2 Quantization for Mobile Deployment

## Problem definition
We fine-tuned EfficientNetV2-B2 for facial emotion classification (7 classes) on the CK+ dataset, then explored whether quantization could shrink the model and speed up inference for mobile deployment without meaningfully hurting accuracy. The core question: does quantization actually deliver on its promise on real hardware, or only on paper?

## Model and tech stack
Model: EfficientNetV2-B2 (ImageNet-pretrained, fine-tuned on CK+). Framework: TensorFlow/Keras with TensorFlow Lite for conversion, and `tensorflow_model_optimization` for quantization-aware training. Techniques applied: QAT, dynamic-range PTQ, and full-integer PTQ. On-device benchmarking used adb and Google's official TFLite `benchmark_model` APK on a physical Android phone.

## Results: Colab (desktop CPU) vs Android (on-device)
On Colab, FP32 was fastest per image (~23.3ms) while quantized models were slower (~52-97ms) — TFLite's int8 kernels aren't accelerated on desktop x86. On real Android hardware the ranking flipped: FP32 and dynamic-range PTQ were slowest (~149.8ms and ~145.3ms respectively), while QAT (~62.6ms) and full-integer PTQ (~41.3ms) were 2.4x-3.6x faster. This confirmed desktop-only profiling was misleading for mobile deployment decisions.

## How evaluation was done on Colab
Accuracy for each model variant was computed using the TFLite Python `Interpreter` (and `model.evaluate()` for the FP32 Keras model) run against a held-out `test_generator` of 152 CK+ images. Results: FP32 97.37%, QAT 96.71%, dynamic-range PTQ 97.37%, and full-integer PTQ 21.71% (a genuine accuracy collapse traced to Squeeze-and-Excitation layers being quantization-sensitive under full-integer conversion). Model file sizes were also compared: FP32 ~101.4MB vs QAT ~21.0MB (4.82x smaller).

## How evaluation was done on Android
Each `.tflite` model was pushed to `/data/local/tmp/` on the device via `adb push`. Google's prebuilt TFLite benchmark APK was installed and launched via `adb shell am start`, passing the model path, thread count, and run count as arguments. Per-image inference latency (`Inference (avg)` in microseconds) was read from `adb logcat`, with multiple launches used to get a median where run-to-run variance was observed. Accuracy was not measured on-device — only latency — since int8 arithmetic is deterministic and the Colab accuracy figures apply regardless of the device running inference.
