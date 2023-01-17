/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================
*/

package org.tensorflow.lite.examples.poseestimation.ml

import android.content.Context
import android.graphics.*
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.poseestimation.data.DetectedObject
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.data.FaceMesh
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.image.ops.ResizeOp
import org.tensorflow.lite.support.image.ops.ResizeWithCropOrPadOp
import kotlin.math.exp

class MobilenetDetector(
    private val interpreter: Interpreter,
    private var gpuDelegate: GpuDelegate?): ObjectDetector(interpreter, gpuDelegate) {

    companion object {
        private const val MEAN = 127.5f
        private const val STD = 127.5f
        private const val THRESHOLD = 0.7

        private const val TAG = "OD"

//        private const val MODEL_FILENAME = "efficientdet-lite0.tflite"
        private const val MODEL_FILENAME = "mobilenetv1.tflite"

        fun create(context: Context, device: Device): MobilenetDetector {
            val settings: Pair<Interpreter.Options, GpuDelegate?> = AbstractDetector.getOption(device)
            val options = settings.first
            var gpuDelegate = settings.second

            return MobilenetDetector(
                Interpreter(
                    FileUtil.loadMappedFile(
                        context,
                        MODEL_FILENAME
                    ), options
                ),
                gpuDelegate
            )
        }
    }
}
