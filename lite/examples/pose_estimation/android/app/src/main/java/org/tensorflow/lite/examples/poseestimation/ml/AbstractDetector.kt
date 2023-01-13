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

import android.graphics.Bitmap
import android.graphics.Canvas
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.gpu.GpuDelegate

interface  AbstractDetector<DetectionResultT> : AutoCloseable {
    companion object {
        protected const val CPU_NUM_THREADS = 4
        fun getOption(device: Device): Pair<Interpreter.Options, GpuDelegate?>{
            val options = Interpreter.Options()
            var gpuDelegate: GpuDelegate? = null
            options.setNumThreads(CPU_NUM_THREADS)
            when (device) {
                Device.CPU -> {
                }
                Device.GPU -> {
                    gpuDelegate = GpuDelegate()
                    options.addDelegate(gpuDelegate)
                }
                Device.NNAPI -> options.setUseNNAPI(true)
            }
            return Pair(options, gpuDelegate)
        }
    }
    open fun inferenceImage(bitmap: Bitmap): DetectionResultT
    open fun visualize(overlay: Canvas, bitmap: Bitmap, result: DetectionResultT )
    open fun lastInferenceTimeNanos(): Long
}
