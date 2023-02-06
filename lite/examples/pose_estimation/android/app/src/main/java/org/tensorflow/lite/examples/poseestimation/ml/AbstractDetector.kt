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
import android.graphics.Rect
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.gpu.GpuDelegate

abstract class AbstractDetector<DetectionResultT> : AutoCloseable {
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
    abstract protected var inference_results: DetectionResultT
    fun getResults(): DetectionResultT{
        return inference_results
    }
    open fun requestInferenceImage(bitmap: Bitmap): DetectionResultT{
        inference_results = inferenceImage(bitmap)
        return inference_results
    }
    abstract protected fun inferenceImage(bitmap: Bitmap): DetectionResultT

    fun visualize(overlay: Canvas, bitmap: Bitmap){
        val output = bitmap.copy(Bitmap.Config.ARGB_8888, true)
        val bitmap = drawResultOnBitmap(output)
        bitmapToOverlay(overlay, bitmap )
    }
    abstract fun lastInferenceTimeNanos(): Long
    abstract fun drawResultOnBitmap(bitmap: Bitmap):Bitmap

    fun bitmapToOverlay(overlay: Canvas, outputBitmap: Bitmap){
        overlay?.let { canvas ->
            val screenWidth: Int
            val screenHeight: Int
            val left: Int
            val top: Int

            if (canvas.height > canvas.width) {
                val ratio = outputBitmap.height.toFloat() / outputBitmap.width
                screenWidth = canvas.width
                left = 0
                screenHeight = (canvas.width * ratio).toInt()
                top = (canvas.height - screenHeight) / 2
            } else {
                val ratio = outputBitmap.width.toFloat() / outputBitmap.height
                screenHeight = canvas.height
                top = 0
                screenWidth = (canvas.height * ratio).toInt()
                left = (canvas.width - screenWidth) / 2
            }
            val right: Int = left + screenWidth
            val bottom: Int = top + screenHeight

            canvas.drawBitmap(
                outputBitmap, Rect(0, 0, outputBitmap.width, outputBitmap.height),
                Rect(left, top, right, bottom), null
            )
        }
    }
}
