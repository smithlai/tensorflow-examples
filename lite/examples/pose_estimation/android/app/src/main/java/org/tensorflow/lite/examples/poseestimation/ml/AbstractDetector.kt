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
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Rect
import android.os.SystemClock
import android.util.Log
import org.tensorflow.lite.HexagonDelegate
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate

abstract class AbstractDetector<DetectionResultT> : AutoCloseable {
    companion object {
        const val printInferenceIntervalNanos = 1_000_000_000
        const val CPU_NUM_THREADS = 1

        fun getOption(device: Device, context: Context): Interpreter.Options{
            val options = Interpreter.Options()


            when (device) {
                Device.CPU -> {
                    options.setNumThreads(CPU_NUM_THREADS)
                }
                Device.GPU -> {
                    // Remember to copy the shared library to your app,
                    // and set jniLib options in gradle.build
                    // https://www.tensorflow.org/lite/android/delegates/hexagon
                    var gpuDelegate: GpuDelegate? = null
                    val compatList = CompatibilityList()
                    if(compatList.isDelegateSupportedOnThisDevice) {
                        // if the device has a supported GPU, add the GPU delegate
                        val delegateOptions = compatList.bestOptionsForThisDevice
                        gpuDelegate = GpuDelegate(delegateOptions)
                    }else{
                        gpuDelegate = GpuDelegate()
                        throw Exception("GPU not supported on this devices")
                    }
                    options.addDelegate(gpuDelegate)
                }
                Device.NNAPI -> options.setUseNNAPI(true)
                Device.HEXGON -> {
                    val hexagonDelegate = HexagonDelegate(context)
                    options.addDelegate(hexagonDelegate);
                }
            }
            return options
        }

    }
    private var lastPrintInferenceTimeNanos:Long = 0

    fun printInferenceTime(tag: String){
        if (printInferenceIntervalNanos > 0){
            val elapse = SystemClock.elapsedRealtimeNanos() - lastPrintInferenceTimeNanos
            if (elapse > printInferenceIntervalNanos) {
                Log.i(tag,String.format("Interpreter took %.2f ms",1.0f * lastInferenceTimeNanos() / 1_000_000))
                lastPrintInferenceTimeNanos = SystemClock.elapsedRealtimeNanos()
            }
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
