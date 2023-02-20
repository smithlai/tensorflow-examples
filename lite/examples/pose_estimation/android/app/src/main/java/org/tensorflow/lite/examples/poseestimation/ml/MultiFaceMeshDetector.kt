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
import org.tensorflow.lite.examples.poseestimation.data.Device
import org.tensorflow.lite.examples.poseestimation.data.FaceCrop
import org.tensorflow.lite.examples.poseestimation.data.FaceMesh
import java.lang.Float.max
import java.lang.Float.min

class MultiFaceMeshDetector(val faceCropDetector: FaceCropDetector, val faceMeshDetector: FaceMeshDetector): AbstractDetector<List<FaceMesh>>() {

    companion object {
        private const val FACECROP_MARGIN = 0.25f
        private const val TAG = "MultiFaceMesh"

        fun create(context: Context, device: Device): MultiFaceMeshDetector {
//            val settings: Pair<Interpreter.Options, GpuDelegate?> = AbstractDetector.getOption(device)
//            val options = settings.first
//            var gpuDelegate = settings.second
            val faceCropDetector = FaceCropDetector.create(context, device)
            val faceMeshDetector = FaceMeshDetector.create(context, device, false)
            return MultiFaceMeshDetector(
                faceCropDetector, faceMeshDetector
            )
        }
    }
    private val visualizationUtils = FaceMeshDetector.VisualizationUtils()
    override var inference_results: List<FaceMesh> = listOf()
    private var lastInferenceTimeNanos: Long = -1
    @Suppress("UNCHECKED_CAST")
    override fun inferenceImage(bitmap: Bitmap): List<FaceMesh> {
        val inferenceStartTimeNanos = SystemClock.elapsedRealtimeNanos()
        var facemeshes = mutableListOf<FaceMesh>()
        faceCropDetector.requestInferenceImage(bitmap).let{ croplist ->
            for (crop in croplist){
                val tl = crop.data.get(FaceCrop.FaceCropPart.TL.position)
                val br = crop.data.get(FaceCrop.FaceCropPart.BR.position)
                var x1 = tl.first
                var y1 = tl.second
                var x2 = br.first
                var y2 = br.second
                val center_x = (x1+x2)/2
                val center_y = (y1+y2)/2
                var w = x2 - x1
                var h = y2 - y1
                // re-calc margin
                val newW2 = w*(1 + FACECROP_MARGIN)/2
                val newH2 = h*(1 + FACECROP_MARGIN)/2
                x1 = max(0f, center_x - newW2)
                y1 = max(0f, center_y - newH2)
                x2 = min(bitmap.width.toFloat(), center_x + newW2)
                y2 = min(bitmap.height.toFloat(), center_y +newH2)
                w = x2 - x1
                h = y2 - y1
                val face = Bitmap.createBitmap(bitmap, x1.toInt(),y1.toInt(), w.toInt(), h.toInt())
                val tmpmesh = faceMeshDetector.requestInferenceImage(face)
                val new_crop_points = crop.data.toMutableList()
                new_crop_points[FaceCrop.FaceCropPart.TL.position] = Pair(x1,y1)
                new_crop_points[FaceCrop.FaceCropPart.BR.position] = Pair(x2,y2)
                tmpmesh.forEach {
                    it.facecrop = FaceCrop(new_crop_points,crop.score)
                }
                facemeshes.addAll(tmpmesh)
            }
        }
        lastInferenceTimeNanos = SystemClock.elapsedRealtimeNanos() - inferenceStartTimeNanos
        printInferenceTime(TAG)
//        Log.e(
//            TAG, "facemeshes.size: " + facemeshes.size.toString()
//        )
        return facemeshes
    }

    override fun lastInferenceTimeNanos(): Long = lastInferenceTimeNanos
    override fun close() {
        faceCropDetector.close()
        faceMeshDetector?.close()
    }

    override fun drawResultOnBitmap(bitmap: Bitmap): Bitmap {
        var outputBitmap = visualizationUtils.drawKeypoints(bitmap, getResults())
        return outputBitmap
    }

}
