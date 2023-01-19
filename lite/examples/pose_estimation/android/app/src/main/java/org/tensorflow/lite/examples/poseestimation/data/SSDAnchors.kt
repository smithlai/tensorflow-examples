package org.tensorflow.lite.examples.poseestimation.data

import android.util.Log
import java.util.*

/*
 * https://github.com/patlevin/face-detection-tflite
 */
class SSDAnchors {
    companion object{
        // (reference: modules/face_detection/face_detection_back_desktop_live.pbtxt)
        val SSD_OPTIONS_BACK = mapOf(
            "num_layers" to 4,
            "input_size_height" to 256,
            "input_size_width" to 256,
            "anchor_offset_x" to 0.5f,
            "anchor_offset_y" to 0.5f,
            "strides" to arrayOf(16, 32, 32, 32),
            "interpolated_scale_aspect_ratio" to 1.0f
        )
        // (reference: modules/face_detection/face_detection_full_range_common.pbtxt)
        val SSD_OPTIONS_FULL = mapOf(
            "num_layers" to 1,
            "input_size_height" to 192,
            "input_size_width" to 192,
            "anchor_offset_x" to 0.5f,
            "anchor_offset_y" to 0.5f,
            "strides" to arrayOf(4),
            "interpolated_scale_aspect_ratio" to 0.0f
        )


        fun ssd_generate_anchors(opts: Map<String,*>) : List<Pair<Float,Float>>{
            """This is a trimmed down version of the C++ code; all irrelevant parts
            have been removed.
            (reference: mediapipe/calculators/tflite/ssd_anchors_calculator.cc)
            """

            val num_layers = opts["num_layers"] as Int
            val strides = opts["strides"] as Array<Int>
            assert(strides.size == num_layers)
            val input_height = opts["input_size_height"] as Int
            val input_width = opts["input_size_width"] as Int
            val anchor_offset_x = opts["anchor_offset_x"] as Float
            val anchor_offset_y = opts["anchor_offset_y"] as Float
            val interpolated_scale_aspect_ratio = opts["interpolated_scale_aspect_ratio"] as Float
            val anchors = mutableListOf<Pair<Float,Float>>()
            var layer_id = 0
            while (layer_id < num_layers){
                var last_same_stride_layer = layer_id
                var repeats = 0
                while (last_same_stride_layer < num_layers &&
                    strides[last_same_stride_layer] == strides[layer_id]){
                    last_same_stride_layer += 1
                    // aspect_ratios are added twice per iteration
                    if (interpolated_scale_aspect_ratio == 1.0f){
                        repeats += 2
                    }else{
                        repeats += 1
                    }
                }
                val stride = strides[layer_id]
                val feature_map_height = input_height / stride
                val feature_map_width = input_width / stride
                for (y in 0 until feature_map_height) {
                    val y_center = (y + anchor_offset_y) / feature_map_height.toFloat()
                    for (x in 0 until feature_map_width) {
                        val x_center = (x + anchor_offset_x) / feature_map_width.toFloat()
                        for (a in 0 until repeats) {
                            anchors.add(Pair(x_center, y_center))
                        }
                    }
                }
                layer_id = last_same_stride_layer
            }
            return anchors
        }

        fun decode_boxes(input_height:Int, raw_boxes: Array<Array<FloatArray>>, shape:IntArray,
                         anchors:List<Pair<Float,Float>>): Array<Array<Pair<Float, Float>>> {
            """Simplified version of
        mediapipe/calculators/tflite/tflite_tensors_to_detections_calculator.cc
        """
            // width == height so scale is the same across the board
            val scale = input_height.toFloat()
            //shape, ex: [1 * 896 * 16]
            //num_anchors = 896
            //num_points = 8
            val num_anchors = shape[shape.size-2]
            val num_points = shape[shape.size-1] / 2

            // reshape to [ 896, 8, 2 ]
            var new_boxes = Array(num_anchors){
                Array(num_points){
                    Pair(0.0f,0.0f)
                }
            }


            for (i in 0 until num_anchors){
                for (j in 0 until num_points){
                    val float_data = raw_boxes[0].get(i) as FloatArray
                    // scale all values (applies to positions, width, and height alike)
                    new_boxes[i][j] = Pair(float_data[2*j]/scale, float_data[2*j+1]/scale)
                }
            }

            for (i in 0 until num_anchors){
                // adjust center coordinates and key points to anchor positions
                for(j in 0 until num_points){
                    if (j == 1){
                        val center_x = new_boxes[i][0].first
                        val center_y = new_boxes[i][0].second
//                        Log.e("center", center_x.toString() +","+ center_y)
                        val halfsize_x = new_boxes[i][1].first / 2.0f
                        val halfsize_y = new_boxes[i][1].second / 2.0f
//                        Log.e("half", halfsize_x.toString() +","+ halfsize_y)
                        new_boxes[i][0] = Pair(center_x - halfsize_x, center_y - halfsize_y)
                        new_boxes[i][1] = Pair(center_x + halfsize_x, center_y + halfsize_y)
                    }else {
                        new_boxes[i][j] = Pair(
                            new_boxes[i][j].first + anchors[i].first,
                            new_boxes[i][j].second + anchors[i].second
                        )
                    }
                }
            }

            return new_boxes
        }

    }
}