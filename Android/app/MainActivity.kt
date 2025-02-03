package com.example.mobilenet_v2

import android.app.Activity
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import org.tensorflow.lite.Interpreter
import org.json.JSONObject
import kotlinx.coroutines.*
import java.io.FileInputStream
import java.io.InputStream
import java.nio.ByteBuffer
import java.nio.ByteOrder
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

class MainActivity : AppCompatActivity() {

    private lateinit var tflite: Interpreter
    private lateinit var imageView: ImageView
    private lateinit var classifyButton: Button
    private lateinit var resultText: TextView
    private var selectedBitmap: Bitmap? = null
    private lateinit var labelsMap: Map<Int, String>

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)

        imageView = findViewById(R.id.imageView)
        classifyButton = findViewById(R.id.classifyButton)
        resultText = findViewById(R.id.resultText)

        loadLabels() // Load labels.json

        try {
            tflite = Interpreter(loadModelFile("mobilenet_v2.tflite"))
            Log.d("TFLite", "Model loaded successfully!")

            // Debugging: Log model's expected input shape
            val inputShape = tflite.getInputTensor(0).shape()
            Log.d("TFLite", "Model expects input shape: ${inputShape.contentToString()}")

        } catch (e: Exception) {
            Log.e("TFLite", "Error loading model", e)
        }

        // Open gallery when clicking the image
        imageView.setOnClickListener {
            val intent = Intent(Intent.ACTION_PICK, MediaStore.Images.Media.EXTERNAL_CONTENT_URI)
            imagePicker.launch(intent)
        }

        // Run classification when clicking the button
        classifyButton.setOnClickListener {
            selectedBitmap?.let {
                val inputData = preprocessImageFast(it) // Convert bitmap to ByteBuffer
                runInferenceAsync(inputData) { predictedResult ->
                    resultText.text = predictedResult // Display only one prediction
                }
            }
        }
    }

    // Image picker result handler
    private val imagePicker =
        registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
            if (result.resultCode == Activity.RESULT_OK) {
                result.data?.data?.let { uri ->
                    val inputStream: InputStream? = contentResolver.openInputStream(uri)
                    selectedBitmap = BitmapFactory.decodeStream(inputStream)
                    imageView.setImageBitmap(selectedBitmap) // Show the image
                }
            }
        }

    // Load the TensorFlow Lite model
    private fun loadModelFile(modelName: String): MappedByteBuffer {
        val fileDescriptor = assets.openFd(modelName)
        val inputStream = FileInputStream(fileDescriptor.fileDescriptor)
        val fileChannel = inputStream.channel
        return fileChannel.map(
            FileChannel.MapMode.READ_ONLY,
            fileDescriptor.startOffset,
            fileDescriptor.declaredLength
        )
    }


    // Load labels from labels.json and ensure proper key conversion
    private fun loadLabels() {
        try {
            val inputStream = assets.open("imagenet_classes.txt") // Use the correct file
            val labelsList = inputStream.bufferedReader().readLines() // Read all lines
            labelsMap =
                labelsList.mapIndexed { index, label -> index to label }.toMap() // Assign index
            Log.d("TFLite", "Labels loaded successfully with ${labelsMap.size} classes.")
        } catch (e: Exception) {
            Log.e("TFLite", "Error loading labels: ${e.message}")
            labelsMap = emptyMap()
        }
    }


    private fun getLabelForIndex(index: Int): String {
        return labelsMap[index] ?: "Unknown"
    }

    // Preprocess the image before sending it to the model
    private fun preprocessImageFast(bitmap: Bitmap): ByteBuffer {
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        val byteBuffer = ByteBuffer.allocateDirect(4 * 1 * 3 * 224 * 224)
        byteBuffer.order(ByteOrder.nativeOrder())

        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = resizedBitmap.getPixel(x, y)
                val r = ((pixel shr 16 and 0xFF) / 255.0f - 0.485f) / 0.229f
                val g = ((pixel shr 8 and 0xFF) / 255.0f - 0.456f) / 0.224f
                val b = ((pixel and 0xFF) / 255.0f - 0.406f) / 0.225f

                byteBuffer.putFloat(r)
                byteBuffer.putFloat(g)
                byteBuffer.putFloat(b)
            }
        }
        return byteBuffer
    }

    // Run inference asynchronously and return only the best prediction
    private fun runInferenceAsync(inputData: ByteBuffer, callback: (String) -> Unit) {
        CoroutineScope(Dispatchers.IO).launch {
            val outputData = Array(1) { FloatArray(1000) } // Modify based on the number of classes
            try {
                Log.d("TFLite", "Running inference in background...")
                tflite.run(inputData, outputData)

                // Find the highest confidence prediction
                val predictions = outputData[0]
                val maxIndex = predictions.indices.maxByOrNull { predictions[it] } ?: -1
                val predictedClass = getLabelForIndex(maxIndex)
                val confidence = predictions[maxIndex] * 100 // Convert to percentage

                withContext(Dispatchers.Main) {
                    callback(predictedClass) // ✅ Only return the label
                }
            } catch (e: Exception) {
                Log.e("TFLite", "Error during inference: ${e.message}")
            }
        }
    }

    // Display the single best classification result
    private fun displayResult(output: FloatArray) {
        // Apply Softmax to convert logits to probabilities
        val expScores = output.map { Math.exp(it.toDouble()) } // Exponential of each score
        val sumExpScores = expScores.sum() // Sum of all exponentials
        val probabilities = expScores.map { it / sumExpScores } // Normalize

        // Find the highest probability index
        val maxIndex = probabilities.indices.maxByOrNull { probabilities[it] } ?: -1
        val predictedLabel = labelsMap[maxIndex] ?: "Unknown"

        // Only set the predicted label, no confidence
        resultText.text = predictedLabel
    }
}
