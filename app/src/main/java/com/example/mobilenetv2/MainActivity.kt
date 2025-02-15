package com.example.mobilenetv2
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
    private var labelsList: List<String> = emptyList()

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)



        imageView = findViewById(R.id.imageView)
        classifyButton = findViewById(R.id.classifyButton)
        resultText = findViewById(R.id.resultText)

        loadLabels() // Load labels.json

        try {
            val options = Interpreter.Options()
            options.setUseNNAPI(false)  // Disable NNAPI delegate
            options.setUseXNNPACK(false)  // Disable XNNPACK delegate
            options.setNumThreads(1)  // Match single-thread execution
            tflite = Interpreter(loadModelFile("mobilenet_v2.tflite"), options)
            Log.d("TFLite", "Model loaded successfully!")

            // Debugging: Log model's expected input shape
            val inputShape = tflite.getInputTensor(0).shape()
            Log.d("TFLite", "Model expects input shape: ${inputShape.contentToString()}")


            val outputShape = tflite.getOutputTensor(0).shape()
            val inputType = tflite.getInputTensor(0).dataType()

            Log.d("TFLite", "debug1001 - Model Output Shape: ${outputShape.contentToString()}")
            Log.d("TFLite", "debug1001 - Model Input Type: $inputType")

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


    // Load labels from labels.txt and ensure proper key conversion

    private fun loadLabels() {
        try {
            val inputStream = assets.open("imagenet_classes.txt")
            labelsList = inputStream.bufferedReader().readLines()

            Log.d("TFLite", "Labels loaded successfully with ${labelsList.size} classes.")
        } catch (e: Exception) {
            Log.e("TFLite", "Error loading labels: ${e.message}")
            labelsList = emptyList()
        }
    }

    private fun getLabelForIndex(index: Int): String {
        return if (index in labelsList.indices) {
            labelsList[index]
        } else {
            Log.e("TFLite", "Invalid Index: $index, Label Not Found!")
            "Unknown (Invalid Index: $index)"
        }
    }

    // Preprocess the image before sending it to the model
    private fun preprocessImageFast(bitmap: Bitmap): ByteBuffer {
        Log.d("TFLite", "debug1001 - Starting image preprocessing")

        // Resize image
        val resizedBitmap = Bitmap.createScaledBitmap(bitmap, 224, 224, true)
        Log.d("TFLite", "debug1001 - Image resized to 224x224")

        // Allocate buffer (Float32, 3x224x224)
        val byteBuffer = ByteBuffer.allocateDirect(4 * 1 * 3 * 224 * 224)
        byteBuffer.order(ByteOrder.nativeOrder())

        val mean = floatArrayOf(0.485f, 0.456f, 0.406f)
        val std = floatArrayOf(0.229f, 0.224f, 0.225f)

        val inputFloats = FloatArray(3 * 224 * 224)
        var index = 0

        // Iterate pixels and normalize
        for (y in 0 until 224) {
            for (x in 0 until 224) {
                val pixel = resizedBitmap.getPixel(x, y)

                // Extract color channels
                val r = (((pixel shr 16) and 0xFF).toFloat() / 255.0f - mean[0]) / std[0]
                val g = (((pixel shr 8) and 0xFF).toFloat() / 255.0f - mean[1]) / std[1]
                val b = ((pixel and 0xFF).toFloat() / 255.0f - mean[2]) / std[2]

                // **Reorder Channels to (C, H, W)**
                inputFloats[index] = r
                inputFloats[index + (224 * 224)] = g
                inputFloats[index + (2 * 224 * 224)] = b

                if (index < 10) { // Debug: Print first 10 values
                    Log.d("TFLite", "debug1001 - Preprocessed Pixel [$x, $y]: R=$r, G=$g, B=$b")
                }

                index++
            }
        }

        // Copy array into ByteBuffer
        for (value in inputFloats) {
            byteBuffer.putFloat(value)
        }

        Log.d("TFLite", "debug1001 - First 10 Preprocessed Input Values: " +
                java.util.Arrays.toString(inputFloats.copyOfRange(0, 10)))

        byteBuffer.rewind()  // Reset position to read from the beginning
        val extractedFloats = FloatArray(10) // Read first 10 values
        for (i in 0 until 10) {
            extractedFloats[i] = byteBuffer.float  // Read float from ByteBuffer
        }

        Log.d("TFLite", "debug1001 - First 10 ByteBuffer Values: " +
                java.util.Arrays.toString(extractedFloats))


        return byteBuffer
    }

    // Run inference asynchronously and return only the best prediction
    private fun runInferenceAsync(inputData: ByteBuffer, callback: (String) -> Unit) {
        CoroutineScope(Dispatchers.IO).launch {
            val outputData = Array(1) { FloatArray(1000) } // Assuming 1000 classes
            try {
                Log.d("TFLite", "debug1001 - Running inference in background...")
                tflite.run(inputData, outputData)

                // Print first 10 output values
                val predictions = outputData[0]
                Log.d("TFLite", "debug1001 - First 10 Raw Output Values: " +
                        predictions.copyOfRange(0, 1000).joinToString())

                // Find the highest confidence prediction
                val maxIndex = predictions.indices.maxByOrNull { predictions[it] } ?: -1
                val predictedClass = getLabelForIndex(maxIndex)
                val confidence = predictions[maxIndex] * 100 // Convert to percentage
                Log.d("TFLite", "debug1001 - Max Value: ${predictions[maxIndex]} at Index: $maxIndex")

                withContext(Dispatchers.Main) {
                    callback(predictedClass) // ✅ Return only the label
                }
            } catch (e: Exception) {
                Log.e("TFLite", "debug1001 - Error during inference: ${e.message}")
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
        val predictedLabel = if (maxIndex in labelsList.indices) labelsList[maxIndex] else "Unknown"

        // Only set the predicted label, no confidence
        resultText.text = predictedLabel
    }
}