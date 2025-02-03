import torch
import torchvision.models as models
import coremltools as ct


def convert_model():
    try:
        # Load the pre-trained MobileNetV2 model
        model = models.mobilenet_v2(weights="IMAGENET1K_V1")
        model.eval()

        # Move model to CPU for conversion
        model = model.cpu()

        # Create example input
        example_input = torch.rand(1, 3, 224, 224)

        # Trace the model
        traced_model = torch.jit.trace(model, example_input)

        # Convert to Core ML
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.ImageType(shape=(1, 3, 224, 224))],
            minimum_deployment_target=ct.target.iOS14  # Specify deployment target
        )

        # Save the model
        mlmodel.save("MobileNetV2.mlmodel")
        print("Model converted and saved as MobileNetV2.mlmodel")

    except Exception as e:
        print(f"Error during conversion: {str(e)}")


if __name__ == "__main__":
    convert_model()