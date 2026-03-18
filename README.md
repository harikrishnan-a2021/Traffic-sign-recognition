# Traffic Sign Recognition

A simple PyTorch-based traffic sign inference script.

The project defines a CNN (`TrafficSignNet`), loads trained weights from `gtsrb_cnn.pth`, reads a test image (`test_sign.jpg`), predicts the traffic sign class, and displays the result with `matplotlib`.

## What the code does

- Builds a CNN model for 43 traffic sign classes.
- Loads saved model weights from disk.
- Preprocesses an input image to `32x32` and normalizes it.
- Runs inference in evaluation mode (`torch.no_grad()`).
- Shows the image with the predicted class name and prints the label in terminal.

## Requirements

Install dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

Required libraries:

- `torch`
- `torchvision`
- `matplotlib`
- `Pillow`

## Project files expected by the script

Place these files in the project root (same folder as `code.py`):

- `code.py`
- `gtsrb_cnn.pth` (trained model weights)
- `test_sign.jpg` (input image for prediction)

## Run

```bash
python code.py
```

If everything is set correctly, the script opens an image window with the predicted class and also prints the recognized sign in the console.

## Notes

- The class list contains named labels for the first few classes and placeholder names (`Class 19` to `Class 42`) for the remaining classes.
- Update `image_path` in `code.py` if you want to test with a different image.
- Update the model path in `torch.load(...)` if your weight file has a different name/location.
