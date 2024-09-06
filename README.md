# SEMANTIC-SEGMENTATION-USING-MASK-R-CNN
Imports:
torch: Imports PyTorch, a popular deep learning library.
torchvision: Imports TorchVision, a package that includes image models and utilities.
cv2: Imports OpenCV, a library for image processing.
numpy: Imports NumPy for numerical operations, such as array manipulation.
matplotlib.pyplot: Imports Matplotlib for plotting images.
PIL.Image: Imports the Python Imaging Library for handling images.
Load Mask R-CNN Model:
Load pre-trained Mask R-CNN: Loads a pre-trained Mask R-CNN model (maskrcnn_resnet50_fpn) with a ResNet-50 backbone. This model is pre-trained to perform object detection and instance segmentation.
Set evaluation mode: Sets the model in evaluation mode using model.eval(), which disables training-specific behaviors like dropout.
Preprocessing Function (preprocess):
Load the image: Loads an image using cv2.imread().
Convert BGR to RGB: OpenCV loads images in BGR format by default, so this converts the image to RGB using cv2.cvtColor().
Resize the image: Resizes the image to (800, 800) using cv2.resize(), which is a standard size for Mask R-CNN models.
Create a preprocessing transform: Uses torchvision.transforms.Compose() to define a transformation pipeline that converts the resized image to a PyTorch tensor.
Convert the image to a tensor: Applies the transformation using preprocess_transform() and adds a batch dimension using .unsqueeze(0).
Return the tensor and resized image: The function returns the preprocessed tensor and the resized image.
Segmentation Function (segment_image):
Disable gradient computation: Uses torch.no_grad() to disable gradient calculation, which speeds up inference.
Run the model: Passes the input tensor through the model to get the output (segmentation masks, bounding boxes, etc.).
Extract masks: Retrieves the segmentation masks from the output and converts them from PyTorch tensors to NumPy arrays using .cpu().numpy().
Extract confidence scores: Retrieves confidence scores for the detected objects and converts them to NumPy arrays.
Filter masks by confidence: Filters out masks with low confidence (scores below 0.5).
Return high-confidence masks: The function returns only the high-confidence segmentation masks.
Mask Overlay Function (overlay_segmentation_mask):
Create a blank mask: Initializes an empty 3-channel image (combined_mask) to store the combined masks for all objects.
Iterate through masks: For each mask in masks, processes it individually.
Convert to binary mask: Converts each mask to a binary mask where pixel values greater than 0.5 become 255 (white) and the rest are 0.
Create a colored mask: Converts the binary mask into a 3-channel (RGB) mask by stacking the binary mask three times.
Update combined mask: Updates the combined_mask by taking the maximum value between the current mask and the previous masks.
Blend the original image with the mask: Combines the original image with the segmentation mask using cv2.addWeighted() to create a blended result.
Return blended image: The function returns the blended image.
Image Processing Pipeline:
Load the image path: Defines the path to the image.
Preprocess the image: Calls the preprocess() function to get the input tensor and resized image.
Segment the image: Calls segment_image() to apply the Mask R-CNN model and get high-confidence segmentation masks.
Overlay segmentation: Calls overlay_segmentation_mask() to overlay the masks on the original image.
Display the Results:
Initialize a plot: Creates a figure for displaying the original and segmented images side by side.
Plot original image: Plots the original image in the first subplot.
Plot segmented image: Plots the result (segmented image with masks) in the second subplot.
Display the plots: Shows the plots using plt.show().
