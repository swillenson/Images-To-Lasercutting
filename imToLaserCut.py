import cv2
import torch
import numpy as np
import torchvision.transforms as T
from torchvision.models.detection import maskrcnn_resnet50_fpn

def get_outline(image_path):
    # Load the Haar cascade xml file for face detection
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    # Read the image
    image = image_path
    # Convert the image to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # Draw a rectangle around each face
    # for (x, y, w, h) in faces:
    #     cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Apply Gaussian blur to the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Detect edges using the Canny algorithm
    edges = cv2.Canny(blurred, 30, 150)
    # Find contours in the edged image
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # Draw the contours on a blank image
    outline = np.zeros(image.shape, dtype="uint8")
    cv2.drawContours(outline, contours, -1, (255, 255, 255), 2)

    return outline

def remove_foreground_outline(image, outline):
    # Convert the outline to grayscale
    gray = cv2.cvtColor(outline, cv2.COLOR_BGR2GRAY)
    # Threshold the grayscale image to obtain a binary mask
    _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    # Invert the binary mask
    mask = cv2.bitwise_not(mask)
    # Apply the mask to the image to remove the foreground outline
    result = cv2.bitwise_and(image, image, mask=mask)
    return result

# Load the pre-trained Mask R-CNN model
model = maskrcnn_resnet50_fpn(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Load and preprocess the image
image_path = 'person5.jpg'
image = cv2.imread(image_path)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image_tensor = T.ToTensor()(image)
image_tensor = image_tensor.unsqueeze(0)

# Run the image through the model
with torch.no_grad():
    predictions = model(image_tensor)

# Get the masks for people with a higher confidence threshold
threshold = 0.8  # Adjust the threshold as needed
masks = predictions[0]['masks']
scores = predictions[0]['scores']
person_masks = masks.squeeze(1)[scores > threshold].detach().numpy()

# Find the largest person mask
largest_mask = None
largest_area = 0
for person_mask in person_masks:
    person_mask = (person_mask > 0.5).astype(np.uint8)
    contours, _ = cv2.findContours(person_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area = cv2.contourArea(contours[0])
    if area > largest_area:
        largest_area = area
        largest_mask = person_mask

# Invert the mask to get the background
background_mask = 1 - largest_mask

# Create the masked image using the largest mask
person_image = cv2.bitwise_and(image, image, mask=largest_mask)

# Create the background image using the inverted mask
background_image = cv2.bitwise_and(image, image, mask=background_mask)

# Get the outline for the images
outline_person = get_outline(person_image)
outline_background = get_outline(background_image)

# Remove the foreground outline from the background outline
outline_background = remove_foreground_outline(outline_background, outline_person)


def inpaint_black_space(image):
    # Create a binary mask of the black space (black pixels)
    mask = np.all(image == 0, axis=2).astype(np.uint8)

    # Perform inpainting on the black space using the PatchMatch algorithm
    filled_image = cv2.inpaint(image, mask, 3, cv2.INPAINT_TELEA)

    return filled_image

background_image = inpaint_black_space(background_image)

# Show the separated person, the background, and the outline
cv2.imshow('Separated Person', person_image)
cv2.imshow('Background', background_image)
cv2.imshow('Outline Foreground', outline_person)
cv2.imshow('Outline Background', outline_background)
cv2.waitKey(0)
cv2.destroyAllWindows()

