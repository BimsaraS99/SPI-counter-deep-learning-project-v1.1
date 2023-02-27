import numpy as np
import os
from tflite_support import metadata
import tensorflow as tf
assert tf.__version__.startswith('2')
tf.get_logger().setLevel('ERROR')

import platform # to get system information
from typing import List, NamedTuple
import json
import cv2
from PIL import Image
import statistics
import math
import time
start_time = time.time()

Interpreter = tf.lite.Interpreter
load_delegate = tf.lite.experimental.load_delegate


class ObjectDetectorOptions(NamedTuple):
  """A config to initialize an object detector."""

  enable_edgetpu: bool = False
  """Enable the model to run on EdgeTPU."""

  label_allow_list: List[str] = None
  """The optional allow list of labels."""

  label_deny_list: List[str] = None
  """The optional deny list of labels."""

  max_results: int = -1
  """The maximum number of top-scored detection results to return."""

  num_threads: int = 1
  """The number of CPU threads to be used."""

  score_threshold: float = 0.0
  """The score threshold of detection results to return."""


class Rect(NamedTuple):
  """A rectangle in 2D space."""
  left: float
  top: float
  right: float
  bottom: float


class Category(NamedTuple):
  """A result of a classification task."""
  label: str
  score: float
  index: int


class Detection(NamedTuple):
  """A detected object as the result of an ObjectDetector."""
  bounding_box: Rect
  categories: List[Category]


def edgetpu_lib_name():
  """Returns the library name of EdgeTPU in the current platform."""
  return {
      'Darwin': 'libedgetpu.1.dylib',
      'Linux': 'libedgetpu.so.1',
      'Windows': 'edgetpu.dll',
  }.get(platform.system(), None)


class ObjectDetector:
  """A wrapper class for a TFLite object detection model."""

  _OUTPUT_LOCATION_NAME = 'location'
  _OUTPUT_CATEGORY_NAME = 'category'
  _OUTPUT_SCORE_NAME = 'score'
  _OUTPUT_NUMBER_NAME = 'number of detections'

  def __init__(
      self,
      model_path: str,
      options: ObjectDetectorOptions = ObjectDetectorOptions()
  ) -> None:
    """Initialize a TFLite object detection model.
    Args:
        model_path: Path to the TFLite model.
        options: The config to initialize an object detector. (Optional)
    Raises:
        ValueError: If the TFLite model is invalid.
        OSError: If the current OS isn't supported by EdgeTPU.
    """

    # Load metadata from model.
    displayer = metadata.MetadataDisplayer.with_model_file(model_path)

    # Save model metadata for preprocessing later.
    model_metadata = json.loads(displayer.get_metadata_json())
    process_units = model_metadata['subgraph_metadata'][0]['input_tensor_metadata'][0]['process_units']
    mean = 0.0
    std = 1.0
    for option in process_units:
      if option['options_type'] == 'NormalizationOptions':
        mean = option['options']['mean'][0]
        std = option['options']['std'][0]
    self._mean = mean
    self._std = std

    # Load label list from metadata.
    file_name = displayer.get_packed_associated_file_list()[0]
    label_map_file = displayer.get_associated_file_buffer(file_name).decode()
    label_list = list(filter(lambda x: len(x) > 0, label_map_file.splitlines()))
    self._label_list = label_list

    # Initialize TFLite model.
    if options.enable_edgetpu:
      if edgetpu_lib_name() is None:
        raise OSError("The current OS isn't supported by Coral EdgeTPU.")
      interpreter = Interpreter(
          model_path=model_path,
          experimental_delegates=[load_delegate(edgetpu_lib_name())],
          num_threads=options.num_threads)
    else:
      interpreter = Interpreter(
          model_path=model_path, num_threads=options.num_threads)

    interpreter.allocate_tensors()
    input_detail = interpreter.get_input_details()[0]

    # From TensorFlow 2.6, the order of the outputs become undefined.
    # Therefore we need to sort the tensor indices of TFLite outputs and to know
    # exactly the meaning of each output tensor. For example, if
    # output indices are [601, 599, 598, 600], tensor names and indices aligned
    # are:
    #   - location: 598
    #   - category: 599
    #   - score: 600
    #   - detection_count: 601
    # because of the op's ports of TFLITE_DETECTION_POST_PROCESS
    # (https://github.com/tensorflow/tensorflow/blob/a4fe268ea084e7d323133ed7b986e0ae259a2bc7/tensorflow/lite/kernels/detection_postprocess.cc#L47-L50).
    sorted_output_indices = sorted(
        [output['index'] for output in interpreter.get_output_details()])
    self._output_indices = {
        self._OUTPUT_LOCATION_NAME: sorted_output_indices[0],
        self._OUTPUT_CATEGORY_NAME: sorted_output_indices[1],
        self._OUTPUT_SCORE_NAME: sorted_output_indices[2],
        self._OUTPUT_NUMBER_NAME: sorted_output_indices[3],
    }

    self._input_size = input_detail['shape'][2], input_detail['shape'][1]
    self._is_quantized_input = input_detail['dtype'] == np.uint8
    self._interpreter = interpreter
    self._options = options

  def detect(self, input_image: np.ndarray) -> List[Detection]:
    """Run detection on an input image.
    Args:
        input_image: A [height, width, 3] RGB image. Note that height and width
          can be anything since the image will be immediately resized according
          to the needs of the model within this function.
    Returns:
        A Person instance.
    """
    image_height, image_width, _ = input_image.shape

    input_tensor = self._preprocess(input_image)

    self._set_input_tensor(input_tensor)
    self._interpreter.invoke()

    # Get all output details
    boxes = self._get_output_tensor(self._OUTPUT_LOCATION_NAME)
    classes = self._get_output_tensor(self._OUTPUT_CATEGORY_NAME)
    scores = self._get_output_tensor(self._OUTPUT_SCORE_NAME)
    count = int(self._get_output_tensor(self._OUTPUT_NUMBER_NAME))

    return self._postprocess(boxes, classes, scores, count, image_width,
                             image_height)

  def _preprocess(self, input_image: np.ndarray) -> np.ndarray:
    """Preprocess the input image as required by the TFLite model."""

    # Resize the input
    input_tensor = cv2.resize(input_image, self._input_size)

    # Normalize the input if it's a float model (aka. not quantized)
    if not self._is_quantized_input:
      input_tensor = (np.float32(input_tensor) - self._mean) / self._std

    # Add batch dimension
    input_tensor = np.expand_dims(input_tensor, axis=0)

    return input_tensor

  def _set_input_tensor(self, image):
    """Sets the input tensor."""
    tensor_index = self._interpreter.get_input_details()[0]['index']
    input_tensor = self._interpreter.tensor(tensor_index)()[0]
    input_tensor[:, :] = image

  def _get_output_tensor(self, name):
    """Returns the output tensor at the given index."""
    output_index = self._output_indices[name]
    tensor = np.squeeze(self._interpreter.get_tensor(output_index))
    return tensor

  def _postprocess(self, boxes: np.ndarray, classes: np.ndarray,
                   scores: np.ndarray, count: int, image_width: int,
                   image_height: int) -> List[Detection]:
    """Post-process the output of TFLite model into a list of Detection objects.
    Args:
        boxes: Bounding boxes of detected objects from the TFLite model.
        classes: Class index of the detected objects from the TFLite model.
        scores: Confidence scores of the detected objects from the TFLite model.
        count: Number of detected objects from the TFLite model.
        image_width: Width of the input image.
        image_height: Height of the input image.
    Returns:
        A list of Detection objects detected by the TFLite model.
    """
    results = []

    # Parse the model output into a list of Detection entities.
    for i in range(count):
      if scores[i] >= self._options.score_threshold:
        y_min, x_min, y_max, x_max = boxes[i]
        bounding_box = Rect(
            top=int(y_min * image_height),
            left=int(x_min * image_width),
            bottom=int(y_max * image_height),
            right=int(x_max * image_width))
        class_id = int(classes[i])
        category = Category(
            score=scores[i],
            label=self._label_list[class_id],  # 0 is reserved for background
            index=class_id)
        result = Detection(bounding_box=bounding_box, categories=[category])
        results.append(result)

    # Sort detection results by score ascending
    sorted_results = sorted(
        results,
        key=lambda detection: detection.categories[0].score,
        reverse=True)

    # Filter out detections in deny list
    filtered_results = sorted_results
    if self._options.label_deny_list is not None:
      filtered_results = list(
          filter(
              lambda detection: detection.categories[0].label not in self.
              _options.label_deny_list, filtered_results))

    # Keep only detections in allow list
    if self._options.label_allow_list is not None:
      filtered_results = list(
          filter(
              lambda detection: detection.categories[0].label in self._options.
              label_allow_list, filtered_results))

    # Only return maximum of max_results detection.
    if self._options.max_results > 0:
      result_count = min(len(filtered_results), self._options.max_results)
      filtered_results = filtered_results[:result_count]

    return filtered_results


_MARGIN = 10  # pixels
_ROW_SIZE = 10  # pixels
_FONT_SIZE = 0
_FONT_THICKNESS = 0
_TEXT_COLOR = (0, 0, 255)  # red

def visualize(
    image: np.ndarray,
    detections: List[Detection],
) -> np.ndarray:
  """Draws bounding boxes on the input image and return it.
  Args:
    image: The input RGB image.
    detections: The list of all "Detection" entities to be visualize.
  Returns:
    Image with bounding boxes.
  """
  seam_detetcted = False
  for detection in detections:
    category = detection.categories[0]
    class_name = category.label
    # Draw bounding_box
    
    if class_name == 'Seam':
        start_point = detection.bounding_box.left, detection.bounding_box.top
        end_point = detection.bounding_box.right, detection.bounding_box.bottom
        cv2.rectangle(image, start_point, end_point, _TEXT_COLOR, 1)

    else:
        start_point_1 = detection.bounding_box.left, detection.bounding_box.top
        end_point_1 = detection.bounding_box.right, detection.bounding_box.bottom
        cv2.rectangle(image, start_point_1, end_point_1, (0, 0, 0), -1)
        seam_detetcted = True
      
  return image, start_point, end_point, seam_detetcted

# ------------------------------------------------------------------------------------

def crop_image(image_path, left_upper_point, right_bottom_point, seam_state):
    image = image_path
    x1, y1 = left_upper_point
    x2, y2 = right_bottom_point
    cropped_image = image[y1+5:y2-5, x1+100:x2-100]
    
    if seam_state is not True:
        height, width, channels = cropped_image.shape
        x1, y1, x2, y2 = 0, height // 2, width, height // 2
        cv2.line(cropped_image, (x1, y1), (x2, y2), (0, 0, 0), (int(height*0.3)))
    
    return cropped_image

# ------------------------------------------------------------------------------------

def process_image(image, color_number):
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(image)
    img_col = ['img', 'r', 'g', 'b']
    
    height, width, channels = image.shape
    
    for i in range(255):
        thresh_value = i
        max_value = 255
        thresh_type = cv2.THRESH_BINARY
        _, binary_img = cv2.threshold(eval(img_col[color_number]), thresh_value, max_value, thresh_type)

        num_white_pixels = cv2.countNonZero(binary_img)
        num_black_pixels = binary_img.size - num_white_pixels
        try:
            a = num_white_pixels/num_black_pixels
        except:
            a = 1
        if (a <= 0.24 and a >=0.228):
            break
    
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)
    
    kernel_size = (5, 15)
    blurred_img = cv2.GaussianBlur(binary_img, kernel_size, 0, borderType=cv2.BORDER_CONSTANT)
    _, blurred_img = cv2.threshold(blurred_img, 127, 255, thresh_type)
    contours, hierarchy = cv2.findContours(blurred_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = np.zeros_like(image)
        
    new_contours = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area >= 10:
            new_contours.append(cnt)
        
    cv2.fillPoly(img, new_contours, (255, 255, 255))
    
    return img, blurred_img
# ------------------------------------------------------------------------------------

def calculate_count(image):
    upper_points, bottom_points, upper_points_diff, bottom_points_diff = [], [], [], []
    
    height, width, channels = image.shape
    img_empty = np.zeros((height, width, 3), np.uint8)
    cv2.line(img_empty, (0, height//2), (width, height//2), (255, 255, 255), 10)
    
    img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    for cnt in contours:
        M = cv2.moments(cnt)
        cx = int(M['m10'] / M['m00'])
        cy = int(M['m01'] / M['m00'])
        
        if cy < height // 2:
            upper_points.append(cx)
            cv2.circle(img_empty, (cx, cy), 5, (255, 0, 0), -1)
        else:
            bottom_points.append(cx)
            cv2.circle(img_empty, (cx, cy), 5, (0, 255, 0), -1)
            
    upper_points = sorted(upper_points)
    bottom_points = sorted(bottom_points)
    
    for i in range(len(upper_points) - 1):
        diff = upper_points[i + 1] - upper_points[i]
        upper_points_diff.append(diff)
        
    for i in range(len(bottom_points) - 1):
        diff = bottom_points[i + 1] - bottom_points[i]
        bottom_points_diff.append(diff) 
    
    all_diff = upper_points_diff + bottom_points_diff
    all_diff = sorted(all_diff)
        
    mod = statistics.mode(all_diff)
    suitable_values = []
    
    for value in all_diff:
        if (value <= (mod + 2) & value >= (mod - 2)):
            suitable_values.append(value)
    
    mean = sum(suitable_values) / len(suitable_values)
    
    return img_empty, mean
    
    
def Decide_color(background):
    background = cv2.bitwise_not(background)
    b, g, r = cv2.mean(background)[:3]
    
    if r > 140 and g > 140 and b > 140:
        return 0
    elif r < 50 and g < 50 and b < 50:
        return 0
    elif r > g and r > b:
        return 1
    elif g > r and g > b:
        return 2
    elif b > r and b > g:
        return 3
    
    
def find_most_possible_number(values):
    nums = [x for x in values if x >= 3]
    print("Possible Values -", nums)
    q1 = np.percentile(nums, 25)
    q3 = np.percentile(nums, 75)
    iqr = q3 - q1
    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr    
    filtered_nums = [num for num in nums if lower_bound <= num <= upper_bound]
    mean_value = np.mean(filtered_nums)    
    rounded_mean = round(mean_value)
    
    return rounded_mean


# ------------------------------------------------------------------------------------
image_name = "image142.jpg"
INPUT_IMAGE_URL = "A:/Internship MAS/22.02.2023/23.02 - dataset/dataset/"+ image_name
DETECTION_THRESHOLD = 0.1
TFLITE_MODEL_PATH = "android.tflite"

image = cv2.imread(INPUT_IMAGE_URL)
image = cv2.convertScaleAbs(image, alpha=0.8, beta=0)


max_size = max(image.shape[:2])
if max_size > 512:
    scale = 512 / max_size
    image = cv2.resize(image, (0,0), fx=scale, fy=scale)

options = ObjectDetectorOptions(
        num_threads=4,
        score_threshold=DETECTION_THRESHOLD,
)

detector = ObjectDetector(model_path=TFLITE_MODEL_PATH, options=options)
detections = detector.detect(image)
image, upper_left, botom_right, seam_state = visualize(image, detections)
cropped_image = crop_image(image, upper_left, botom_right, seam_state)

x, y, w, h = 100, 0, botom_right[0]-100, upper_left[1]
background = image[y:y+h, x:x+w]
color = Decide_color(background)

possible_count_values = []
for i in range(4):
    try:
        final_image, b = process_image(cropped_image, i)
        center_c, length = calculate_count(final_image)
        height, width, channels = center_c.shape
        possible_count_values.append(round(width/length))
        if i == color:
            img_concat = cv2.hconcat([cropped_image, final_image, center_c])
            cv2.imshow(str(i), img_concat)
        cv2.imwrite('A:/Internship MAS/22.02.2023/23.02 - dataset/dataset_4/'+str(i)+image_name, final_image)
        cv2.imwrite('A:/Internship MAS/22.02.2023/23.02 - dataset/dataset_5/'+str(i)+image_name, center_c)
        
    except:
        possible_count_values.append(0)

most_possible_value = find_most_possible_number(possible_count_values)
text = "Most Possible SPI - " + str(most_possible_value)

cv2.putText(image, text, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
cv2.imshow('Original', image)

end_time = time.time()
total_time = end_time - start_time
print("Processing time:", total_time, "seconds")

cv2.waitKey(0)
cv2.destroyAllWindows()
