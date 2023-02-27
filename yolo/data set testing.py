import os
import cv2

count = 0

folder_path = "C:/Users/Bimsara Sandaruwan/Pictures/Screenshots"

file_names = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f)) and f.lower().endswith((".png", ".jpg", ".jpeg"))]

for file_name in file_names:
    count = count + 1
    image = cv2.imread(os.path.join(folder_path, file_name))
    cv2.imshow("file_name", image)
    cv2.waitKey(0)
    text = input("Enter text for image {}: ".format(file_name))
    text = str("real-")+text
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 4
    font_thickness = 5
    text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
    text_x = image.shape[1] - text_size[0] - 10
    text_y = image.shape[0] - 10
    cv2.putText(image, text, (text_x, text_y), font, font_scale, (0, 0, 255), font_thickness)
    
    cv2.imwrite(os.path.join("A:/Internship MAS/22.02.2023/testing results", "{}.jpg".format(str(count))), image)
    print("Image saved with name", text)
