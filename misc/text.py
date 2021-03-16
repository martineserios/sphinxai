from PIL import ImageFont, ImageDraw, Image  
import cv2  
import numpy as np  
   
   
def write_image_custom_font(frame, text:str, path_to_font:str, color, size, coords:tuple):
    # Convert the image to RGB (OpenCV uses BGR)  
    cv2_im_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Pass the image to PIL  
    pil_im = Image.fromarray(cv2_im_rgb)  
    
    draw = ImageDraw.Draw(pil_im)  
    # use a truetype font  
    font = ImageFont.truetype(path_to_font, size)  
    
    # Draw the text  
    draw.text(coords, text, font=font)  
    
    # Get back the image to OpenCV  
    cv2_im_processed = cv2.cvtColor(np.array(pil_im), cv2.COLOR_RGB2BGR)  
    
    return cv2_im_processed