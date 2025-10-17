import cv2
import os

def gen_images_from_vid( vid_path, store_path ): 
    # Derived from code found here: https://www.geeksforgeeks.org/python/extract-images-from-video-in-python/ 
    
    # Read in the video 
    cam = cv2.VideoCapture( vid_path )

    # Create a folder to store the data 
    try:
        # Make it if it doesn't exist 
        if not os.path.exists(store_path):
            os.makedirs(store_path)

    # If not created then raise error
    except OSError:
        print (f'Error: Creating data storage directory at "{store_path}"')

    # frame
    current_frame = 0

    # Loop over all of the data 
    while(True):
        # Reading from frame
        ret, frame = cam.read() 

        if ret: 
            # If video is still left continue creating images
            name = f'{store_path}/frame{str(current_frame)}.jpg'
            print ('Creating...' + name)

            # Writing the extracted images
            cv2.imwrite(name, frame)

            # Increasing counter
            current_frame += 1
        else:
            break
        
    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()
    return 