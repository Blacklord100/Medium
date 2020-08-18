#########
######### Preprocessing file to adapat all your images to vgg input.
######### This formatting is needed as we will be using the last prediction layer which is a fully connected layer (size matter :D)
#########
from keras.preprocessing.image import load_img,img_to_array
from keras.applications import vgg16
import numpy as np


def format_img_vgg(img_file,img_width=224,img_height=224):
    """
    This function reads and formats an image so that it can be fed to the VGG16 network.   
    Parameters
    ----------
    img_file : image file name
    img_width : the target image width
    img_height : he target image height
    Returns
    -------
    img_out_vgg : the correctly formatted image for VGG16
    img : the image as read by the load_img function of keras.preprocessing.image
    """
    img = load_img(img_file,target_size=(img_height,img_width))
    img_out = img_to_array(img)
    img_out_vgg = vgg16.preprocess_input(img_out)
    # add dimemsion for batch training
    img_out_vgg = np.expand_dims(img_out_vgg, axis=0)
    return img_out_vgg, img

def unformat_image(img_in):
    """
    This function inverts the preprocessing applied to images for use in the VGG16 network
    Parameters
    ----------
    img_file : formatted image of shape (batch_size,m,n,3)
    Returns
    -------
    img_out : a m-by-n-by-3 array, representing an image that can be written to an image file
    """
    # delete batch dim
    img_out=np.squeeze(img_in)
    #remove VGG16 preprocessing
    img_out[:, :, 0] += 103.939
    img_out[:, :, 1] += 116.779
    img_out[:, :, 2] += 123.68
    # BGR to RGB
    img_out = img_out[:, :, ::-1]
    # histogramm normalization between 0 to 255
    img_out = np.clip(img_out, 0, 255).astype('uint8')
    return img_out