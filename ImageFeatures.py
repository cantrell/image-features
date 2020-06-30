import numpy as np
import scipy
import scipy.spatial
import scipy.ndimage
import tensorflow as tf
import tensorflow_hub as hub
from PIL import Image, ImageOps, ImageChops
class ImageFeatures:
    def __init__(self):
        # self.module = hub.Module( "https://tfhub.dev/google/imagenet/inception_v3/feature_vector/1" )
        self.module = hub.Module( "./module" )
        self.img_h, self.img_w = hub.get_expected_image_size( self.module )
        self.inputs = {k: tf.placeholder(v.dtype, v.get_shape().as_list(), k) for k, v in self.module.get_input_info_dict().items()}
        self.output = self.module( self.inputs )
    def cropToSquare(self, img_pil, axis_len, centering=(0.5,0.5)):
        return ImageOps.fit( img_pil, ( axis_len, axis_len ), method=Image.NEAREST, bleed=0.0, centering=centering )
    def resizeImage(self, img_pil, targ_width, targ_height, method=Image.ANTIALIAS):
        return img_pil.resize( ( targ_width, targ_height ), method )
    def conformImage(self, img, targ_w, targ_h):
        if targ_w != img.width or targ_h != img.height:
            img = self.cropToSquare( img , max( targ_w, targ_h ) )
        arr = np.asarray( img )
        arr = arr.reshape((1, targ_w, targ_h, 3))
        arr = np.float32( arr )
        arr = arr / 255.0
        arr = np.clip( arr, 0.0, 255.0 )
        return arr
    def deconformImage(self, arr, targ_w, targ_h):
        orig_w = arr.shape[ 1 ]
        orig_h = arr.shape[ 2 ]
        arr = arr.reshape((orig_w, orig_h, 3))
        arr = arr * 255.0
        arr = np.clip( arr, 0, 255 )
        arr = np.uint8( arr )
        img = Image.fromarray( arr )
        if targ_w != orig_w or targ_h != orig_h:
            img = self.resizeImage( img, targ_w, targ_h )
        return img
    def getFeatures(self, images):
        ims = []
        if isinstance( images, (list,) ):
            for img in images:
                ims.append( self.conformImage( img, self.img_w, self.img_h ) )
            ims = np.concatenate( ims, axis=0 )
        else:
            ims = self.conformImage( images, self.img_w, self.img_h )
        with tf.Session() as session:
            initializer = tf.global_variables_initializer()
            session.run( initializer )
            input_op  = self.inputs['images']
            num_rows  = ims.shape[ 0 ]
            batch_sze = 8
            features  = []
            for batch_start in range(0, num_rows, batch_sze):
                s = slice(batch_start, min(num_rows, batch_start + batch_sze))
                feed_dict = { input_op: ims[s] }
                features.append( session.run( self.output, feed_dict=feed_dict ) )
            features = np.concatenate( features, axis=0 )
        session.close()
        return features
    def getFeatureSimilarity(self, features_a, features_b):
        return 1.0 - scipy.spatial.distance.cosine( features_a, features_b )
    def getImageSimilarity(self, image_a, image_b):
        feature_vecs = self.getFeatures( [ image_a, image_b ] )
        similarity   = self.getFeatureSimilarity( feature_vecs[ 0, : ], feature_vecs[ 1, : ] )
        return similarity
