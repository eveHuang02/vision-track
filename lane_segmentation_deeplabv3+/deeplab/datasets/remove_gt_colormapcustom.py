import glob
import os.path
import numpy as np
from PIL import Image
import tensorflow as tf

FLAGS = tf.compat.v1.flags.FLAGS

tf.compat.v1.flags.DEFINE_string('original_gt_folder',
                                 './VOCdevkit/VOC2012/SegmentationClass',
                                 'Original ground truth annotations.')

tf.compat.v1.flags.DEFINE_string('segmentation_format', 'png', 'Segmentation format.')

tf.compat.v1.flags.DEFINE_string('output_dir',
                                 './VOCdevkit/VOC2012/SegmentationClassRaw',
                                 'Folder to save modified ground truth annotations.')


def _remove_colormap(filename, colormap):
    """Removes the color map from the annotation by converting RGB to class index.

    Args:
        filename: Ground truth annotation filename.
        colormap: A dictionary mapping RGB values to class indices.

    Returns:
        Annotation with class indices (no colormap).
    """
    image = np.array(Image.open(filename))  # Load the image as a numpy array
    class_annotation = np.zeros(image.shape[:2], dtype=np.uint8)  # Initialize empty class annotation array

    # Iterate over each pixel and map the RGB value to the class index
    for rgb, class_idx in colormap.items():
        class_annotation[np.all(image == rgb, axis=-1)] = class_idx

    return class_annotation


def _save_annotation(annotation, filename):
    """Saves the annotation as a png file.

    Args:
        annotation: Segmentation annotation.
        filename: Output filename.
    """
    pil_image = Image.fromarray(annotation.astype(dtype=np.uint8))
    with tf.io.gfile.GFile(filename, mode='w') as f:
        pil_image.save(f, 'PNG')


def main(unused_argv):
    # Define the colormap (RGB -> class index mapping for Background and Lane)
    colormap = {
        (0, 0, 0): 0,    # Background (0)
        (255, 255, 255): 1, # Lane (255)
    }

    # Create the output directory if not exists.
    if not tf.io.gfile.isdir(FLAGS.output_dir):
        tf.io.gfile.makedirs(FLAGS.output_dir)

    annotations = glob.glob(os.path.join(FLAGS.original_gt_folder,
                                         '*.' + FLAGS.segmentation_format))
    for annotation in annotations:
        raw_annotation = _remove_colormap(annotation, colormap)
        filename = os.path.basename(annotation)[:-4]
        _save_annotation(raw_annotation,
                         os.path.join(
                             FLAGS.output_dir,
                             filename + '.' + FLAGS.segmentation_format))


if __name__ == '__main__':
    tf.compat.v1.app.run()
