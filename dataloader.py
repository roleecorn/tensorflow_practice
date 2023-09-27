import pathlib
from tensorflow import keras
import pathlib


def load_flower(**kwargs):
    """
    Load flother with specified parameters.
    
    Parameters:
    - batch_size (int, optional): Size of the batches of data. Default is 32.
    - img_height (int, optional): Height of the image. Default is 180.
    - img_width (int, optional): Width of the image. Default is 180.
    - mode (str, optional): Mode of the image (e.g., 'grayscale', 'rgb'). Default is 'grayscale'.

    Returns:
    None
    """
    dataset_url = "https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz"
    data_dir = keras.utils.get_file('flower_photos.tar', origin=dataset_url, extract=True)
    data_dir = pathlib.Path(data_dir).with_suffix('')
    batch_size = kwargs.get('batch_size', 32)
    img_height = kwargs.get('img_height', 180)
    img_width = kwargs.get('img_width', 180)
    mode = kwargs.get('mode', 'grayscale')
    validation_split = kwargs.get('validation_split', 0.2)
    train_ds = keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=validation_split,
      subset="training",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size,
      color_mode=mode,
      )
    val_ds = keras.utils.image_dataset_from_directory(
      data_dir,
      validation_split=validation_split,
      subset="validation",
      seed=123,
      image_size=(img_height, img_width),
      batch_size=batch_size,
      color_mode=mode,
      )
    return train_ds,val_ds