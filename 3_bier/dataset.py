from __future__ import division
from __future__ import print_function

import h5py
import datasets as datasets
import numpy as np
import tensorflow as tf
#tf.enable_eager_execution()
from PIL import Image
import hypia
import random

def class_random_sampler(image_dict, image_list, batch_size,samples_per_class):
    sampler_length     = len(image_list)//batch_size
    assert  batch_size% samples_per_class==0, '#Samples per class must divide batchsize!'
    batch_ids = []
    for _ in range(sampler_length):
        subid = []
        for _ in range(batch_size//samples_per_class):
            class_key = random.choice(list(image_dict.keys()))
            subid.extend([random.choice(image_dict[class_key])[-1] for _ in range(samples_per_class)])
        batch_ids.append(subid)
    return batch_ids

def get_dataset(dataset_name,source_path, is_training):
    # select the train/val dataset
    data_set = datasets.select(dataset_name,source_path)
    if is_training :
        hdf_file = source_path +'/' + dataset_name +'/train.hdf5' 
        data_dict  = data_set["training"]
    else:
        hdf_file = source_path +'/' + dataset_name +'/val.hdf5' 
        data_dict = data_set["validation"]

    # image dict contains {label:[image_path,idx]}
    image_dict = {}
    counter = 0
    for key in data_dict.keys():
        image_dict[key] = []
        for path in data_dict[key]:
            image_dict[key].append([path, counter])
            counter += 1

    # image_list contains: [image_path, idx, label]
    image_list = [[(x[0],int(key)) for x in image_dict[key]] for key in image_dict.keys()]
    image_list = [x for y in image_list for x in y]
    return image_dict, image_list, hdf_file


def process_image(img, crop=None, mean=None,std= None,
                  mirror=True, is_training=True):
    """
    Preprocessing code. For training this function randomly crop images and
    flips the image randomly.
    For testing we use the center crop of the image.

    Args:
        img: The input image.
        crop: Size of the output image.
        mean: three dimensional array indicating the mean values to subtract
              from the image.
        mirror: Flag, which indicates if images should be mirrored.
        is_training: Flag which indicates whether training preprocessing
                     or testing preprocessing should be used.

    Returns:
        A pre-processed image.
    """
    if img.get_shape():
        img_dim,_,num_channels = img.get_shape()
        if is_training:
            # random_image_crop
            if img_dim == crop: tl =[0,0]
            else: tl = np.random.choice(range(img_dim-crop),2)
            img = hypia.functionals.crop(img, tl, crop, crop,channel_pos='last')
            #img = tf.image.random_crop(img, [crop, crop, num_channels], name='random_image_crop')
            if mirror:
                choice = np.random.choice([1,2],1)
                if choice ==1 :
                    img = hypia.functionals.hflip(img,channel_pos='last')
                else:
                    img = hypia.functionals.vflip(img,channel_pos='last')

        else:
            offset = (img_dim - crop) // 2
            img = hypia.functionals.crop(img, [offset,offset], crop, crop,channel_pos='last')

        # normalize
        img = hypia.functionals.normalise(img,mean,std) 
    return tf.to_float(img)


class NpyDatasetProvider(object):
    """
    This class hooks up a numpy dataset file to tensorflow queue runners.
    """
    def __init__(self, data_spec, dataset_name,source_path, 
                samples_per_class=None,batch_size=32, is_training=True, num_concurrent=4):
        super(NpyDatasetProvider, self).__init__()
        self.data_spec = data_spec
        self.batch_size = batch_size
        self.is_training = is_training
        self.samples_per_class = samples_per_class

        # The data specifications describe how to process the image
        self.new_image_shape = [data_spec.crop_size, data_spec.crop_size, data_spec.channels]
        if 'MLRSNet' in dataset_name:
            self.image_shape = [256,256,3]
        if 'BigEarthNet' in dataset_name:
            self.image_shape = [120,120,12]

        image_dict, image_list,self.hdf_file = get_dataset(dataset_name,source_path,is_training)
        if self.is_training:
            self.batch_ids = class_random_sampler(image_dict, image_list, self.batch_size, self.samples_per_class)
        self.image_paths = np.array(image_list)[:,0]
        self.labels = np.array(image_list)[:,-1]

        if self.labels.dtype != np.int32:
            self.labels = self.labels.astype(np.int32)

        self.original_num_images = len(self.image_paths)
        self.num_images = len(self.image_paths)

        if not self.is_training and self.num_images % self.batch_size != 0:
            to_pad = self.batch_size - (self.num_images % self.batch_size)
            pad_path = np.array(["pad.pad"]*to_pad)
            pad_label = -np.ones([to_pad], dtype=np.int32)
            self.labels = np.r_[self.labels, pad_label]
            self.image_paths = np.r_[self.image_paths, pad_path]
            self.num_images = len(self.image_paths )
        
        self.unique_labels = np.unique(self.labels)
        self.setup(num_concurrent)

    def _setup_test(self, num_concurrent):
        """
        Setup the test queue.

        Args:
            num_concurrent: Number of concurrent threads.
        """
        self.num_batches = self.num_images // self.batch_size
        indices = tf.range(self.num_images)

        self.preprocessing_queue = tf.FIFOQueue(capacity=self.num_images,
                                                dtypes=[tf.int32],
                                                shapes=[()],
                                                name='preprocessing_queue')
        self.test_queue_op = self.preprocessing_queue.enqueue_many([indices])

        processed_queue = tf.FIFOQueue(capacity=self.num_images,
                                       dtypes=[tf.int32, tf.float32],
                                       shapes=[(), self.new_image_shape],
                                       name='processed_queue')
        label, img = self.process_test()
        enqueue_processed_op = processed_queue.enqueue([label, img])

        self.dequeue_op = processed_queue.dequeue_many(self.batch_size)
        num_concurrent = min(num_concurrent, self.num_images)
        self.queue_runner = tf.train.QueueRunner(
            processed_queue,
            [enqueue_processed_op] * num_concurrent)
        tf.train.add_queue_runner(self.queue_runner)

    def setup(self, num_concurrent):
        """
        Setup of the queues.

        Args:
            num_concurrent: Number of concurrent threads.
        """
        if self.is_training:
            return self._setup_train(num_concurrent)
        else:
            return self._setup_test(num_concurrent)

    def _setup_train(self, num_concurrent):
        """
        Setup of the training queues.

        Args:
            num_concurrent: Number of concurrent threads.
        """
        self.num_batches = self.num_images // self.batch_size

        # Crate a batch queue.
        self.batch_queue = tf.RandomShuffleQueue(
            capacity=self.num_batches,
            min_after_dequeue=0,
            dtypes=[tf.int32],
            shapes=[()],
            name='batch_queue')
        batch_indices = tf.range(self.num_batches)
        self.batch_queue_op = self.batch_queue.enqueue_many([batch_indices])

        (labels, processed_images) = self.process()
       
        processed_queue = tf.FIFOQueue(  # capacity=self.num_images,
            capacity=self.batch_size * 6,
            dtypes=[tf.int32, tf.float32],
            shapes=[(), self.new_image_shape],
            name='processed_queue')

        # Enqueue the processed image and path
        enqueue_processed_op = processed_queue.enqueue_many(
            [labels, processed_images])

        self.dequeue_op = processed_queue.dequeue_many(self.batch_size)
        num_concurrent = min(num_concurrent, self.num_images)
        #self.batch_runner = tf.compat.v1.train.QueueRunner(self.batch_queue, [self.batch_queue_op] * (num_concurrent + 1))
        self.queue_runner = tf.train.QueueRunner(processed_queue,[enqueue_processed_op] * num_concurrent)
        #tf.train.add_queue_runner(self.batch_runner)
        tf.train.add_queue_runner(self.queue_runner)

    def start(self, session, coordinator, num_concurrent=4):
        """
        Start the processing worker threads.

        Args:
            session: A tensorflow session.
            coordinator: A tensorflow coordinator.
            num_concurrent: Number of concurrent threads.

        Returns:
            a create threads operation.
        """
        if self.is_training:
            #self.batch_runner.create_threads(session, coord=coordinator, start=True)
            session.run(self.batch_queue_op) # enqueue batch indices once
        else:
            session.run(self.test_queue_op)  # just enqueue labels once!
        return self.queue_runner.create_threads(session, coord=coordinator, start=True)

    def feed_data(self, session):
        """
        Call this function for testing NpyDatasetProvider. It pushes
        the testing dataset once into the queue.

        Args:
            session: A tensorflow session
        """
        assert(not self.is_training)
        session.run(self.test_queue_op)  # just enqueue labels once!

    def get(self, session):
        """
        Get a single batch of images along with their labels.

        Returns:
            a tuple of (labels, images)
        """
        (labels, images) = session.run(self.dequeue_op)
        return (labels, images)

    def batches(self, session):
        """
        Yield a batch until no more images are left.

        Yields:
            Tuples in the form (labels, images)
        """
        for _ in range(self.num_batches):
            yield self.get(session=session)
   
    def fetch_images(self, the_idx):
        img_path = self.image_paths[the_idx]
        patch_name = img_path.split('/')[-1]
        if  ".pad" in patch_name:
            input_image = np.zeros(self.image_shape)
        else:
            f = h5py.File(self.hdf_file, 'r')
            data = f[patch_name][()]
            f.close()
            input_image = data.reshape(self.image_shape[2],self.image_shape[0],self.image_shape[1])
            input_image = np.transpose(input_image,(1,2,0))
        processed_img = process_image(img=input_image,
                                      crop=self.data_spec.crop_size,
                                      mean=self.data_spec.mean,
                                      std =self.data_spec.std,
                                      is_training=self.is_training)
        return processed_img

    def fetch_labels(self, the_idx):
        return self.labels[the_idx]

    def process_test(self):
        """
        Processes the test images.

        Returns:
            Tuple consisting of (label, processed_image).
        """
        index = self.preprocessing_queue.dequeue()
        label = tf.py_func(self.fetch_labels, [index], tf.int32)
        the_img = tf.py_func(self.fetch_images, [index], tf.float32)
        label.set_shape([])
        the_img.set_shape(self.new_image_shape)
        return (label, the_img)
    
    def process(self):
        """
        Processes a training image.

        Returns:
            A tuple consisting of (label, image).
        """
        def get_inds(batch_id):
            return self.batch_ids[batch_id]

        batch_id = self.batch_queue.dequeue()
        inds = tf.stack(tf.py_func(get_inds,[batch_id],tf.int32))
        label = tf.py_func(self.fetch_labels,[inds], tf.int32)
        the_img = tf.py_func(self.fetch_images, [inds], tf.float32)
        label.set_shape([self.batch_size])
        the_img.set_shape([self.batch_size]+ self.new_image_shape)
        return (label, the_img)