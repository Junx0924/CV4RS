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
        # The data specifications describe how to process the image
        self.data_spec = data_spec
        if 'MLRSNet' in dataset_name:
            self.image_shape = [256,256,3]
            self.tf_image_type = tf.uint8
        if 'BigEarthNet' in dataset_name:
            self.image_shape = [120,120,12]
            self.tf_image_type = tf.float32

        # select the train/val dataset
        data_set = datasets.select(dataset_name,source_path)
        if is_training :
            self.hdf_file = source_path +'/' + dataset_name +'/train.hdf5' 
            data_dict  = data_set["training"]
        else:
            self.hdf_file = source_path +'/' + dataset_name +'/val.hdf5' 
            data_dict = data_set["validation"]

        # image dict contains {label:[image_path,idx]}
        self.image_dict = {}
        counter = 0
        for key in data_dict.keys():
            self.image_dict[key] = []
            for path in data_dict[key]:
                self.image_dict[key].append([path, counter])
                counter += 1

        # image_list contains: [image_path, idx, label]
        self.image_list = [[(x[0],int(key)) for x in self.image_dict[key]] for key in self.image_dict.keys()]
        self.image_list = [x for y in self.image_list for x in y]
        
        self.image_paths = np.array(self.image_list)[:,0]
        self.labels = np.array(self.image_list)[:,-1]

        if self.labels.dtype != np.int32:
            self.labels = self.labels.astype(np.int32)

        self.original_num_images = len(self.image_paths)
        self.batch_size = batch_size
        self.is_training = is_training
        self.num_images = len(self.image_paths)
        self.samples_per_class = samples_per_class

        if not self.is_training and self.num_images % self.batch_size != 0:
            to_pad = self.batch_size - (self.num_images % self.batch_size)
            pad_path = np.array(["pad.pad"]*to_pad)
            pad_label = -np.ones([to_pad], dtype=np.int32)
            self.labels = np.r_[self.labels, pad_label]
            self.image_paths = np.r_[self.image_paths, pad_path]
            self.num_images = len(self.image_paths )
        
        self.unique_labels = np.unique(self.labels)
        self.tf_unique_labels = tf.convert_to_tensor(self.unique_labels)

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

        image_shape = (self.data_spec.crop_size,
                       self.data_spec.crop_size, self.data_spec.channels)
        processed_queue = tf.FIFOQueue(capacity=self.num_images,
                                       dtypes=[tf.int32, tf.float32],
                                       shapes=[(), image_shape],
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

        # Crate a label queue.
        self.label_queue = tf.queue.RandomShuffleQueue(
            capacity=len(self.unique_labels),
            min_after_dequeue=0,
            dtypes=[tf.int32],
            shapes=[()],
            name='label_queue')
        self.label_queue_op = self.label_queue.enqueue_many(
            [self.tf_unique_labels])

        (labels, processed_images) = self.process()

        image_shape = (self.data_spec.crop_size,
                       self.data_spec.crop_size, self.data_spec.channels)
        processed_queue = tf.FIFOQueue(  # capacity=self.num_images,
            capacity=self.batch_size * 6,
            dtypes=[tf.int32, tf.float32],
            shapes=[(), image_shape],
            name='processed_queue')

        # Enqueue the processed image and path
        enqueue_processed_op = processed_queue.enqueue_many(
            [labels, processed_images])

        self.dequeue_op = processed_queue.dequeue_many(self.batch_size)
        num_concurrent = min(num_concurrent, self.num_images)
        self.label_runner = tf.compat.v1.train.QueueRunner(
            self.label_queue, [self.label_queue_op] * (num_concurrent + 1))
        self.queue_runner = tf.train.QueueRunner(
            processed_queue,
            [enqueue_processed_op] * num_concurrent)
        tf.train.add_queue_runner(self.label_runner)
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
            self.label_runner.create_threads(
                session, coord=coordinator, start=True)
        else:
            session.run(self.test_queue_op)  # just enqueue labels once!
        return self.queue_runner.create_threads(
            session, coord=coordinator, start=True)

    def feed_data(self, session):
        """
        Call this function for testing NpyDatasetProvider. It pushes
        the testing dataset once into the queue.

        Args:
            session: A tensorflow session
        """
        assert(not self.is_training)
        session.run(self.test_queue_op)  # just enqueue labels once!

    def get_labels(self, session):
        """
        Returns a list of labels from the queue.

        Args:
            session: A tensorflow session.

        Returns:
            An array of labels from the queue.
        """
        labels = session.run(
            self.label_queue.dequeue_many(len(self.unique_labels)))
        return labels

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
        for _ in xrange(self.num_batches):
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
        return input_image

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
        the_img = tf.py_func(self.fetch_images, [index], self.tf_image_type)
        label.set_shape([])
        the_img.set_shape(self.image_shape)
        processed_img = process_image(img=the_img,
                                      #img=self.tf_images[index, ...],
                                      crop=self.data_spec.crop_size,
                                      mean=self.data_spec.mean,
                                      std =self.data_spec.std,
                                      is_training=self.is_training)
        # return (self.tf_labels[index], processed_img)
        return (label, processed_img)

    def process(self):
        """
        Processes a training image.

        Returns:
            A tuple consisting of (label, image).
        """
        def class_random_sampler(sampled_class):
            assert self.batch_size%self.samples_per_class==0, '#Samples per class must divide batchsize!'
            all_ids = []
            for _ in range(self.batch_size//self.samples_per_class):
                class_key = random.choice(list(sampled_class))
                all_ids.extend([random.choice(self.image_dict[class_key])[-1] for _ in range(self.samples_per_class)])
            valid_labels = self.labels[all_ids]
            valid_images = [self.fetch_images(ids) for ids in all_ids]
            return valid_labels, valid_images

        labels = self.label_queue.dequeue_many(len(self.unique_labels))
        labels.set_shape([len(self.unique_labels)])

        results = tf.py_func(class_random_sampler, [labels], [tf.int32, self.tf_image_type])
        labels, images = results[0],tf.stack(results[1:])
        labels.set_shape([self.batch_size])
        #images.set_shape([self.batch_size] + list(self.img_shape))

        processed_images = []
        for i in range(self.batch_size):
            # Process the image
            processed_img = process_image(img=images[i, ...],
                                          crop=self.data_spec.crop_size,
                                          mean=self.data_spec.mean,
                                          std =self.data_spec.std,
                                          is_training=self.is_training)
            processed_images.append(processed_img)
        processed_images = tf.stack(processed_images)
        return (labels, processed_images)
