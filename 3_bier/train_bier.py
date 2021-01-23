from __future__ import print_function
from __future__ import division

import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR) 
from tensorflow.contrib import slim
import argparse
import collections
import numpy as np
import random
import os
import time

import flip_gradient
import parameters as par
from utilities import misc
from utilities import logger
import model.resnet50 as models
import dataset
from evaluate import similarity, nmi,recall




def do_print(*args, **kwargs):
    """
    Wrapper around tf.Print to enable/disable verbose printing.
    """
    if VERBOSE:
        return tf.Print(*args, **kwargs)
    else:
        return args[0]


def embedding_tower(hidden_layer, embedding_sizes, reuse=False):
    """
    Creates the embedding tower on top of a feature extractor.

    Args:
        hidden_layer: Last hidden layer of the feature extractor.
        embedding_sizes: array indicating the sizes of our
                         embedding, e.g. [96, 160, 256]
        reuse: tensorflow reuse flag for the variable scope.
    Returns: A tuple consisting of the embedding and end_points.
    """
    end_points = {}
    final_layers = []

    with tf.variable_scope(EMBEDDING_SCOPE_NAME, reuse=reuse) as scope:
        hidden_layer = slim.flatten(hidden_layer)
        for idx, embedding_size in enumerate(embedding_sizes):

            scope_name = 'embedding/fc_{}'.format(idx)
            embedding = slim.fully_connected(
                hidden_layer, embedding_size, activation_fn=None,
                scope=scope_name)
            regul_out = slim.fully_connected(tf.stop_gradient(
                hidden_layer), embedding_size, scope=scope_name,
                reuse=True, activation_fn=None, biases_initializer=None)

            end_points['{}/embedding/fc_{}'.format(
                EMBEDDING_SCOPE_NAME, idx)] = embedding
            end_points['{}/embedding/fc_{}_regularizer'.format(
                EMBEDDING_SCOPE_NAME, idx)] = regul_out
            final_layers.append(embedding)

        embedding = tf.concat(final_layers, axis=1)
        end_points['{}/embedding'.format(EMBEDDING_SCOPE_NAME)] = embedding

        weight_variables = slim.get_variables_by_name('weights', scope=scope)
    for w in weight_variables:
        tf.add_to_collection('weights', w)
    return embedding, end_points

def get_embeddings(sess, data_provider,preds, hidden_output, labels,embedding_size,batch_size):
    """
    Get the feature vectors from the model

    Args:
        sess: tf.compat.v1.train.MonitoredTrainingSession 
        data_provider:  dataset
        hidden_output:  feature vectors from the model
        preds:  feature vectors from the embedding_tower 
        labels: labels
        embedding_size: default [96,160,256]
    Returns:
        a tuple contains:
        nomalized numpy array of feature vectors of the data_provider
        nomalized numpy array of hidden feature vectors of the data_provider
        labels
    """
    all_fvecs = []
    all_fvecs_hidden = []
    all_labels = []
    num_batches = int(np.ceil(data_provider.num_images / float(batch_size)))
    print('Evaluating {} batches'.format(num_batches))
    for batch_idx in range(num_batches):
        fvec, fvec_hidden, cls = sess.run([preds, hidden_output, labels])
        fvec = fvec[cls >= 0, ...]
        fvec_hidden = fvec_hidden[cls >= 0, ...]
        cls = cls[cls >= 0, ...]
        all_fvecs.append(np.array(fvec))
        all_fvecs_hidden.append(np.array(fvec_hidden[:, 0, 0, :]))
        all_labels.append(np.array(cls))
    
    fvecs = np.vstack(all_fvecs)
    fvecs_hidden = np.vstack(all_fvecs_hidden)
    labels = np.concatenate(all_labels)

    # normalize the feature vectors
    fvecs_hidden /= np.maximum(1e-5, np.linalg.norm(fvecs_hidden, axis=1, keepdims=True))
    
    # normalize the feature vectors of different learners
    embedding_scales = [float(e) / sum(embedding_size) for e in embedding_size]
    start_idx = 0
    for e, s in zip(embedding_size, embedding_scales):
        stop_idx = start_idx + e
        fvecs[:, start_idx:stop_idx] /= np.maximum(1e-5, np.linalg.norm(fvecs[:, start_idx:stop_idx], axis=1, keepdims=True)) / s
        start_idx = stop_idx
    return  fvecs, fvecs_hidden, labels
        

def evaluate(query, query_labels, gallery,gallery_labels, LOG, log_key ='',backend="faiss-gpu",with_nmi = False):
    """
    Evaluation the retrieval performance.

    Args:
        query: numpy array of feature vectors of query dataset
        gallery: numpy array of feature vectors of gallery dataset
        query_labels: labels
        gallery_labels: labels
        backend: faiss-gpu
        with_nmi: calculate nmi
    Returns:
        The recall @1, 2, 4, 8
        nmi if with_nmi = True
    """
    if log_val =="Val_h":
        print("Calculate metrics for hidden layer features")
    else:
        print("Calculate metrics for bier layer features")
    K = [1, 2, 4, 8]
    nb_classes = len(set(query_labels))
    assert nb_classes == len(set(gallery_labels))
    # calculate full similarity matrix, choose only first `len(query)` rows
    # and only last columns corresponding to the column
    T_eval = np.concatenate([query_labels, gallery_labels],axis=0)
    X_eval = np.concatenate([query, gallery],axis=0)
    D = similarity.pairwise_distance(X_eval)[:len(query_labels), len(query_labels):]

    # get top k labels with smallest distance
    ind = similarity.get_sorted_top_k(D,max(K),axis = 1)
    Y = gallery_labels[ind]

    flag_checkpoint = False
    history_recall1 = 0
    if "recall" not in LOG.progress_saver[log_key].groups.keys():
        flag_checkpoint = True
    else: 
        history_recall1 = np.max(LOG.progress_saver[log_key].groups['recall']["recall@1"]['content'])

    scores = {}
    recall = []
    for k in K:
        r_at_k = recall.calculate(query_labels, Y, k)
        recall.append(r_at_k)
        LOG.progress_saver[log_key].log("recall@"+str(k),r_at_k,group='recall')
        print("recall@{} : {:.3f}".format(k, 100 * r_at_k))
        if k==1 and r_at_k > history_recall1:
            flag_checkpoint = True
        
    scores['recall'] = recall

    if with_nmi:
        # calculate NMI with kmeans clustering
        nmi = nmi.calculate(T_eval,similarity.cluster_by_kmeans(X_eval, nb_classes,gpu_id=0, backend=backend))
        LOG.progress_saver[log_key].log("nmi",nmi)
        print("NMI: {:.3f}".format(nmi * 100))
        scores['nmi'] = nmi

    return scores,flag_checkpoint
    

def build_train(predictions, end_points, y, embedding_sizes,lambda_weight,
                shrinkage=0.06,
                lambda_div=0.0, C=25, alpha=2.0, beta=0.5, initial_acts=0.5,
                eta_style=False, dtype=tf.float32, regularization=None):
    """
    Builds the boosting based training.

    Args:
        predictions: tensor of the embedding predictions
        end_points: dictionary of endpoints of the embedding tower
        y: tensor class labels
        embedding_sizes: list, which indicates the size of the sub-embedding
                         (e.g. [96, 160, 256])
        shrinkage: if you use eta_style = True, set to 1.0, otherwise keep it
                   small (e.g. 0.06).
        lambda_div: regularization parameter.
        C: parameter for binomial deviance.
        alpha: parameter for binomial deviance.
        dtype: data type for computations, typically tf.float32
        initial_acts: 0.5 if eta_style is false, 0.0 if eta_style is true
        regularization: regularization method (either activation or
                        adversarial)
    Returns:
        The training loss.
    """
    shape = predictions.get_shape().as_list()
    num_learners = len(embedding_sizes)
    # Pairwise labels.
    pairs = tf.reshape(
        tf.cast(tf.equal(y[:, tf.newaxis], y[tf.newaxis, :]), dtype), [-1])

    m = 1.0 * pairs + (-C * (1.0 - pairs))
    W = tf.reshape((1.0 - tf.eye(shape[0], dtype=dtype)), [-1])
    W = W * pairs / tf.reduce_sum(pairs) + W * \
        (1.0 - pairs) / tf.reduce_sum(1.0 - pairs)

    # * boosting_weights_init
    boosting_weights = tf.ones(shape=(shape[0] * shape[0],), dtype=dtype)

    normed_fvecs = []
    regular_fvecs = []

    # L2 normalize fvecs
    for i in range(len(embedding_sizes)):
        start = int(sum(embedding_sizes[:i]))
        stop = int(start + embedding_sizes[i])

        fvec = tf.cast(predictions[:, start:stop], dtype)
        regular_fvecs.append(fvec)
        fvec = do_print(fvec, [tf.norm(fvec, axis=1)],
                        'fvecs_{}_norms'.format(i))
        tf.summary.histogram('fvecs_{}'.format(i), fvec)
        tf.summary.histogram('fvecs_{}_norm'.format(i), tf.norm(fvec, axis=1))
        normed_fvecs.append(
            fvec / tf.maximum(tf.constant(1e-5, dtype=dtype),
                              tf.norm(fvec, axis=1, keep_dims=True)))

    alpha = tf.constant(alpha, dtype=dtype)
    beta = tf.constant(beta, dtype=dtype)
    C = tf.constant(C, dtype=dtype)
    shrinkage = tf.constant(shrinkage, dtype=dtype)

    loss = tf.constant(0.0, dtype=dtype)
    acts = tf.constant(initial_acts, dtype=dtype)
    tf.summary.histogram('boosting_weights_0', boosting_weights)
    tf.summary.histogram('boosting_weights_0_pos', tf.boolean_mask(
        boosting_weights, tf.equal(pairs, 1.0)))
    tf.summary.histogram('boosting_weights_0_neg', tf.boolean_mask(
        boosting_weights, tf.equal(pairs, 0.0)))
    Ds = []
    for i in range(len(embedding_sizes)):
        fvec = normed_fvecs[i]
        Ds.append(tf.matmul(fvec, tf.transpose(fvec)))

        D = tf.reshape(Ds[-1], [-1])
        my_act = alpha * (D - beta) * m
        my_loss = tf.log(tf.exp(-my_act) + tf.constant(1.0, dtype=dtype))
        tmp = (tf.reduce_sum(my_loss * boosting_weights * W) /
               tf.constant(num_learners, dtype=dtype))
        loss += tmp

        tf.summary.scalar('learner_loss_{}'.format(i), tmp)

        if eta_style:
            nu = 2.0 / (1.0 + 1.0 + i)
            if shrinkage != 1.0:
                acts = (1.0 - nu) * acts + nu * shrinkage * D
                inputs = alpha * (acts - beta) * m
                booster_loss = tf.log(tf.exp(-(inputs)) + 1.0)
                boosting_weights = tf.stop_gradient(
                    -tf.gradients(tf.reduce_sum(booster_loss), inputs)[0])
            else:
                acts = (1.0 - nu) * acts + nu * shrinkage * my_act
                booster_loss = tf.log(tf.exp(-acts) + 1.0)
                boosting_weights = tf.stop_gradient(
                    -tf.gradients(tf.reduce_sum(booster_loss), acts)[0])
        else:
            # simpler variant of the boosting algorithm.
            acts += shrinkage * (D - beta) * alpha * m
            booster_loss = tf.log(tf.exp(-acts) + 1.0)
            cls_weight = tf.cast(1.0 * pairs + (1.0 - pairs) * 2.0,
                                 dtype=dtype)
            boosting_weights = tf.stop_gradient(-tf.gradients(
                tf.reduce_sum(booster_loss), acts)[0] * cls_weight)

            tf.summary.histogram(
                'boosting_weights_{}'.format(i + 1), boosting_weights)
            pos_weights = tf.boolean_mask(
                boosting_weights, tf.equal(pairs, 1.0))
            neg_weights = tf.boolean_mask(
                boosting_weights, tf.equal(pairs, 0.0))
            pos_bins = tf.histogram_fixed_width(pos_weights, (tf.constant(
                0.0, dtype=dtype), tf.constant(1.0, dtype=dtype)), nbins=10)
            neg_bins = tf.histogram_fixed_width(neg_weights, (tf.constant(
                0.0, dtype=dtype), tf.constant(1.0, dtype=dtype)), nbins=10)
            loss = do_print(loss, [tf.reduce_mean(
                booster_loss)], 'Booster loss {}'.format(i + 1))
            loss = do_print(loss, [pos_bins, neg_bins],
                            'Positive and negative boosting weights {}'.format(
                                i + 1), summarize=100)

            tf.summary.histogram(
                'boosting_weights_{}_pos'.format(i + 1), pos_weights)
            tf.summary.histogram(
                'boosting_weights_{}_neg'.format(i + 1), neg_weights)
            tf.summary.scalar('booster_loss_{}'.format(
                i + 1), tf.reduce_mean(booster_loss))

    # add the independence loss
    tf.summary.scalar('discriminative_loss', loss)

    embedding_weights = [v for v in tf.get_collection(
        tf.GraphKeys.GLOBAL_VARIABLES) if 'embedding' in v.name and
                                          'weight' in v.name]
    if lambda_div > 0.0:
        loss += REGULARIZATION_FUNCTIONS[regularization](
            fvecs=normed_fvecs, end_points=end_points,
            embedding_weights=embedding_weights,
            embedding_sizes=embedding_sizes,
            lambda_weight=lambda_weight,dtype=dtype) * tf.dtypes.cast(lambda_div, dtype)
    tf.summary.scalar('loss', loss)
    return  tf.dtypes.cast(loss, dtype)


def build_pairwise_tower_loss(fvecs_i, fvecs_j, scope=None,lambda_weight=100.0,dtype=tf.float32):
    """
    Builds an adversarial regressor from fvecs_j to fvecs_i.

    Args:
        fvecs_i: the target embedding (i.e. the smaller embedding)
        fvecs_j: the source embedding (i.e. the begger embedding)
        scope: scope name of the regressor.
        lambda_weight: the regularization parameter for the weights.
    Returns:
        An adversarial regressor loss function.
    """
    # build a regressor from fvecs_j to fvecs_i
    fvecs_i = flip_gradient.flip_gradient(fvecs_i)
    fvecs_j = flip_gradient.flip_gradient(fvecs_j)
    net = fvecs_j

    bias_loss = 0.0
    weight_loss = 0.0
    adversarial_loss = 0.0
    with tf.variable_scope(scope):
        for i in range(NUM_HIDDENS_ADVERSARIAL):
            if i < NUM_HIDDENS_ADVERSARIAL - 1:
                net = slim.fully_connected(
                    net, HIDDEN_ADVERSARIAL_SIZE, scope='fc_{}'.format(i),
                    activation_fn=tf.nn.relu)
            else:
                net = slim.fully_connected(net, fvecs_i.get_shape().as_list(
                )[-1], scope='fc_{}'.format(i), activation_fn=None)
            b = slim.get_variables(
                scope=tf.get_variable_scope().name + '/fc_{}/biases'.format(i)
            )[0]
            W = slim.get_variables(
                scope=tf.get_variable_scope().name + '/fc_{}/weights'.format(i)
            )[0]
            weight_loss += tf.reduce_mean(
                tf.square(tf.reduce_sum(W * W, axis=1) - 1)) * lambda_weight
            if b is not None:
                bias_loss += tf.maximum(
                    0.0,
                    tf.reduce_sum(b * b) - 1.0) * lambda_weight
        adversarial_loss += -tf.reduce_mean(tf.square(fvecs_i * net))

        tf.summary.scalar('adversarial loss', adversarial_loss)
        tf.summary.scalar('weight loss', weight_loss)
        tf.summary.scalar('bias loss', bias_loss)
        pairwise_tower_loss = adversarial_loss + weight_loss + bias_loss
    return  tf.dtypes.cast(pairwise_tower_loss, dtype)


def adversarial_loss(fvecs, end_points, embedding_weights, embedding_sizes,
                     lambda_weight,dtype = tf.float32):
    """
    Applies the adversarial loss on our embedding.

    Args:
        fvecs: tensor of the embedding feature vectors.
        end_points: dictionary of end_points of the embedding tower.
        embedding_weights: weight matrices of the embedding.
        embedding_sizes: list of embedding sizes, e.g. [96, 160, 256]
        lambda_weight: weight regularization parameter.
    Returns:
        The regularization loss.
    """
    loss = 0.0
    with tf.variable_scope('pws'):
        for layer_idx, fvecs in enumerate(iterate_regularization_acts(
                end_points, embedding_sizes)):

            for i in range(len(fvecs)):
                for j in range(i + 1, len(fvecs)):
                    name = 'pw_tower_loss_layer_{}_from_{}_to_{}'.format(
                        layer_idx, i, j)

                    loss += build_pairwise_tower_loss(
                        fvecs[i], fvecs[j],
                        name,
                        lambda_weight=lambda_weight, dtype = dtype)

    weight_loss = 0.0
    for W in embedding_weights:
        weight_loss += tf.reduce_mean(
            tf.square(tf.reduce_sum(W * W, axis=1) - 1))

    weight_loss = do_print(weight_loss, [weight_loss], 'weight loss')
    loss = do_print(loss, [loss], 'adversarial correlation dann hidden loss')
    tf.summary.scalar('adversarial correlation dann hidden losss', loss)
    tf.summary.scalar('weight loss', weight_loss)

    adv_loss = loss + lambda_weight * tf.dtypes.cast(weight_loss, dtype)
    return tf.dtypes.cast(adv_loss, dtype)


def iterate_regularization_acts(end_points, embedding_sizes):
    """
    Iterates through the regularization activations.

    Args:
        end_points: Dictionary of end_points.
        embedding_sizes: List of embedding sizes, e.g. [96, 160, 256].
    Yields:
        All iteration endpoints
    """
    num_embeddings = len(embedding_sizes)

    fvecs = []
    # yield the output layer.
    for i in range(num_embeddings):
        fvecs.append(end_points[EMBEDDING_SCOPE_NAME +
                                '/embedding/fc_{}_regularizer'.format(i)])
    yield fvecs


def activation_loss(fvecs, end_points, embedding_weights, embedding_sizes,
                    lambda_weight,dtype= tf.float32):
    """
    Applies the activation loss on our embedding.

    Args:
        fvecs: embedding tensors.
        end_points: dictionary of end_points from embedding_tower.
        embedding_weights: weight matrices of embeddings
        embedding_sizes: list of embedding sizes, e.g. [96, 160, 256].
        lambda_weight: Weight regularization parameter.
    Returns:
        The activation loss.
    """
    loss = 0.0
    for fvecs in iterate_regularization_acts(end_points, embedding_sizes):
        print(fvecs)
        for i in range(len(fvecs)):
            for j in range(i + 1, len(fvecs)):
                loss += tf.reduce_mean(
                    tf.square(fvecs[i][:, tf.newaxis, :] *
                              fvecs[j][:, :, tf.newaxis]))

    weight_loss = 0.0
    for W in embedding_weights:
        weight_loss += tf.reduce_mean(
            tf.square(tf.reduce_sum(W * W, axis=0) - 1))
    weight_loss = do_print(weight_loss, [weight_loss], 'weight loss')
    loss = do_print(loss, [loss], 'group loss')
    act_loss = loss + weight_loss * lambda_weight
    return tf.dtypes.cast(act_loss, dtype)

def load_config():
    parser = argparse.ArgumentParser()
    parser = par.basic_training_parameters(parser)
    parser = par.setup_parameters(parser)
    parser = par.wandb_parameters(parser)
    ##### Read in parameters
    args = parser.parse_args()
    if args.savename =="":
        args.savename = args.dataset_name +'_s{}'.format(args.seed)
    if args.logdir == "":
        args.logdir = os.path.dirname(__file__) + '/log'
    if args.log_online:
        import wandb
        os.environ['WANDB_API_KEY'] = args.wandb_key
        os.environ["WANDB_MODE"] = "dryrun" # for wandb logging on HPC
        _ = os.system('wandb login --relogin {}'.format(args.wandb_key))
        ## update savename
        args.savename = args.group+'_s{}'.format(args.seed)
        wandb.init(project=args.project, group=args.group, name=args.savename, dir=args.save_path)
        wandb.config.update(args)
    args.save_path   += '/'+args.dataset_name
    return args

def main():
    dtype = tf.float32
    args = load_config()
    LEARNING_RATE = args.lr
    lr_decay = args.lr_decay
    weight_decay = args.weight_decay
    batch_size = args.batch_size
    lambda_weight = args.lambda_weight
    regularization = args.regularization
    NUM_HIDDENS_ADVERSARIAL = args.num_hidden_adversarial
    HIDDEN_ADVERSARIAL_SIZE = args.hidden_adversarial_size

    embedding_sizes = args.embedding_sizes
    skip_test = args.skip_test
    random.seed(args.seed)
    np.random.seed(args.seed)
    tf.set_random_seed(args.seed)

    #################### CREATE LOGGING FILES ###############
    sub_loggers = ['Train', 'Val', 'Val_h']
    LOG = logger.LOGGER(args, sub_loggers=sub_loggers, start_new=True, log_online= args.log_online)

    if 'MLRSNet' in args.dataset_name:
        crop_size = 224
        channels = 3
        mean = [0.485, 0.456, 0.406]
        std =  [0.229, 0.224, 0.225]
    if 'BigEarthNet' in args.dataset_name:
        crop_size = 100
        channels =12
        mean = 0.485
        std =  0.229

    TrainingData = collections.namedtuple('TrainingData', ('crop_size', 'channels', 'mean','std'))
    spec = TrainingData(crop_size, channels, mean,std)
    print('creating datasets...')
    train_provider = dataset.DatasetProvider(
        data_spec=spec,
        samples_per_class=args.samples_per_class,
        dataset_name=args.dataset_name,
        source_path= args.source_path,
        dataset_type ="train",
        batch_size=batch_size,
        num_concurrent =4,
        is_training = True)
    
    train_labels, train_data = train_provider.dequeue_op
    if not skip_test:
        query_provider = dataset.DatasetProvider(
            data_spec=spec,
            dataset_name=args.dataset_name,
            source_path= args.source_path,
            dataset_type = "query",
            batch_size=batch_size,
            num_concurrent =4,
            is_training=False)
        query_labels, query_data = query_provider.dequeue_op

        gallery_provider = dataset.DatasetProvider(
            data_spec=spec,
            dataset_name=args.dataset_name,
            source_path= args.source_path,
            dataset_type = "gallery",
            batch_size=batch_size,
            num_concurrent =4,
            is_training=False)
        gallery_labels, gallery_data = gallery_provider.dequeue_op

    
    net = models.ResNet50({'data': train_data})
    hidden_layer = net.get_output()
    preds, end_points = embedding_tower(hidden_layer, embedding_sizes)
    end_points['pool5_7x7_s1'] = hidden_layer
    if not skip_test:
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            query_net = models.ResNet50({'data': query_data}, trainable=False)
            query_hidden_layer = query_net.get_output()
            query_preds, query_endpoints = embedding_tower(
                query_hidden_layer,
                embedding_sizes,
                reuse=True)
            gallery_net = models.ResNet50({'data': gallery_data}, trainable=False)
            gallery_hidden_layer = gallery_net.get_output()
            gallery_preds, query_endpoints = embedding_tower(
                gallery_hidden_layer,
                embedding_sizes,
                reuse=True)
    loss = build_train(
        preds,
        end_points,
        train_labels,
        embedding_sizes,
        lambda_weight,
        shrinkage=args.shrinkage,
        lambda_div=args.lambda_div,
        eta_style=args.eta_style,
        dtype=dtype,
        regularization=args.regularization)

    # Add weight decay.
    all_weights = tf.get_collection('weights')
    all_weights = list(set(all_weights))
    for w in all_weights:
        loss += tf.cast(tf.reduce_sum(w * w) * weight_decay, dtype=dtype)

    all_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
    hidden_vars = [v for v in all_vars if 'embedding' not in v.name]
    embedding_vars = [v for v in all_vars if 'embedding' in v.name]

    global_step = tf.train.get_or_create_global_step()

    lr = tf.constant(LEARNING_RATE, dtype=dtype,shape=(), name='learning_rate')
    if args.lr_anneal:
        lr = tf.train.exponential_decay(
            lr, global_step, args.lr_anneal, lr_decay, staircase=True)
    lr = do_print(lr, [lr], 'learning rate')

    opt_hidden = tf.train.AdamOptimizer(learning_rate=lr)
    train_op_hidden = opt_hidden.minimize(loss, var_list=hidden_vars)

    opt_embedding = tf.train.AdamOptimizer(learning_rate=lr* args.embedding_lr_multiplier)
    train_op_embedding = opt_embedding.minimize(loss, global_step=global_step, var_list=embedding_vars)

    with tf.control_dependencies([train_op_hidden, train_op_embedding]):
        train_op = tf.no_op()

    init_op = tf.global_variables_initializer()
    net_weights = os.path.dirname(__file__) + '/model/weights/resnet50.npy'
   
    with tf.control_dependencies([init_op]):
        load_train_op = net.create_load_op(net_weights, ignore_missing=True)
        if not skip_test:
            load_query_op = query_net.create_load_op(net_weights, ignore_missing=True)
            load_gallery_op = gallery_net.create_load_op(net_weights, ignore_missing=True)

    checkpoint_saver = tf.train.CheckpointSaverHook(
        args.logdir,
        save_steps=args.eval_every,
        saver=tf.train.Saver(max_to_keep=100000))
    latest_checkpoint = tf.train.latest_checkpoint(args.logdir)
   
    if latest_checkpoint is None:
        start_iter = 0
    else:
        start_iter = int(latest_checkpoint.split('-')[-1])
        assign_op = global_step.assign(start_iter)

    best_recall = 0.0
    best_epoch = 0
    t1 = time.time()
    print("Initialize training...")
    with tf.compat.v1.train.MonitoredTrainingSession(
            checkpoint_dir=args.logdir,
            is_chief=True,
            hooks=[checkpoint_saver],save_summaries_steps=None, 
            save_summaries_secs=None,
            save_checkpoint_secs=None) as sess:
        if start_iter == 0:
            sess.run(init_op)
            sess.run(load_train_op)
            if not skip_test:
                sess.run(load_query_op)
                sess.run(load_gallery_op)
        else:
            sess.run(assign_op)
        print('Initialization elapsed time: {:.2f} s'.format(time.time() - t1))
        writer = tf.summary.FileWriter(args.logdir)
        for i in range(start_iter, args.num_iterations):
            time_per_epoch_1 = time.time()
            lossval, _ = sess.run([loss, train_op])
            time_per_epoch_2 = time.time()
            LOG.progress_saver['Train'].log('epochs', e)
            LOG.progress_saver["Train"].log('Train_loss',lossval)
            LOG.progress_saver['Train'].log('Train_time', np.round(time_per_epoch_2 - time_per_epoch_1, 4))
            print("Epoch: {}, loss: {}, time (seconds): {:.2f}.".format(i,lossval,time_per_epoch_2 - time_per_epoch_1))

            # for evaluation
            if not skip_test:
                print("Start evaluation...")
                tic = time.time()
                query_provider.feed_data(sess)
                gallery_provider.feed_data(sess)
                query, query_hidden, query_labels = get_embeddings(sess, query_provider,query_preds, query_hidden_output, query_labels,embedding_sizes ,batch_size)
                gallery, gallery_hidden, gallery_labels = get_embeddings(sess, gallery_provider,gallery_preds, gallery_hidden_output, gallery_labels,embedding_sizes,batch_size)
                score, checkpoint_flag = evaluate(query, query_labels, gallery,gallery_labels, LOG, log_key ='Val')
                score_hidden, _ = evaluate(query_hidden, query_labels, gallery_hidden,gallery_labels, LOG, log_key ='Val_h')
                
                LOG.progress_saver['Val'].log('Val_time', np.round(time.time() - tic, 4))
                LOG.update(all=True)
                if checkpoint_flag == True: 
                    best_epoch = i
                    best_recall = score['recall'][0] # take recall@1
                    print('Best epoch!')
                print('Evaluation total elapsed time: {:.2f} s'.format(time.time() - tic))
    t2 = time.time()
    print( "Total training time (minutes): {:.2f}.".format((t2 - t1) / 60))
    print("Best recall@1 = {} at epoch {}.".format(best_recall, best_epoch))

############## Gobal variables #############
REGULARIZATION_FUNCTIONS = {'activation': activation_loss,'adversarial': adversarial_loss}
EMBEDDING_SCOPE_NAME = 'embedding_tower'
VERBOSE = False
NUM_HIDDENS_ADVERSARIAL = 2
HIDDEN_ADVERSARIAL_SIZE = 512

if __name__ == '__main__':
    main()
