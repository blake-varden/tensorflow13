import tensorflow as tf


def load_optimizer(learning_rate_config, optimizer_config, steps_per_epoch, global_step):
    """
    constructs an optimizer using the leraning rate and optmizer config.

    :param learning_rate_config: 
    :param optimizer_conifg: 
    :return: optimizer constructed using confs

    """
    learning_rate = configure_learning_rate(learning_rate_config,
                                            steps_per_epoch,
                                            global_step)
    optimizer = configure_optimizer(optimizer_config,
                                    learning_rate)
    return optimizer


def configure_learning_rate(learning_rate_config, steps_per_epoch, global_step):
    """Configures the learning rate.

    Args:
        num_samples_per_epoch: The number of samples in each epoch of training.
        global_step: The global_step tensor.

    Returns:
        A `Tensor` representing the learning rate.

    Raises:
        ValueError: if
    """
    lr_type = learning_rate_config['class']
    lr_params = learning_rate_config['params']
    learning_rate = learning_rate_config['learning_rate']

    if lr_type == 'exponential':
        return lr_exponential(learning_rate,
                              steps_per_epoch,
                              global_step,
                              lr_params)
    elif lr_type == 'fixed':
        return lr_fixed(learning_rate)
    elif lr_type == 'polynomial':
        return lr_polynomial(learning_rate,
                             steps_per_epoch,
                             global_step,
                             lr_params)
    else:
        raise ValueError('learning_rate_decay_type [%s] was not recognized', lr_type)


def lr_exponential(learning_rate, steps_per_epoch, global_step, learning_rate_params):
    epochs_per_decay = learning_rate_params['epochs_per_decay']
    decay_steps = steps_per_epoch / epochs_per_decay

    learning_rate_decay_factor = learning_rate_params['learning_rate_decay_factor']
    staircase = learning_rate_params.get('learning_rate_config', False)
    return tf.train.exponential_decay(learning_rate,
                                      global_step,
                                      decay_steps,
                                      learning_rate_decay_factor,
                                      staircase=staircase,
                                      name='exponential_decay_learning_rate')


def lr_fixed(learning_rate):
    return tf.constant(learning_rate, name='fixed_learning_rate')


def lr_polynomial(learning_rate, steps_per_epoch, global_step, learning_rate_params):
    epochs_per_decay = learning_rate_params['epochs_per_decay']
    decay_steps = steps_per_epoch / epochs_per_decay
    end_learning_rate = learning_rate_params['end_learning_rate']
    power = learning_rate_params.get('power', 1.0)
    cycle = learning_rate_params.get('cycle', True)
    return tf.train.polynomial_decay(learning_rate,
                                     global_step,
                                     decay_steps,
                                     end_learning_rate,
                                     power=power,
                                     cycle=cycle,
                                     name='polynomial_decay_learning_rate')


def configure_optimizer(optimizer_config, learning_rate):
    """Configures the optimizer used for training.

    Args:
        learning_rate: A scalar or `Tensor` learning rate.

    Returns:
        An instance of an optimizer.

    Raises:
        ValueErrr: if optimizer_config['optimizer is not recognized.
    """

    opt_type = optimizer_config['class']
    opt_params = optimizer_config['params']
    if opt_type == 'adadelta':
        optimizer = tf.train.AdadeltaOptimizer(
            learning_rate,
            rho=opt_params['adadelta_rho'],
            epsilon=opt_params['opt_epsilon'])
    elif opt_type == 'adagrad':
        optimizer = tf.train.AdagradOptimizer(
            learning_rate,
            initial_accumulator_value=opt_params['adagrad_initial_accumulator_value'])
    elif opt_type == 'adam':
        optimizer = tf.train.AdamOptimizer(
            learning_rate,
            beta1=opt_params['adam_beta1'],
            beta2=opt_params['adam_beta2'],
            epsilon=opt_params['opt_epsilon'])
    elif opt_type == 'ftrl':
        optimizer = tf.train.FtrlOptimizer(
            learning_rate,
            learning_rate_power=opt_params['ftrl_learning_rate_power'],
            initial_accumulator_value=opt_params['ftrl_initial_accumulator_value'],
            l1_regularization_strength=opt_params['ftrl_l1'],
            l2_regularization_strength=opt_params['ftrl_l2'])
    elif opt_type == 'momentum':
        optimizer = tf.train.MomentumOptimizer(
            learning_rate,
            momentum=opt_params['momentum'],
            name='Momentum')
    elif opt_type == 'rmsprop':
        optimizer = tf.train.RMSPropOptimizer(
            learning_rate,
            decay=opt_params['rmsprop_decay'],
            momentum=opt_params['rmsprop_momentum'],
            epsilon=opt_params['opt_epsilon'])
    elif opt_type == 'sgd':
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    else:
        raise ValueError('Optimizer [%s] was not recognized', opt_type)
    return optimizer
