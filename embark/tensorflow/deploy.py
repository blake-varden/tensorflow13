import tensorflow as tf
from tensorflow.python.ops.control_flow_ops import with_dependencies
from tensorflow.python.summary import summary
from tensorflow.contrib.training.python.training import evaluation
from tensorflow.python.training import monitored_session
from tensorflow.python.training import saver as tf_saver
from embark.tensorflow.optimizers import create_optimizer
import imp

create_optimizer = create_optimizer
slim = tf.contrib.slim
_USE_DEFAULT = slim.evaluation._USE_DEFAULT
TOWER_NAME = 'tower'
MOVING_AVERAGE_DECAY = 0.9999


def create_test_op(model_config, inputs, num_gpus=1):
    with tf.get_default_graph().as_default(), tf.device('/cpu:0'):
        model = None
        losses = []
        total_loss = []
        global_step = slim.get_or_create_global_step()
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        losses, total_loss, model = tower_loss(model_config, inputs, scope, is_train=False)
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

        summaries = []
        for l in losses + [total_loss]:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            loss_name = l.op.name
            loss_summary = tf.summary.scalar(loss_name, l)
            summaries.append(loss_summary)

        with tf.name_scope('test_op'):
            # Ensure the train_tensor computes grad_updates.
            test_op = tf.identity(total_loss)

        return test_op, model, summaries


def create_train_op(model_config, inputs, opt, num_gpus=1, histograms=False):
    with tf.get_default_graph().as_default(), tf.device('/cpu:0'):
        tower_grads = []
        model = None
        losses = []
        total_loss = []
        global_step = slim.get_or_create_global_step()
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (TOWER_NAME, i)) as scope:
                        # Calculate the loss for one tower of the CIFAR model. This function
                        # constructs the entire CIFAR model but shares the variables across
                        # all towers.
                        losses, total_loss, model = tower_loss(model_config, inputs, scope, is_train=True)
                        # Reuse variables for the next tower.
                        tf.get_variable_scope().reuse_variables()

                        # Calculate the gradients for the batch of data on this CIFAR tower.
                        grads = opt.compute_gradients(total_loss)

                        # Keep track of the gradients across all towers.
                        tower_grads.append(grads)

        summaries = []
        for l in losses + [total_loss]:
            # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
            # session. This helps the clarity of presentation on tensorboard.
            loss_name = l.op.name
            loss_summary = tf.summary.scalar(loss_name, l)
            summaries.append(loss_summary)
        # We must calculate the mean of each gradient. Note that this is the
        # synchronization point across all towers.
        grads = average_gradients(tower_grads)

        # Add histograms for gradients.
        if histograms:
            for grad, var in grads:
                if grad is not None:
                    summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))

        # Apply the gradients to adjust the shared variables.
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

        # Add histograms for trainable variables.
        if histograms:
            for var in tf.trainable_variables():
                summaries.append(tf.summary.histogram(var.op.name, var))

        # Track the moving averages of all trainable variables.
        variable_averages = tf.train.ExponentialMovingAverage(
            MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())

        # Group all updates to into a single train op.
        train_op = tf.group(apply_gradient_op, variables_averages_op)

        with tf.name_scope('train_op'):
            # Ensure the train_tensor computes grad_updates.
            train_op = with_dependencies([train_op], total_loss)

        return train_op, model, summaries


def tower_loss(model_config, inputs, scope, is_train=True):
    """Calculate the total loss on a single tower running the CIFAR model.
    Args:
      scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'
    Returns:
       Tensor of shape [] containing the total loss for a batch of data
    """
    # Get images and labels for CIFAR-10.

    model = create_model(model_config, inputs, is_train=is_train)
    update_ops = set(tf.get_collection(tf.GraphKeys.UPDATE_OPS))
    update_ops.update(model.batch_ops())

    # Assemble all of the losses for the current tower only.
    losses = tf.losses.get_losses(scope=scope)

    # Calculate the total loss for the current tower.
    total_loss = tf.add_n(losses, name='sum_losses')
    total_loss = tf.check_numerics(total_loss,
                                   'LossTensor is inf or nan', name='check_loss_numerics')

    # Make sure update_ops are computed before total_loss.
    with tf.control_dependencies(update_ops):
        barrier = tf.no_op(name='update_barrier')
    total_loss = with_dependencies([barrier], total_loss, name='total_loss')

    # Attach a scalar summary to all individual losses and the total loss; do the
    # same for the averaged version of the losses.

    return losses, total_loss, model


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.
    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def create_train_step(data_provider, model, steps_per_epoch):
    def train_step(sess, train_op, global_step, train_step_kwargs):
        # Begin loading of data
        if not train_step_kwargs.get('data_loaded', False):
            data_provider.start(sess)
            train_step_kwargs['data_loaded'] = True
        summary_writer = train_step_kwargs['summary_writer']
        cur_step = sess.run(global_step)

        # Beginning of Epoch Operations
        begin_epoch = cur_step % steps_per_epoch == 0
        if begin_epoch:
            sess.run(model.begin_epoch_ops())

        if begin_epoch and model.begin_epoch_summary() is not None:
            begin_summary = sess.run(model.begin_epoch_summary())
            summary_writer.add_summary(begin_summary, cur_step)

        # Batch Train Step
        train_step_res = slim.learning.train_step(sess, train_op, global_step, train_step_kwargs)

        # End of Epoch Operations
        end_epoch = cur_step % steps_per_epoch == steps_per_epoch - 1

        if end_epoch:
            sess.run(model.end_epoch_ops())
        if end_epoch and model.end_epoch_summary() is not None:
            end_summary = sess.run(model.end_epoch_summary())
            summary_writer.add_summary(end_summary, cur_step)

        return train_step_res

    return train_step


def create_restore_fn(restore_config):

    restore_path = restore_config['checkpoint']
    if restore_path is None:
        return None
        
    include_variables = restore_config['include_variables']
    exclude_variables = ['global_step']
    if restore_config['exclude_variables'] is not None:
        exclude_variables.extend(restore_config['exclude_variables'])

    variables_to_restore = slim.get_variables_to_restore(include=include_variables,
                                                         exclude=exclude_variables)

    init_assign_op, init_feed_dict = slim.assign_from_checkpoint(
        restore_path, variables_to_restore)

    # Create an initial assignment function.


    def InitAssignFn(sess):
        sess.run(init_assign_op, init_feed_dict)

    return InitAssignFn


def evaluation_loop(master,
                    checkpoint_dir,
                    logdir,
                    num_evals=1,
                    initial_op=None,
                    initial_op_feed_dict=None,
                    eval_op=None,
                    eval_op_feed_dict=None,
                    final_op=None,
                    final_op_feed_dict=None,
                    summary_op=_USE_DEFAULT,
                    summary_op_feed_dict=None,
                    variables_to_restore=None,
                    eval_interval_secs=60,
                    max_number_of_evaluations=None,
                    session_config=None,
                    timeout=None,
                    hooks=None):
    """Runs TF-Slim's Evaluation Loop.
    Args:
      master: The BNS address of the TensorFlow master.
      checkpoint_dir: The directory where checkpoints are stored.
      logdir: The directory where the TensorFlow summaries are written to.
      num_evals: The number of times to run `eval_op`.
      initial_op: An operation run at the beginning of evaluation.
      initial_op_feed_dict: A feed dictionary to use when executing `initial_op`.
      eval_op: A operation run `num_evals` times.
      eval_op_feed_dict: The feed dictionary to use when executing the `eval_op`.
      final_op: An operation to execute after all of the `eval_op` executions. The
        value of `final_op` is returned.
      final_op_feed_dict: A feed dictionary to use when executing `final_op`.
      summary_op: The summary_op to evaluate after running TF-Slims metric ops. By
        default the summary_op is set to tf.summary.merge_all().
      summary_op_feed_dict: An optional feed dictionary to use when running the
        `summary_op`.
      variables_to_restore: A list of TensorFlow variables to restore during
        evaluation. If the argument is left as `None` then
        slim.variables.GetVariablesToRestore() is used.
      eval_interval_secs: The minimum number of seconds between evaluations.
      max_number_of_evaluations: the max number of iterations of the evaluation.
        If the value is left as 'None', the evaluation continues indefinitely.
      session_config: An instance of `tf.ConfigProto` that will be used to
        configure the `Session`. If left as `None`, the default will be used.
      timeout: The maximum amount of time to wait between checkpoints. If left as
        `None`, then the process will wait indefinitely.
      hooks: Hooks to use while evaluating
    Returns:
      The value of `final_op` or `None` if `final_op` is `None`.
    """

    if summary_op == _USE_DEFAULT:
        summary_op = summary.merge_all()

    if hooks is None:
        hooks = []
    hooks.append(evaluation.StopAfterNEvalsHook(num_evals))

    if summary_op is not None:
        hooks.append(evaluation.SummaryAtEndHook(
            log_dir=logdir, summary_op=summary_op, feed_dict=summary_op_feed_dict))

    saver = None
    if variables_to_restore is not None:
        saver = tf_saver.Saver(variables_to_restore)

    return evaluation.evaluate_repeatedly(
        checkpoint_dir,
        master=master,
        scaffold=monitored_session.Scaffold(
            init_op=initial_op, init_feed_dict=initial_op_feed_dict, saver=saver),
        eval_ops=eval_op,
        feed_dict=eval_op_feed_dict,
        final_ops=final_op,
        final_ops_feed_dict=final_op_feed_dict,
        eval_interval_secs=eval_interval_secs,
        hooks=hooks,
        config=session_config,
        max_number_of_evaluations=max_number_of_evaluations,
        timeout=timeout)


def create_model(model_config, inputs, is_train=True):
    model_file = model_config['file']
    model_mod = imp.load_source('file_loaded_architecture', model_file)
    model_params = model_config['params']
    model_class = model_mod.model
    model = model_class(inputs, is_train=is_train, **model_params)
    return model


def create_data_provider(data_provider_config, tower_batch_size, data_source):
    data_provider_file = data_provider_config['file']
    data_provider_mod = imp.load_source('file_loaded_architecture', data_provider_file)
    data_provider_class = data_provider_mod.data_provider

    data_provider_params = data_provider_config['data_provider_params']
    data_provider = data_provider_class(tower_batch_size, **data_provider_params)
    source = data_source['source']
    max_num_examples = data_source['num_examples']
    data_provider.add_data_source(source, max_num_examples=max_num_examples)
    return data_provider


def get_steps(examples_per_epoch,
              num_gpus,
              num_examples,
              num_epochs,
              tower_batch_size):
    num_examples = num_examples if num_examples is not None else examples_per_epoch * num_epochs

    num_steps = num_examples / (num_gpus * tower_batch_size)
    steps_per_epoch = examples_per_epoch / (num_gpus * tower_batch_size)
    return num_steps, steps_per_epoch
