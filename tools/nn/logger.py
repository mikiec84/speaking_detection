import tensorflow as tf
import os.path

class TFLogger(object):
    """ Creates an "empty model" that writes Tensorflow summaries. Can
        visualize these summaries with Tensorboard.
    """
    def __init__(self, summary_dir):
        super(TFLogger, self).__init__()
        self.summary_dir = summary_dir
        self.__initialize()

    def __initialize(self):
        sess = tf.Session()
        loss = tf.Variable(0.0, name="loss", trainable=False)
        acc = tf.Variable(0.0, name="accuracy", trainable=False)
        loss_summary = tf.summary.scalar("loss", loss)
        acc_summary = tf.summary.scalar("accuracy", acc)
        summary_op = tf.summary.merge([loss_summary, acc_summary])
        summary_writer = tf.summary.FileWriter(self.summary_dir, sess.graph)
        saver = tf.train.Saver(tf.all_variables())
        sess.run(tf.initialize_all_variables())

        self.sess = sess
        self.summary_op = summary_op
        self.summary_writer = summary_writer
        self.loss = loss
        self.acc = acc

    def log(self, step, loss, accuracy):
        feed_dict = {
            self.loss: loss,
            self.acc: accuracy,
        }

        # sess.run returns a list, so we have to explicitly
        # extract the first item using sess.run(...)[0]
        summaries = self.sess.run([self.summary_op], feed_dict)[0]
        self.summary_writer.add_summary(summaries, step)
        
train_tf_logger = TFLogger('../data/logs/train')
# eval_tf_logger = TFLogger(os.path.join('.', 'summaries', 'eval'))
# 
# 
# for step, (x_batch, y_batch) in enumerate(batch_iterator):
#     acc, loss = model.train(x_batch, y_batch)
#     train_tf_logger.log(step=step, accuracy=acc, loss=loss)
# 
#     if step % eval_step == 0:
#         acc, loss = evalute(model)
#         eval_tf_logger.log(step=step, accuracy=acc, loss=loss)
