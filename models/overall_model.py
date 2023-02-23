import numpy as np
import tensorflow as tf
import tensorflow.keras as keras


class overallModel(keras.Model):
    def __init__(self, adaptor, generator, latent_dim=None):
        super(overallModel, self).__init__()
        self.adaptor = adaptor
        self.generator = generator
        self.latent_dim = latent_dim

    def compile(self, d_optimizer, g_optimizer, d_loss_fn, g_loss_fn, metrics=None, loss_weight=None):
        super(overallModel, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.d_loss_fn = d_loss_fn
        self.g_loss_fn = g_loss_fn
        self.mymetrics = metrics
        # self.adaptor.compile(metrics=metrics[0])
        # self.generator.compile(metrics=metrics[1:])
        self.loss_weight = loss_weight

    def train_step(self, data):

        x_data = data[0]
        y_adapt_data = data[1][0]
        y_pose_data = data[1][1:]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            out_pred = self.generator(x_data, training=True)
            out_adapton = self.adaptor(out_pred[0], training=True)
            adapt_loss = self.d_loss_fn(y_adapt_data, out_adapton)

            out = {"adapt-acc": self.mymetrics[0][0](y_adapt_data, out_adapton)}
            pred_loss = 0

            # sum = np.array(self.loss_weight).sum(axis=0)

            for i in range(0, len(y_pose_data)):
                pred_loss += self.g_loss_fn[i](y_pose_data[i], out_pred[i + 1]) * self.loss_weight[i + 1]
                self.mymetrics[i + 1][0].update_state(y_pose_data[i], out_pred[i + 1])
                out.update({"s" + str(i + 1) + "-AnalogRecall": self.mymetrics[i + 1][0].result()})
            overall_Loss = pred_loss - self.loss_weight[0] * adapt_loss
            # overall_Loss = pred_loss
        gradients_of_generator = gen_tape.gradient(overall_Loss, self.generator.trainable_variables)
        gradients_of_discriminator = disc_tape.gradient(adapt_loss, self.adaptor.trainable_variables)

        self.g_optimizer.apply_gradients(zip(gradients_of_generator, self.generator.trainable_variables))
        self.d_optimizer.apply_gradients(zip(gradients_of_discriminator, self.adaptor.trainable_variables))
        # self.generator.compiled_metrics.update_state(y_pose_data, out_pred[1:])
        # self.adaptor.compiled_metrics.update_state(y_adapt_data, out_adapton)

        out.update({"pose_mse": pred_loss, "adaption_mse": adapt_loss, "overall_loss": overall_Loss})
        # out.update({m.name: m.result() for m in self.adaptor.metrics})
        # out.update({m.name: m.result() for m in self.generator.metrics[1:]})
        return out

    def test_step(self, data):
        x_data = data[0]
        y_adapt_data = data[1][0]
        y_pose_data = data[1][1:]

        out_pred = self.generator(x_data, training=False)
        out_adapton = self.adaptor(out_pred[0], training=False)
        adapt_loss = self.d_loss_fn(y_adapt_data, out_adapton)
        out = {"adapt-acc": self.mymetrics[0][0](y_adapt_data, out_adapton)}
        pred_loss = 0

        # sum = np.array(self.loss_weight).sum(axis=0)

        for i in range(0, len(y_pose_data)):
            pred_loss += self.g_loss_fn[i](y_pose_data[i], out_pred[i + 1]) * self.loss_weight[i + 1]
            self.mymetrics[i + 1][0].update_state(y_pose_data[i], out_pred[i + 1])
            out.update({"s" + str(i + 1) + "-AnalogRecall": self.mymetrics[i + 1][0].result()})
        overall_Loss = pred_loss - self.loss_weight[0] * adapt_loss
        # overall_Loss = pred_loss
        # self.generator.compiled_metrics.update_state(y_pose_data, out_pred[1:])
        # self.adaptor.compiled_metrics.update_state(y_adapt_data, out_adapton)
        out.update({"pose_mse": pred_loss, "adaption_mse": adapt_loss, "overall_loss": overall_Loss})
        # out.update({m.name: m.result() for m in self.adaptor.metrics})
        # out.update({m.name: m.result() for m in self.generator.metrics[1:]})
        return out
