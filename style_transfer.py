import tensorflow as tf
import numpy as np
import collections

class StyleTransfer:

    def __init__(self, content_layer_ids, style_layer_ids, init_image, content_image,
                 style_image, session, net, num_iter, loss_ratio, content_loss_norm_type):

        self.net = net
        self.sess = session

        # sort layers info
        self.CONTENT_LAYERS = collections.OrderedDict(sorted(content_layer_ids.items()))
        self.STYLE_LAYERS = collections.OrderedDict(sorted(style_layer_ids.items()))

        # preprocess input images
        self.p0 = np.float32(self.net.preprocess(content_image))
        self.a0 = np.float32(self.net.preprocess(style_image))
        self.x0 = np.float32(self.net.preprocess(init_image))

        # parameters for optimization
        self.content_loss_norm_type = content_loss_norm_type
        self.num_iter = num_iter
        self.loss_ratio = loss_ratio

        # build graph for style transfer
        self._build_graph()

    def _build_graph(self):

        """ prepare data """
        # this is what must be trained
        self.x = tf.Variable(self.x0, trainable=True, dtype=tf.float32)

        # graph input
        self.p = tf.placeholder(tf.float32, shape=self.p0.shape, name='content')
        self.a = tf.placeholder(tf.float32, shape=self.a0.shape, name='style')

        # get content-layer-feature for content loss
        content_layers = self.net.feed_forward(self.p, scope='content')
        self.Ps = {}
        for id in self.CONTENT_LAYERS:
            self.Ps[id] = content_layers[id]

        # get style-layer-feature for style loss
        style_layers = self.net.feed_forward(self.a, scope='style')
        self.As = {}
        for id in self.STYLE_LAYERS:
            self.As[id] = self._gram_matrix(style_layers[id])
        
        # get layer-values for x
        self.Fs = self.net.feed_forward(self.x, scope='mixed')

        """ compute loss """
        L_content = 0
        L_style = 0
        for id in self.Fs:
            if id in self.CONTENT_LAYERS:
                ## content loss ##

                F = self.Fs[id]            # content feature of x
                P = self.Ps[id]            # content feature of p

                _, h, w, d = F.get_shape() # first return value is batch size (must be one)
                N = h.value*w.value        # product of width and height
                M = d.value                # number of filters

                w = self.CONTENT_LAYERS[id]# weight for this layer

                # You may choose different normalization constant
                if self.content_loss_norm_type==1:
                    L_content += w * tf.reduce_sum(tf.pow((F-P), 2)) / 2 # original paper
                elif self.content_loss_norm_type == 2:
                    L_content += w * tf.reduce_sum(tf.pow((F-P), 2)) / (N*M) #artistic style transfer for videos
                elif self.content_loss_norm_type == 3: # this is from https://github.com/cysmith/neural-style-tf/blob/master/neural_style.py
                    L_content += w * (1. / (2. * np.sqrt(M) * np.sqrt(N))) * tf.reduce_sum(tf.pow((F - P), 2))

            elif id in self.STYLE_LAYERS:
                ## style loss ##

                F = self.Fs[id]

                _, h, w, d = F.get_shape()  # first return value is batch size (must be one)
                N = h.value * w.value       # product of width and height
                M = d.value                 # number of filters

                w = self.STYLE_LAYERS[id]   # weight for this layer

                G = self._gram_matrix(F)    # style feature of x
                A = self.As[id]             # style feature of a

                L_style += w * (1. / (4 * N ** 2 * M ** 2)) * tf.reduce_sum(tf.pow((G-A), 2))


        # fix beta as 1
        alpha = self.loss_ratio
        beta = 1

        self.L_content = L_content
        self.L_style = L_style
        self.L_total = alpha*L_content + beta*L_style

    def update(self):
        """ define optimizer L-BFGS """
        # this call back function is called every after loss is updated
        global _iter
        _iter = 0
        def callback(tl, cl, sl):
            global _iter
            print('iter : %4d, ' % _iter, 'L_total : %g, L_content : %g, L_style : %g' % (tl, cl, sl))
            _iter += 1

        optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.L_total, method='L-BFGS-B', options={'maxiter': self.num_iter})

        """ session run """
        # initialize variables
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # optmization
        optimizer.minimize(self.sess,feed_dict={self.a:self.a0, self.p:self.p0},
                           fetches=[self.L_total, self.L_content, self.L_style], loss_callback=callback)

        """ get final result """
        final_image = self.sess.run(self.x)

        # ensure the image has valid pixel-values between 0 and 255
        final_image = np.clip(self.net.undo_preprocess(final_image), 0.0, 255.0)

        return final_image

    def _gram_matrix(self, tensor):

        shape = tensor.get_shape()

        # Get the number of feature channels for the input tensor,
        # which is assumed to be from a convolutional layer with 4-dim.
        num_channels = int(shape[3])

        # Reshape the tensor so it is a 2-dim matrix. This essentially
        # flattens the contents of each feature-channel.
        matrix = tf.reshape(tensor, shape=[-1, num_channels])

        # Calculate the Gram-matrix as the matrix-product of
        # the 2-dim matrix with itself. This calculates the
        # dot-products of all combinations of the feature-channels.
        gram = tf.matmul(tf.transpose(matrix), matrix)

        return gram










