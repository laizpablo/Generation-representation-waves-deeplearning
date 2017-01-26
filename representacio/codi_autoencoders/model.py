import tensorflow as tf


# Class AutoEncoderModel. Define tha arquitecture
class AutoEncoderModel(object):

    def __init__(self, batch_size, histograms):

        
        self.batch_size = batch_size
        self.histograms = histograms
        self.n_input = 1024
        self.n_hidden_1 = 900  # 1st layer num features
        self.n_hidden_2 = 700  # 2nd layer num features
        self.n_hidden_3 = 512   # 3rd layer num features
        self.n_hidden_4 = 700  # 4th layer num features
        self.n_hidden_5 = 900  # 5th layer num features
        self.keep_prob = None
        self.x = None
        self.y = None
        self.weights, self.biases = self._create_variables()


    # Method that create all the necesary variables for the model
    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        with tf.name_scope("AutoEncoder_variables"):
            
            self.x = tf.placeholder("float", [None, self.n_input], name='x')
            self.y = tf.placeholder("float", [None, self.n_input], name='y')
            self.keep_prob = tf.placeholder(tf.float32) #Input parameter: dropout probability

            # Store layers weight & bias
            c = 0.1 
            initializer = tf.contrib.layers.xavier_initializer_conv2d()

            

            with tf.name_scope("weights"):
                weights = {
                    'h1': tf.Variable(initializer(shape=[self.n_input, self.n_hidden_1]), name='W1'),
                    'h2': tf.Variable(initializer(shape=[self.n_hidden_1, self.n_hidden_2]), name='W2'),
                    'h3': tf.Variable(initializer(shape=[self.n_hidden_2, self.n_hidden_3]), name='W3'),
                    'h4': tf.Variable(initializer(shape=[self.n_hidden_3, self.n_hidden_4]), name='W4'),
                    'h5': tf.Variable(initializer(shape=[self.n_hidden_4, self.n_hidden_5]), name='W5'),
                    'out': tf.Variable(initializer(shape=[self.n_hidden_5, self.n_input]))
            }
            with tf.name_scope("biases"):
                biases = {
                    'b1': tf.Variable(initializer(shape=[self.n_hidden_1]), name='b1'),
                    'b2': tf.Variable(initializer(shape=[self.n_hidden_2]), name='b2'),
                    'b3': tf.Variable(initializer(shape=[self.n_hidden_3]), name='b3'),
                    'b4': tf.Variable(initializer(shape=[self.n_hidden_4]), name='b4'),
                    'b5': tf.Variable(initializer(shape=[self.n_hidden_5]), name='b5'),
                    'out': tf.Variable(initializer(shape=[self.n_input]))
                }

        return weights, biases


    
    # Construct the network (Graph).
    # Output:
    #   - graph 
    def _create_network(self):

        with tf.name_scope("AutoEncoder"):
            with tf.name_scope("dropout"):
                pre_layer_drop = tf.nn.dropout(self.x, self.keep_prob)
            with tf.name_scope("layer_1"):
                layer_1 = tf.nn.relu(tf.add(tf.matmul(pre_layer_drop, self.weights['h1']), self.biases['b1']))
            with tf.name_scope("layer_2"):
                layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))
            with tf.name_scope("layer_3"):
                layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['b3']))
            with tf.name_scope("layer_4"):
                layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, self.weights['h4']), self.biases['b4']))
            with tf.name_scope("layer_5"):
                layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, self.weights['h5']), self.biases['b5']))
            with tf.name_scope("output"):
                self.output = tf.add(tf.matmul(layer_5, self.weights['out']), self.biases['out'])


        if self.histograms:
            with tf.device('/cpu:0'):
                tf.histogram_summary('pre_layer_drop',pre_layer_drop)
                tf.histogram_summary('layer_1',layer_1)
                tf.histogram_summary('layer_2',layer_2)
                tf.histogram_summary('layer_3',layer_3)
                tf.histogram_summary('layer_4',layer_4)
                tf.histogram_summary('layer_5',layer_5)
                tf.histogram_summary('output',self.output)

    
    # Creates a network and returns the autoencoding loss.
    # The variables are all scoped to the given name.
    def loss_snr(self, name='Autoencoder'):
            
        epsilon = 1e-2
        # Creation neural graph
        self._create_network()

        with tf.name_scope('loss'):
            # SNR ERROR with not dB
            signal = tf.pow(self.y, 2)
            noise = tf.pow(self.output - self.y, 2)
            loss = signal/(noise + epsilon)
            
            reduced_loss = tf.reduce_mean(1/(1+loss))

            with tf.device('/cpu:0'):
                tf.scalar_summary('loss', reduced_loss)

        return reduced_loss
    
    
    def loss_squared_error(self, name='Autoencoder_squared_error'):
            
        epsilon = 1e-2
        # Creation neural graph
        self._create_network()

        with tf.name_scope('loss'):
            # Quadratic ERROR with not dB
            loss = tf.pow(self.output - self.y, 2)
            reduced_loss = tf.reduce_mean(loss)


            with tf.device('/cpu:0'):
                tf.scalar_summary('loss', reduced_loss)

        return reduced_loss


# Class AutoEncoderSpectrogram. Define tha arquitecture
class AutoEncoderSpectrogram(object):

    def __init__(self, batch_size, histograms):

        
        self.batch_size = batch_size
        self.histograms = histograms
        self.n_input = 903
        self.n_hidden_1 = 900  # 1st layer num features
        self.n_hidden_2 = 700  # 2nd layer num features
        self.n_hidden_3 = 512   # 3rd layer num features
        self.n_hidden_4 = 700  # 4th layer num features
        self.n_hidden_5 = 900  # 5th layer num features
        self.keep_prob = None
        self.x = None
        self.y = None
        self.weights, self.biases = self._create_variables()


    # Method that create all the necesary variables for the model
    def _create_variables(self):
        '''This function creates all variables used by the network.
        This allows us to share them between multiple calls to the loss
        function and generation function.'''

        with tf.name_scope("AutoEncoder_variables"):
            
            self.x = tf.placeholder("float", [None, self.n_input], name='x')
            self.y = tf.placeholder("float", [None, self.n_input], name='y')
            self.keep_prob = tf.placeholder(tf.float32) #Input parameter: dropout probability

            # Store layers weight & bias
            c = 0.1 
            initializer = tf.contrib.layers.xavier_initializer_conv2d()

            with tf.name_scope("weights"):
                weights = {
                    'h1': tf.Variable(initializer(shape=[self.n_input, self.n_hidden_1]), name='W1'),
                    'h2': tf.Variable(initializer(shape=[self.n_hidden_1, self.n_hidden_2]), name='W2'),
                    'h3': tf.Variable(initializer(shape=[self.n_hidden_2, self.n_hidden_3]), name='W3'),
                    'h4': tf.Variable(initializer(shape=[self.n_hidden_3, self.n_hidden_4]), name='W4'),
                    'h5': tf.Variable(initializer(shape=[self.n_hidden_4, self.n_hidden_5]), name='W5'),
                    'out': tf.Variable(initializer(shape=[self.n_hidden_5, self.n_input]))
            }
            with tf.name_scope("biases"):
                biases = {
                    'b1': tf.Variable(initializer(shape=[self.n_hidden_1]), name='b1'),
                    'b2': tf.Variable(initializer(shape=[self.n_hidden_2]), name='b2'),
                    'b3': tf.Variable(initializer(shape=[self.n_hidden_3]), name='b3'),
                    'b4': tf.Variable(initializer(shape=[self.n_hidden_4]), name='b4'),
                    'b5': tf.Variable(initializer(shape=[self.n_hidden_5]), name='b5'),
                    'out': tf.Variable(initializer(shape=[self.n_input]))
                }

        return weights, biases


    # Construct the network (Graph).
    # Output:
    #   - graph 
    def _create_network(self):

        with tf.name_scope("AutoEncoder"):
            with tf.name_scope("dropout"):
                pre_layer_drop = tf.nn.dropout(self.x, self.keep_prob)
            with tf.name_scope("layer_1"):
                layer_1 = tf.nn.relu(tf.add(tf.matmul(pre_layer_drop, self.weights['h1']), self.biases['b1']))
            with tf.name_scope("layer_2"):
                layer_2 = tf.nn.relu(tf.add(tf.matmul(layer_1, self.weights['h2']), self.biases['b2']))
            with tf.name_scope("layer_3"):
                layer_3 = tf.nn.relu(tf.add(tf.matmul(layer_2, self.weights['h3']), self.biases['b3']))
            with tf.name_scope("layer_4"):
                layer_4 = tf.nn.relu(tf.add(tf.matmul(layer_3, self.weights['h4']), self.biases['b4']))
            with tf.name_scope("layer_5"):
                layer_5 = tf.nn.relu(tf.add(tf.matmul(layer_4, self.weights['h5']), self.biases['b5']))
            with tf.name_scope("output"):
                self.output = tf.add(tf.matmul(layer_5, self.weights['out']), self.biases['out'])

        if self.histograms:
            with tf.device('/cpu:0'):
                tf.histogram_summary('pre_layer_drop',pre_layer_drop)
                tf.histogram_summary('layer_1',layer_1)
                tf.histogram_summary('layer_2',layer_2)
                tf.histogram_summary('layer_3',layer_3)
                tf.histogram_summary('layer_4',layer_4)
                tf.histogram_summary('layer_5',layer_5)
                tf.histogram_summary('output',self.output)

    
    # Creates a network and returns the autoencoding loss.
    # The variables are all scoped to the given name.
    def loss_snr(self, name='Autoencoder'):
            
        epsilon = 1e-2
        # Creation neural graph
        self._create_network()

        with tf.name_scope('loss'):
            # SNR ERROR with not dB
            signal = tf.pow(self.y, 2)
            noise = tf.pow(self.output - self.y, 2)
            loss = signal/(noise + epsilon)
            
            reduced_loss = tf.reduce_mean(1/(1+loss))

            with tf.device('/cpu:0'):
                tf.scalar_summary('loss', reduced_loss)

        return reduced_loss
    
    
    def loss_squared_error(self, name='Autoencoder_squared_error'):
            
        epsilon = 1e-2
        # Creation neural graph
        self._create_network()

        with tf.name_scope('loss'):
            # Quadratic ERROR with not dB
            loss = tf.pow(self.output - self.y, 2)
            reduced_loss = tf.reduce_mean(loss)


            with tf.device('/cpu:0'):
                tf.scalar_summary('loss', reduced_loss)

        return reduced_loss

