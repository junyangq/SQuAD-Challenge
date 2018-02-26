# Copyright 2018 Stanford University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This file contains some basic model components"""

import tensorflow as tf
from tensorflow.python.ops.rnn_cell import DropoutWrapper
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import rnn_cell


class RNNEncoder(object):
    """
    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)

    def build_graph(self, inputs, masks):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("RNNEncoder"):
            input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).
            (fw_out, bw_out), _ = tf.nn.bidirectional_dynamic_rnn(self.rnn_cell_fw, self.rnn_cell_bw, inputs, input_lens, dtype=tf.float32)

            # Concatenate the forward and backward hidden states
            out = tf.concat([fw_out, bw_out], 2)

            # Apply dropout
            out = tf.nn.dropout(out, self.keep_prob)

            return out



class DPDecoder(object):
    """
    TODO: MODIFY THE COMMENT!

    General-purpose module to encode a sequence using a RNN.
    It feeds the input through a RNN and returns all the hidden states.

    Note: In lecture 8, we talked about how you might use a RNN as an "encoder"
    to get a single, fixed size vector representation of a sequence
    (e.g. by taking element-wise max of hidden states).
    Here, we're using the RNN as an "encoder" but we're not taking max;
    we're just returning all the hidden states. The terminology "encoder"
    still applies because we're getting a different "encoding" of each
    position in the sequence, and we'll use the encodings downstream in the model.

    This code uses a bidirectional GRU, but you could experiment with other types of RNN.
    """

    def __init__(self, hidden_size, keep_prob, num_iterations,m,l,p):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          m: flag.context_length
          l: 2*flag.hidden_size=value_vec_size=400
          p: pool_size
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.hidden_size = hidden_size
        self.keep_prob = keep_prob
        self.num_iterations = num_iterations
        self.LSTM_dec = tf.nn.rnn_cell.LSTMCell(hidden_size)
        self.m = m
        
        # self.rnn_cell_fw = rnn_cell.GRUCell(self.hidden_size)
        # self.rnn_cell_fw = DropoutWrapper(self.rnn_cell_fw, input_keep_prob=self.keep_prob)
        # self.rnn_cell_bw = rnn_cell.GRUCell(self.hidden_size)
        # self.rnn_cell_bw = DropoutWrapper(self.rnn_cell_bw, input_keep_prob=self.keep_prob)
        
    
    def HMN(U, hi, us, ue, scope):
      '''
      Inputs:
        U: batch*2l*m=?*800*600
      '''
      with tf.variable_scope(scope):
        xavier=tf.contrib.layers.xavier_initializer()
        W1=tf.get_variable('W1',shape=[p,l,3*l],initializer=xavier,dtype=tf.float32)
        W2=tf.get_variable('W2',shape=[p,l,l],initializer=xavier,dtype=tf.float32)
        W3=tf.get_variable('W3',shape=[p,1,2*l],initializer=xavierx)
        WD=tf.get_variable('WD',shape=[l,5*l],initializer=xavier,dtype=tf.float32)
        b1=tf.get_variable('b1',shape=[1,p,l,1],initializer=tf.zeros_initializer())#totile
        #expand_b1
        b2=tf.get_variable('b2',shape=[1,p,l],initializer=tf.zeros_initializer(),dtype=tf.float32)#totile
        b3=tf.get_variable('b3',shape=[p],initializer=tf.zeros_initializer(),dtype=tf.float32)


      U = tf.transpose(U, perm=[0, 2, 1]) # U: (B * m * 2l)
      r=tf.tanh(tf.matmul(WD,tf.concat([hi, us, ue], axis = 1)))
      # r: (B * l)
      expand_r = tf.tile(tf.expand_dims(r, 2), [1, 1, self.m])
      # expand_r: (B * l * m)

      # check DIMS! tf broadcast or not?
      t1 = tf.matmul(tf.expand_dims(W1, 0), 
                    tf.expand_dims(tf.concat([U, expand_r], axis=1), 1)) + \
           + tf.reshape(b1, shape=[1, p, l, 1])
      # t1: (B * p * l * m)
      mt1 = tf.reduce_max(t1, axis=1)
      # mt1: (B * l * m)

      t2 = tf.matmul(tf.expand_dims(W2, 0), tf.expand_dims(mt1, 1)) + \
           + tf.reshape(b2, shape=[1, p, l, 1])
      # t2: (B * p * l * m)

      mt2 = tf.reduce_max(t2, axis=1)
      # mt2: (B * l * m)

      z_out = tf.matmul(tf.expand_dims(W3, 0),
                       tf.expand_dims(tf.concat([mt1, mt2], axis=1), 1)) + \
              + tf.reshape(b3, shape=[1, p, 1, 1])
      # z_out: (B * p * 1 * m)

      out = tf.reduce_max(z_out, axis=[1, 2])
      # out: (B * m)

      # expected mt1: (batch * p * l * m)
      # mt1: 


    

    def build_graph(self, U):
        """
        Inputs:
          inputs: Tensor shape (batch_size, seq_len, input_size)
          masks: Tensor shape (batch_size, seq_len).
            Has 1s where there is real input, 0s where there's padding.
            This is used to make sure tf.nn.bidirectional_dynamic_rnn doesn't iterate through masked steps.

        Returns:
          out: Tensor shape (batch_size, seq_len, hidden_size*2).
            This is all hidden states (fw and bw hidden states are concatenated).
        """
        with vs.variable_scope("DPDecoder"):

            # input_lens = tf.reduce_sum(masks, reduction_indices=1) # shape (batch_size)

            # Note: fw_out and bw_out are the hidden states for every timestep.
            # Each is shape (batch_size, seq_len, hidden_size).

            initial_state = tf.get_variable("initial_state", shape=(hidden_size), dtype=tf.float32)

            hidden = tf.reshape(tf.tile(initial_state, multiples=tf.shape(U)[0]), [-1, hidden_size])
            s = start_pos
            e = end_pos
            for _ in range(self.num_iterations):
                Us = U[:, s]
                Ue = U[:, e]
                hidden = self.LSTM_dec(tf.concat([Us, Ue], 1), hidden)
                alpha = self.HMN(U, hidden, Us, Ue, scope="start")
                beta = self.HMN(U, hidden, Us, Ue, scope="end")
                s = tf.argmax(alpha, axis=1) # s: B
                e = tf.argmax(beta, axis=1) # e: B

            # (fw_out, bw_out), _ = tf.nn.dynamic_rnn(self.lstm_cell, inputs, input_lens, dtype=tf.float32)

            # # Concatenate the forward and backward hidden states
            # out = tf.concat([fw_out, bw_out], 2)

            # # Apply dropout
            # out = tf.nn.dropout(out, self.keep_prob)

            return alpha, beta  #alpha: B * m, beta: B * m



class SimpleSoftmaxLayer(object):
    """
    Module to take set of hidden states, (e.g. one for each context location),
    and return probability distribution over those states.
    """

    def __init__(self):
        pass

    def build_graph(self, inputs, masks):
        """
        Applies one linear downprojection layer, then softmax.

        Inputs:
          inputs: Tensor shape (batch_size, seq_len, hidden_size)
          masks: Tensor shape (batch_size, seq_len)
            Has 1s where there is real input, 0s where there's padding.

        Outputs:
          logits: Tensor shape (batch_size, seq_len)
            logits is the result of the downprojection layer, but it has -1e30
            (i.e. very large negative number) in the padded locations
          prob_dist: Tensor shape (batch_size, seq_len)
            The result of taking softmax over logits.
            This should have 0 in the padded locations, and the rest should sum to 1.
        """
        with vs.variable_scope("SimpleSoftmaxLayer"):

            # Linear downprojection layer
            logits = tf.contrib.layers.fully_connected(inputs, num_outputs=1, activation_fn=None) # shape (batch_size, seq_len, 1)
            logits = tf.squeeze(logits, axis=[2]) # shape (batch_size, seq_len)

            # Take softmax over sequence
            masked_logits, prob_dist = masked_softmax(logits, masks, 1)

            return masked_logits, prob_dist





class BasicAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("BasicAttn"):

            # Calculate attention distribution
            values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            attn_logits = tf.matmul(keys, values_t) # shape (batch_size, num_keys, num_values)
            attn_logits_mask = tf.expand_dims(values_mask, 1) # shape (batch_size, 1, num_values)
            _, attn_dist = masked_softmax(attn_logits, attn_logits_mask, 2) # shape (batch_size, num_keys, num_values). take softmax over values

            # Use attention distribution to take weighted sum of values
            output = tf.matmul(attn_dist, values) # shape (batch_size, num_keys, value_vec_size)

            # Apply dropout
            output = tf.nn.dropout(output, self.keep_prob)

            return attn_dist, output



class CoAttn(object):
    """Module for basic attention.

    Note: in this module we use the terminology of "keys" and "values" (see lectures).
    In the terminology of "X attends to Y", "keys attend to values".

    In the baseline model, the keys are the context hidden states
    and the values are the question hidden states.

    We choose to use general terminology of keys and values in this module
    (rather than context and question) to avoid confusion if you reuse this
    module with other inputs.
    """

    def __init__(self, keep_prob, key_vec_size, value_vec_size):
        """
        Inputs:
          keep_prob: tensor containing a single scalar that is the keep probability (for dropout)
          key_vec_size: size of the key vectors. int
          value_vec_size: size of the value vectors. int
        """
        self.keep_prob = keep_prob
        self.key_vec_size = key_vec_size
        self.value_vec_size = value_vec_size

    def build_graph(self, values, values_mask, keys):
        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.

        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)

        Outputs:
          attn_dist: Tensor shape (batch_size, num_keys, num_values).
            For each key, the distribution should sum to 1,
            and should be 0 in the value locations that correspond to padding.
          output: Tensor shape (batch_size, num_keys, hidden_size).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
        """
        with vs.variable_scope("CoAttn"):

            print('value_vec_size is: ', self.value_vec_size)
            print('num_values size is: ', values.shape[1])
            print('num_keys size is: ', keys.shape[1])
            print('value_vec_size is (key):', keys.shape[2])
            # Declare variable 
            W = tf.get_variable("W", shape = (self.value_vec_size, self.value_vec_size), \
                initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape = (values.shape[1], self.value_vec_size), initializer = tf.constant_initializer(0))

            # Compute projected question hidden states

            values_t = tf.reshape(values, shape = [-1, self.value_vec_size]) # of shape (batch_size * num_values, value_vec_size)
            Q = tf.tanh(tf.reshape(tf.matmul(values_t, W), shape = [-1, values.shape[1], self.value_vec_size]) + tf.expand_dims(b, axis = 0)) #(batch_size, num_values, value_vec_size)
            #values_t = tf.transpose(values, perm=[0, 2, 1]) # (batch_size, value_vec_size, num_values)
            #Q = tf.tanh(tf.matmul(W, values_t) + b) # (batch_size, value_vec_size, num_values)

            print('Q shape is: ', Q.shape)
            Q = tf.transpose(Q, perm = [0, 2, 1]) #(batch_size, value_vec_size, num_values)
            print('Q shape is: ', Q.shape)
            D = keys # (batch_size, num_keys, value_vec_size)
            ### Start your code here to implement 'Sentinel Vector'


            ### End your code here to implement 'Sentinel Vector'

            # Compute affinity matrix L
            L = tf.matmul(D, Q) # shape (batch_size, num_keys, num_values)

            # Compute Context-to-Question (C2Q) Attention, we obtain C2Q attention outputs
            A_D = tf.nn.softmax(tf.transpose(L, perm = [0, 2, 1]), dim = -1) #(batch_size, num_values, num_keys)
            C2Q_Attn = tf.matmul(Q, A_D) # (batch_size, value_vec_size, num_keys)

            # Compute Question-to-Context (Q2C) Attention, we obtain Q2C attention outputs
            A_Q = tf.nn.softmax(L, dim = -1) # (batch_size, num_keys, num_values)
            Q2C_Attn = tf.matmul(tf.transpose(D, perm = [0, 2, 1]), A_Q) # (batch_size, value_vec_size, num_values)


            # Compute second-level attention outputs S
            S = tf.matmul(Q2C_Attn, A_D) # (batch_size, value_vec_size, num_keys)

            print('S size is: ', S.shape)

            # Concatenate C2Q_Attn and S:
            C_D = tf.transpose(tf.concat([C2Q_Attn, S], 1), perm = [0, 2, 1] ) # (batch_size, 2 * value_vec_size, num_keys)

            print('co_context size is: ', C_D.shape)




            # co_input = tf.concat([tf.transpose(D, perm = [0, 2, 1]), C_D], 1)
            # print('co_input size is: ', co_input.shape)


            size = int(self.value_vec_size / 2)
            (u_fw_out, u_bw_out), _ = tf.nn.bidirectional_dynamic_rnn(\
                tf.nn.rnn_cell.BasicLSTMCell(size),\
                  tf.nn.rnn_cell.BasicLSTMCell(size),\
                   C_D,\
                   dtype = tf.float32)
            print('u_fw_out shape is : ', u_fw_out.shape)
            print('u_bw_out shape is : ', u_bw_out.shape)

            U = tf.concat([u_fw_out, u_bw_out], 2)


            return U






def masked_softmax(logits, mask, dim):
    """
    Takes masked softmax over given dimension of logits.

    Inputs:
      logits: Numpy array. We want to take softmax over dimension dim.
      mask: Numpy array of same shape as logits.
        Has 1s where there's real data in logits, 0 where there's padding
      dim: int. dimension over which to take softmax

    Returns:
      masked_logits: Numpy array same shape as logits.
        This is the same as logits, but with 1e30 subtracted
        (i.e. very large negative number) in the padding locations.
      prob_dist: Numpy array same shape as logits.
        The result of taking softmax over masked_logits in given dimension.
        Should be 0 in padding locations.
        Should sum to 1 over given dimension.
    """
    exp_mask = (1 - tf.cast(mask, 'float')) * (-1e30) # -large where there's padding, 0 elsewhere
    masked_logits = tf.add(logits, exp_mask) # where there's padding, set logits to -large
    prob_dist = tf.nn.softmax(masked_logits, dim)
    return masked_logits, prob_dist
