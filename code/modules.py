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
from tensorflow.contrib.cudnn_rnn.python.ops import cudnn_rnn_ops
from tensorflow.python.client import device_lib


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
    Dynamic Pointing Decoder with Highway Maxout Network (HMN)

    """

    def __init__(self, keep_prob, num_iterations, context_len, hidden_size, pool_size):
        """
        Inputs:
          hidden_size: int. Hidden size of the RNN
          m: flag.context_length
          l: 2*flag.hidden_size=value_vec_size=400
          p: pool_size
          keep_prob: Tensor containing a single scalar that is the keep probability (for dropout)
        """
        self.keep_prob = keep_prob
        self.num_iterations = num_iterations
        self.context_len = context_len
        self.hidden_size = hidden_size
        self.pool_size = pool_size
        local_device_protos = device_lib.list_local_devices()
        '''
	if len([x for x in local_device_protos if x.device_type == 'GPU']) > 0:
            # Only NVidia GPU is supported for now
            self.device = 'gpu'
            self.LSTM_dec = tf.contrib.cudnn_rnn.CudnnCompatibleLSTMCell(self.hidden_size)
        else:
	'''
        self.device = 'gpu'
        self.LSTM_dec = tf.nn.rnn_cell.BasicLSTMCell(self.hidden_size)
	self.LSTM_dec = DropoutWrapper(self.LSTM_dec, input_keep_prob=self.keep_prob)
        
    
    def HMN(self, U, hi, us, ue, mask, scope):
      '''
      Inputs:
        U: batch * m * 2l
      '''
      with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        xavier = tf.contrib.layers.xavier_initializer()
        W11 = tf.get_variable('W11',shape=[self.pool_size, self.hidden_size, 2*self.hidden_size], initializer=xavier, dtype=tf.float32)
        W12 = tf.get_variable('W12',shape=[self.pool_size, self.hidden_size, self.hidden_size], initializer=xavier, dtype=tf.float32)
        W2 = tf.get_variable('W2',shape=[self.pool_size, self.hidden_size, self.hidden_size], initializer=xavier, dtype=tf.float32)
        W3 = tf.get_variable('W3',shape=[self.pool_size, 1, 2*self.hidden_size], initializer=xavier, dtype=tf.float32)
        WD = tf.get_variable('WD',shape=[self.hidden_size, 5*self.hidden_size], initializer=xavier, dtype=tf.float32)
        b1 = tf.get_variable('b1',shape=[self.pool_size, self.hidden_size], initializer=tf.zeros_initializer(), dtype=tf.float32)
        b2 = tf.get_variable('b2',shape=[self.pool_size, self.hidden_size],initializer=tf.zeros_initializer(),dtype=tf.float32)
        b3 = tf.get_variable('b3',shape=[self.pool_size], initializer=tf.zeros_initializer(), dtype=tf.float32)

        concat_h_us_ue = tf.concat([hi, us, ue], axis=1)
	concat_h_us_ue = tf.nn.dropout(concat_h_us_ue, self.keep_prob)
        r = tf.tanh(tf.tensordot(concat_h_us_ue, WD, [[1],[1]]))  # r: (B * l)

        Z11 = tf.tensordot(U, W11, [[2],[2]])  # Z11: (B * m * p * l)

        Z12 = tf.tensordot(r, W12, [[1],[2]])  # Z12: (B * p * l)

        Z1 = Z11 + tf.expand_dims(Z12, 1) + b1
	#Z1 = tf.nn.dropout(Z1, self.keep_prob)

        mt1 = tf.reduce_max(Z1, axis=2)  # mt1: (B * m * l)

        Z2 = tf.tensordot(mt1, W2, [[2],[2]]) + b2  # Z2: (B * m * p * l)
	#Z2 = tf.nn.dropout(Z2, self.keep_prob)

        mt2 = tf.reduce_max(Z2, axis=2)  # mt2: (B * m * l)

        concat_mt1_mt2 = tf.concat([mt1, mt2], axis=2)
        Z3 = tf.squeeze(tf.tensordot(concat_mt1_mt2, W3, [[2],[2]]), 3) + b3 # Z3: (B * m * p)
        #Z3 = tf.nn.dropout(Z3, self.keep_prob)
        logits = tf.reduce_max(Z3, 2)  # out: (B * m)

      return masked_softmax(logits, mask, 1)

    

    def build_graph(self, U, context_mask):
        """
        Inputs:
          U: Tensor shape (batch_size, context_len, 2 * hidden_size). Vector representation of context words
          context_mask: Tensor shape (batch_size, context_len). 1s where there's real input, 0s where there's padding

        Returns:
          out:
            alpha: Tensor shape (batch_size, context_len). Logits of start position for each word
            beta: Tensor shape (batch_size, context_len). Logits of end position for each word
        """
        with vs.variable_scope("DPDecoder", reuse=tf.AUTO_REUSE):

            h_state = self.LSTM_dec.zero_state(tf.shape(U)[0], dtype=tf.float32)

            # s = start_pos
            s = tf.zeros(shape=[tf.shape(U)[0]], dtype=tf.int32)  # TODO: random init
            # e = end_pos
            e = tf.zeros(shape=[tf.shape(U)[0]], dtype=tf.int32)  # TODO: random init
            alphas = [None] * self.num_iterations
            betas = [None] * self.num_iterations
            for i in range(self.num_iterations):
                idx = tf.range(0, tf.shape(U)[0], dtype=tf.int32)
                s_stk = tf.stack([idx, s], axis=1)
                e_stk = tf.stack([idx, e], axis=1)
                Us = tf.gather_nd(U, s_stk)
                Ue = tf.gather_nd(U, e_stk)
                _, h_state = self.LSTM_dec(tf.concat([Us, Ue], axis=1), h_state)
                hidden = h_state[0]
                alpha, prob_start = self.HMN(U, hidden, Us, Ue, context_mask, scope="start")
                beta, prob_end = self.HMN(U, hidden, Us, Ue, context_mask, scope="end")
                alphas[i] = alpha
                betas[i] = beta

                s = tf.argmax(alpha, axis=1, output_type=tf.int32) # s: (B)
                e = tf.argmax(beta, axis=1, output_type=tf.int32) # e: (B)

            return alphas, betas, prob_start, prob_end  # alpha, beta, prob_start, prob_end: (B * m)



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
        local_device_protos = device_lib.list_local_devices()
        if len([x for x in local_device_protos if x.device_type == 'GPU']) > 0:
            # Only NVidia GPU is supported for now
            self.device = 'gpu'
        else:
            self.device = 'cpu'


    def build_graph(self, values, values_mask, keys_mask, keys, use_mask=True):

        """
        Keys attend to values.
        For each key, return an attention distribution and an attention output vector.
        Inputs:
          values: Tensor shape (batch_size, num_values, value_vec_size).
          values_mask: Tensor shape (batch_size, num_values).
            1s where there's real input, 0s where there's padding
          keys: Tensor shape (batch_size, num_keys, value_vec_size)
        Outputs:
          U: Tensor shape (batch_size, num_keys, hidden_size*4).
            This is the attention output; the weighted sum of the values
            (using the attention distribution as weights).
	  A_D: C2Q attn dist
	  A_Q: Q2C attn dist
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

            Q = tf.tanh(tf.tensordot(values, W, 1) + tf.expand_dims(b, axis=0)) # (batch_size, num_values, value_vec_size)
	    Q = tf.nn.dropout(Q, self.keep_prob)
            print('Q shape is: ', Q.shape)

            Q = concat_sentinel('question_sentinel', Q, self.value_vec_size)  # (batch_size, num_values, value_vec_size)

            # sentinel = tf.get_variable(name='question_sentinel', shape=tf.shape(Q)[2], \
            #     initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
            # sentinel = tf.tile(sentinel, [tf.shape(original_tensor)[0], 1, 1])
            # concat_tensor = tf.concat([original_tensor, sentinel], 2)

            print('Q shape is: ', Q.shape)
            D = keys # (batch_size, num_keys, value_vec_size)
	    D = tf.nn.dropout(D, self.keep_prob)
            D = concat_sentinel('document_sentinel', D, self.value_vec_size)

            # key = document, value = question here
            ### End your code here to implement 'Sentinel Vector'
            # Compute affinity matrix L
            L = tf.matmul(D, tf.transpose(Q, perm=[0, 2, 1])) # shape (batch_size, num_keys, num_values)

            # Compute Context-to-Question (C2Q) Attention, we obtain C2Q attention outputs
            if use_mask:
                print('tf.shape(values)[0] is: ', tf.shape(values)[0])
                print('tf.ones([tf.shape(values)[0], 1] is ', tf.ones([tf.shape(values)[0], 1], dtype=tf.int32))
                values_mask = tf.expand_dims(tf.concat([values_mask, tf.ones([tf.shape(values)[0], 1], dtype=tf.int32)], axis=1), 1)
                print "value_mask shape:", values_mask.shape
                print "L shape:", L.shape
                _, A_D = masked_softmax(L, mask=values_mask, dim=2) #(batch_size, num_keys, num_values)
            else:
                A_D = tf.nn.softmax(L, dim=-1)

            C2Q_Attn = tf.matmul(A_D, Q) # (batch_size, num_keys, value_vec_size)
            print('C2Q_Attn shapeis ', C2Q_Attn.shape)

            # Compute Question-to-Context (Q2C) Attention, we obtain Q2C attention outputs
            if use_mask:
                keys_mask = tf.expand_dims(tf.concat([keys_mask, tf.ones([tf.shape(keys)[0], 1], dtype=tf.int32)], axis=1), 1)
                print "key_mask shape:", keys_mask.shape
                print "L shape:", L.shape
                _, A_Q = masked_softmax(tf.transpose(L, perm=[0, 2, 1]), mask=keys_mask, dim=-1) # (batch_size, num_values, num_keys)
            else:
                A_Q = tf.nn.softmax(tf.transpose(L, perm=[0, 2, 1]), dim=2)

            Q2C_Attn = tf.matmul(A_Q, D) # (batch_size, num_values, key_vec_size)
            print('Q2C_Attn shapeis ', Q2C_Attn.shape)

            # Compute second-level attention outputs S
            S = tf.matmul(A_D, Q2C_Attn) # (batch_size, num_keys, value_vec_size)
            print('S size is: ', S.shape)

            # Concatenate C2Q_Attn and S:
            C_D = tf.concat([C2Q_Attn, S], 2)  # (batch_size, num_keys, 2 * value_vec_size)
            C_D = tf.nn.dropout(C_D, self.keep_prob)
            print('co_context size is: ', C_D.shape)

            # co_input = tf.concat([tf.transpose(D, perm = [0, 2, 1]), C_D], 1)
            # print('co_input size is: ', co_input.shape)
            size = int(self.value_vec_size)
            
            if self.device == 'gpu':
                bidirection_rnn = tf.contrib.cudnn_rnn.CudnnLSTM(1, size, 3*size, dropout=0.2, direction=cudnn_rnn_ops.CUDNN_RNN_BIDIRECTION, dtype=tf.float32)
                C_D = tf.transpose(C_D, perm=[1, 0, 2])
                print 'C_D shape', C_D.shape
                input_h = tf.zeros([2, tf.shape(values)[0], size])
                input_c = tf.zeros([2, tf.shape(values)[0], size])
                params = tf.get_variable("RNN", shape=(estimate_cudnn_parameter_size(2*self.value_vec_size, size, 2)),
                    initializer=tf.contrib.layers.xavier_initializer(), dtype=tf.float32)
                
                U, _, _ = bidirection_rnn(C_D, input_h, input_c, params)

                print 'U shape:', U.shape
                U = tf.transpose(U, perm=[1, 0, 2])

            else:
                (u_fw_out, u_bw_out), _ = tf.nn.bidirectional_dynamic_rnn(
                    cell_fw=DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size),input_keep_prob=self.keep_prob), cell_bw=DropoutWrapper(tf.nn.rnn_cell.BasicLSTMCell(size),input_keep_prob=self.keep_prob), 
                    inputs=C_D, dtype = tf.float32)
                U = tf.concat([u_fw_out, u_bw_out], 2)

            print 'U shape:', U.shape
            U = U[:,:-1, :]
            U = tf.nn.dropout(U, self.keep_prob)
            print('U shape is: ', U.shape)
            
        return U,A_D,A_Q


class DCNplusEncoder(object):
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
        local_device_protos = device_lib.list_local_devices()
        if len([x for x in local_device_protos if x.device_type == 'GPU']) > 0:
            # Only NVidia GPU is supported for now
            self.device = 'gpu'
        else:
            self.device = 'cpu'

    def build_graph(self, values, values_mask, keys_mask, keys, use_mask=True, sentinel=True):

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

        with vs.variable_scope("encoder_initialization"):

            print('value_vec_size is: ', self.value_vec_size)
            print('num_values size is: ', values.shape[1])
            print('num_keys size is: ', keys.shape[1])
            print('value_vec_size is (key):', keys.shape[2])
            # Declare variable 
            # Compute projected question hidden states
            W = tf.get_variable("W", shape = (self.value_vec_size, self.value_vec_size), \
                initializer = tf.contrib.layers.xavier_initializer())
            b = tf.get_variable("b", shape = (values.shape[1], self.value_vec_size), initializer = tf.constant_initializer(0))
            Q = tf.tanh(tf.tensordot(values, W, 1) + tf.expand_dims(b, axis=0)) # (batch_size, num_values, value_vec_size)
            D = keys # (batch_size, num_keys, value_vec_size)
            Q_length = values.shape[1]
            D_length = keys.shape[1]
            if sentinel:
                Q = concat_sentinel('question_sentinel', Q, self.value_vec_size)  # (batch_size, num_values, value_vec_size)
                D = concat_sentinel('document_sentinel', D, self.value_vec_size)
                Q_length += 1
                D_length += 1

        with vs.variable_scope("coattention_layer_1"):
            S_D_1, S_Q_1, C_D_1 = coattention(\
                Q, Q_length, D, D_length, values_mask, keys_mask, use_mask)

        with vs.variable_scope('encode_summaries_from_coattention_layer_1'):

            print('Q Length is: ', Q_length)
            print('D length is: ', D_length)

            size = int(self.value_vec_size)
            cell = tf.nn.rnn_cell.BasicLSTMCell(size)
	    cell = DropoutWrapper(cell, input_keep_prob=self.keep_prob)
            Q_fw_bw_encodings, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = cell,
                cell_bw = cell,
                dtype = tf.float32,
                inputs = S_Q_1,
    #            sequence_length = Q_length
            )
            E_Q_2 = tf.concat(Q_fw_bw_encodings, 2)

            D_fw_bw_encodings, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = cell,
                cell_bw = cell,
                dtype = tf.float32,
                inputs = S_D_1,
     #           sequence_length = D_length
            )    
            E_D_2 = tf.concat(D_fw_bw_encodings, 2)

        with vs.variable_scope('coattention_layer_2'):
            S_D_2, S_Q_2, C_D_2 = coattention(\
                E_Q_2, Q_length, E_D_2, D_length, values_mask, keys_mask, use_mask)
 

        with vs.variable_scope('final_encoder'):
            document_representations = tf.concat(\
                [D, E_D_2, S_D_1, S_D_2, C_D_1, C_D_2], 2)#(N, D, 2H)

            size = int(self.value_vec_size)
            cell = tf.nn.rnn_cell.BasicLSTMCell(size)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                cell_fw = cell,
                cell_bw = cell,
                dtype = tf.float32,
                inputs = document_representations,
  #              sequence_length = D_length,
            )
            encoding = tf.concat(outputs, 2)
            encoding = encoding[:, :-1, :]
            return encoding


def concat_sentinel(sentinel_name, original_tensor, size):
    '''

    Args:  
        sentinel_name: Variable name of sentinel.  # string
        original_tensor: Tensor of rank 3 to left concatenate sentinel to. # of shape (batch_size, num_values, value_vec_size)
    Returns:  
        original_tensor with sentinel. 

    '''

    sentinel = tf.get_variable(name=sentinel_name, shape=(size),
        initializer=tf.contrib.layers.xavier_initializer(), dtype = tf.float32)
    sentinel = tf.tile(tf.reshape(sentinel, [1, 1, -1]), [tf.shape(original_tensor)[0], 1, 1])
    concat_tensor = tf.concat([original_tensor, sentinel], 1)
    print('the shape of concat tensor is: ', concat_tensor.get_shape())
    return concat_tensor


def coattention(Q, Q_length, D, D_length, Q_mask, D_mask, use_mask=True):
    """ DCN+ Coattention layer.

    Args:  
        Q: Tensor of rank 3, shape [N, Q, 2H].  
        Q_length: Tensor of rank 1, shape [N]. Lengths of questions without sentinel.  
        D: Tensor of rank 3, shape [N, D, 2H].   
        D_length: Tensor of rank 1, shape [N]. Lengths of documents without sentinel. 
        Q_mask:
        D_mask:
        sentinel: Scalar boolean. If True, then sentinel vectors are temporarily left concatenated
        use_mask: Scalar boolean. If True, then use mask 
        to the query's and document's second dimension, letting the attention focus on nothing.  
    Returns:  
        A tuple containing:  
            summary matrix of the question S_Q, shape [N, Q, 2H].  
            summary matrix of the document S_D, shape [N, D, 2H].  
            coattention matrix of the document and query in document space C_D, shape [N, D, 2H].
    """

    L = tf.matmul(D, tf.transpose(Q, perm=[0, 2, 1])) # shape (batch_size, num_keys, num_values)
    print('L shape is :', L.shape)
    # Compute Context-to-Question (C2Q) Attention, we obtain C2Q attention outputs
    if use_mask:
        print('Q_mask dimension is: ', Q_mask.shape)

        Q_mask = tf.expand_dims(tf.concat([Q_mask, tf.ones([tf.shape(Q)[0], 1], dtype=tf.int32)], axis=1), 1)


        print('Q_mask dimension is: ', Q_mask.shape)
        _, A_D = masked_softmax(L, mask=Q_mask, dim=2) #(N, D, Q)
        print('A_D shape after mask is, ', A_D.shape)
    else:
        A_D = tf.nn.softmax(L, dim=-1)
    S_D = tf.matmul(A_D, Q) # (N, D, 2H)
    print('***S_D shape is ', S_D.shape)
    # Compute Question-to-Context (Q2C) Attention, we obtain Q2C attention outputs
    if use_mask:
        D_mask = tf.expand_dims(tf.concat([D_mask, tf.ones([tf.shape(D)[0], 1], dtype=tf.int32)], axis=1), 1)
        _, A_Q = masked_softmax(tf.transpose(L, perm=[0, 2, 1]), mask=D_mask, dim=-1) # (batch_size, num_values, num_keys)
    else:
        A_Q = tf.nn.softmax(tf.transpose(L, perm=[0, 2, 1]), dim=2)
    S_Q = tf.matmul(A_Q, D) # (batch_size, num_values, key_vec_size)
    print('***S_Q shape is ', S_Q.shape)
    # Compute second-level attention outputs S
    C_D = tf.matmul(A_D, S_Q) # (batch_size, num_keys, value_vec_size)
    print('***C_D size is: ', C_D.shape)

    return S_D, S_Q, C_D


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


def estimate_cudnn_parameter_size(input_size, hidden_size, direction):
    """
    Compute the number of parameters needed to
    construct a stack of LSTMs. 
    """
    single_rnn_size = 8 * hidden_size + 4 * (hidden_size * input_size) + 4 * (hidden_size * hidden_size)
    return 2 * single_rnn_size
