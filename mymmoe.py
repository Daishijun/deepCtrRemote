import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.initializers import (Zeros, glorot_normal,
                                                  glorot_uniform)
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras.regularizers import l2
from tensorflow.python.layers import utils

class MMoE(Layer):
    def __init__(self, units, num_experts, num_tasks, expert_activation='relu', gate_activation='softmax', l2_reg=0, seed=1024, **kwargs):
        # Hidden nodes parameter
        self.units = units
        self.num_experts = num_experts
        self.num_tasks = num_tasks
        self.expert_activation = tf.keras.layers.Activation(expert_activation)
        self.gate_activation = tf.keras.layers.Activation(gate_activation)
        self.l2_reg = l2_reg
        self.seed = seed
        super(MMoE, self).__init__(**kwargs)

    def build(self, input_shape):

        if len(input_shape) != 2:
            raise ValueError(
                "Unexpected inputs dimensions %d, expect to be 2 dimensions" % (len(input_shape),))

        input_dimension = int(input_shape[-1])
        # Initialize expert weights (number of input features * number of units per expert * number of experts)
        self.expert_kernels = self.add_weight(
            name='expert_kernel',
            shape=(input_dimension, self.units, self.num_experts),
            initializer=glorot_normal(seed=self.seed),
            regularizer=l2(self.l2_reg),
            trainable=True
        )

        # Initialize expert bias (number of units per expert * number of experts)
        self.expert_bias = self.add_weight(
            name='expert_bias',
            shape=(self.units, self.num_experts),
            initializer=Zeros(),
            trainable=True
        )

        # Initialize gate weights (number of input features * number of experts * number of tasks)
        self.gate_kernels = [self.add_weight(
            name='gate_kernel_task_{}'.format(i),
            shape=(input_dimension, self.num_experts),
            initializer=glorot_normal(seed=self.seed),
            regularizer=l2(self.l2_reg),
            trainable=True
        ) for i in range(self.num_tasks)]

        # Initialize gate bias (number of experts * number of tasks)
        self.gate_bias = [self.add_weight(
            name='gate_bias_task_{}'.format(i),
            shape=(self.num_experts,),
            initializer=Zeros(),
            trainable=True
        ) for i in range(self.num_tasks)]
        # Be sure to call this somewhere!
        super(MMoE, self).build(input_shape)

    def call(self, inputs, **kwargs):
        gate_outputs = []
        final_outputs = []

        # f_{i}(x) = activation(W_{i} * x + b), where activation is ReLU according to the paper
        expert_outputs = tf.tensordot(a=inputs, b=self.expert_kernels, axes=1)
        # Add the bias term to the expert weights if necessary
        expert_outputs = K.bias_add(x=expert_outputs, bias=self.expert_bias)
        expert_outputs = self.expert_activation(expert_outputs)

        # g^{k}(x) = activation(W_{gk} * x + b), where activation is softmax according to the paper
        for index, gate_kernel in enumerate(self.gate_kernels):
            gate_output = K.dot(x=inputs, y=gate_kernel)
            # Add the bias term to the gate weights if necessary
            gate_output = K.bias_add(x=gate_output, bias=self.gate_bias[index])
            gate_output = self.gate_activation(gate_output)
            gate_outputs.append(gate_output)

        # f^{k}(x) = sum_{i=1}^{n}(g^{k}(x)_{i} * f_{i}(x))
        for gate_output in gate_outputs:
            expanded_gate_output = tf.expand_dims(gate_output, axis=1)
            weighted_expert_output = expert_outputs * K.repeat_elements(expanded_gate_output, self.units, axis=1)
            final_outputs.append(K.sum(weighted_expert_output, axis=2))
        return final_outputs

    def get_config(self, ):

        config = {'units': self.units,
                    'num_experts': self.num_experts,
                    'num_tasks': self.num_tasks,
                    'expert_activation': self.expert_activation,
                    'gate_activation': self.gate_activation,
                    'l2_reg': self.l2_reg,
                    'seed': self.seed}
        base_config = super(MMoE, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape