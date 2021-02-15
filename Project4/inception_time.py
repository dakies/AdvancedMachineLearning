import keras


class InceptionModule(keras.layers.Layer):
    def __init__(self, num_filters=32, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.num_filters = num_filters
        self.activation = keras.activations.get(activation)

    def _default_Conv1D(selfself, filters, kernel_size):
        return keras.layers.Conv1D(filters=filters,
                                   kernel_size=kernel_size,
                                   strides=1,
                                   activation='relu',
                                   use_bias=False)

    def call(self, inputs, **kwargs):
        # Step 1
        Z_bottleneck = self._default_Conv1D(filters=self.num_filters, kernel_size=1)(inputs)
        Z_maxpool = keras.layers.MaxPool1D(pool_size=3, strides=1, padding='same')(inputs)

        # Step 2
        Z1= self._default_Conv1D(filters=self.num_filters, kernel_size=10)(inputs)
        Z2 = self._default_Conv1D(filters=self.num_filters, kernel_size=20)(inputs)
        Z3 = self._default_Conv1D(filters=self.num_filters, kernel_size=40)(inputs)
        Z4 = self._default_Conv1D(filters=self.num_filters, kernel_size=1)(inputs)

        # Step 3
        Z = keras.layers.Concatenate(axis=2)([Z1, Z2, Z3, Z4])
        Z = keras.layers.BatchNormalization()(Z)

        return self.activation(Z)


def shortcut_layer(inputs, Z_incpetion):
    # create shortcut connection
    Z_shortcut = keras.layers.Conv1D(filters=int(out_tensor.shape[-1]), kernel_size=1,
                                     padding='same', use_bias='False')(inputs)
    Z_shortcut = keras.layers.BatchNormalization()(Z_shortcut)

    # add shortcut to model
    Z = keras.layers.add([Z_shortcut, Z_incpetion])

    return keras.layers. Activation('relu')(Z)

def build_model(input_shape, num_classes, num_modules=6):
    # create series of inception modules with shortcut
    input_layer = keras.layers.Input(input_shape)
    Z = input_layer
    Z_residual = input_layer

    for i in range(num_modules):
        Z = InceptionModule()(Z)
        if i % 3 == 2:
            Z = shortcut_layer(Z_residual, Z)
            Z_residual = Z

    gap_layer = keras.layers.GlobalAveragePooling1D()(Z)
    output_layer = keras.layers.Dense(num_classes, activation='softmax')(gap_layer)

    model = keras.models.Model(inputs=input_layer, outputs=output_layer)
    model.compile(loss='categorical_crossentropy', optimizer=keras.optimizers.Adam(), metrics=['accuracy'])

    return model
