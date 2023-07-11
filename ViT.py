import tensorflow as tf
from tensorflow.keras.layers.experimental.preprocessing import Rescaling,Normalization,RandomFlip,RandomRotation,RandomZoom
# from tensorflow.keras.layers import Rescaling,Normalization,RandomFlip,RandomRotation,RandomZoom
from tensorflow.keras.layers import  Layer,Dense,Dropout,Embedding,Input,LayerNormalization,MultiHeadAttention,Add,Flatten
from tensorflow.keras import Model,Sequential


def DataNormlayer(input_data): ### Data Augmentation Layer 
    DataNormlayer = Sequential(
        [
            Rescaling(1./255),
            Normalization(),
            # RandomFlip(),
            # RandomRotation(factor=0.02),
            # RandomZoom(height_factor=0.2, width_factor=0.2),
        ],name="data_augmentation",
    )
    DataNormlayer.layers[0].adapt(input_data)
    
    return DataNormlayer

def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = Dense(units, activation=tf.nn.gelu)(x)
        x = Dropout(dropout_rate)(x)
    return x

def mlp2(x,dropout_rate): ### For Suboutput Layer
    x=Flatten()(x)
    x=Dense(64,activation=tf.nn.gelu)(x)
    x=Dropout(dropout_rate)(x)
    x=Dense(32,activation=tf.nn.gelu)(x)
    x=Dropout(dropout_rate)(x)
    return x

class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size

    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches

class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = Dense(units=projection_dim)
        self.position_embedding = Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )

    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded

### Original ViT
def create_vit_classifier_original(input_data,CFG): 
    inputs = Input(shape=CFG.input_shape)
    aug_layer=augmentation_layer(input_data)(inputs)
    patches=Patches(CFG.patch_size)(aug_layer)
    # patches=Patches(CFG.patch_size)(inputs)

    # Encode patches.
    encoded_patches = PatchEncoder(CFG.num_patches, CFG.projection_dim)(patches)
    # Create multiple layers of the Transformer block.
    for i in range(CFG.transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=CFG.num_heads, key_dim=CFG.projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=CFG.transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = Add(name='transfotmer_block_out_{}'.format(i))([x3, x2])

                                           
    
    # Create a [ch_size, projection_dim] tensor.
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=CFG.mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = Dense(CFG.num_classes,name="fin_out")(features)
    # Create the Keras model.
    model = Model(inputs=inputs, outputs=logits)
    return model


def create_vit_classifier(input_data,CFG):
    inputs = Input(shape=CFG.input_shape)
    aug_layer=augmentation_layer(input_data)(inputs)
    patches=Patches(CFG.patch_size)(aug_layer)

    # Encode patches.
    encoded_patches = PatchEncoder(CFG.num_patches, CFG.projection_dim)(patches)
    transformer_layer_output=[]
    # Create multiple layers of the Transformer block.
    for i in range(CFG.transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=CFG.num_heads, key_dim=CFG.projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=CFG.transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = Add(name='transfotmer_block_out_{}'.format(i))([x3, x2])
        if i is (CFG.transformer_layers-1):
            pass
        else:
            # x=mlp(encoded_patches,hidden_units=mlp_head_units,dropout_rate=0.5)
            x=mlp2(encoded_patches,dropout_rate=0.5)
            x=Flatten()(x)
            transformer_layer_output.append(
                Dense(CFG.num_classes,name="subout_{}".format(i))(x)
            )
                                           
    
    # Create a [batch_size, projection_dim] tensor.
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=CFG.mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = Dense(CFG.num_classes,name="fin_out")(features)
    transformer_layer_output.insert(0,logits)
    # Create the Keras model.
    model = Model(inputs=inputs, outputs=transformer_layer_output)
    return model


def create_vit_classifier_ver2(input_data,CFG):
    inputs = Input(shape=CFG.input_shape)
    DataNorm_layer=DataNormlayer(input_data)(inputs)
    patches=Patches(CFG.patch_size)(DataNorm_layer)

    # Encode patches.
    encoded_patches = PatchEncoder(CFG.num_patches, CFG.projection_dim)(patches)
        
    # Create multiple layers of the Transformer block.
    transformer_layer_output=[]
    for i in range(CFG.transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(name='MHA_{}'.format(i),
            num_heads=CFG.num_heads, key_dim=CFG.projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=CFG.transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = Add(name='transfotmer_block_out_{}'.format(i))([x3, x2])
        if i is (CFG.transformer_layers-1):
            pass
        else:
            # x=mlp(encoded_patches,hidden_units=mlp_head_units,dropout_rate=0.5)
            x=mlp2(encoded_patches,dropout_rate=0.5)
            x=Flatten()(x)
            transformer_layer_output.append(
                Dense(CFG.num_classes,name="subout_{}".format(i))(x)
            )
                                           
    
    # Create a [batch_size, projection_dim] tensor.
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    representation = Flatten()(representation)
    representation = Dropout(0.5)(representation)
    # Add MLP.
    features = mlp(representation, hidden_units=CFG.mlp_head_units, dropout_rate=0.5)
    # Classify outputs.
    logits = Dense(CFG.num_classes,name="fin_out")(features)
    transformer_layer_output.insert(0,logits)
    # Create the Keras model.
    return Model(inputs=inputs, outputs=transformer_layer_output[0],name="main"), Model(inputs=inputs, outputs=transformer_layer_output[1],name="aux")