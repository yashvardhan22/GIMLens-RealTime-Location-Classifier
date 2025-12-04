# models_custom.py
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2, ResNet50, InceptionV3
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

def build_mobilenetv2(num_classes, input_shape=(224,224,3), unfreeze_last=20, base_weights='imagenet'):
    base = MobileNetV2(weights=base_weights, include_top=False, input_shape=input_shape)
    for layer in base.layers[:-unfreeze_last]:
        layer.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(64, activation='relu', kernel_regularizer=l2(0.001))(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(base.input, out, name='mobilenetv2_ft')
    return model

def build_resnet50(num_classes, input_shape=(224,224,3), base_weights='imagenet'):
    base = ResNet50(weights=base_weights, include_top=False, input_shape=input_shape)
    for layer in base.layers[:-50]:
        layer.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(base.input, out, name='resnet50_ft')
    return model

def build_inceptionv3(num_classes, input_shape=(224,224,3), base_weights='imagenet'):
    base = InceptionV3(weights=base_weights, include_top=False, input_shape=input_shape)
    for layer in base.layers[:-100]:
        layer.trainable = False
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(base.input, out, name='inceptionv3_ft')
    return model

def shallow_cnn_block(inp, filters, k=3, stride=1):
    x = layers.Conv2D(filters, k, strides=stride, padding='same')(inp)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def build_shallowcnn(num_classes, input_shape=(224,224,3)):
    inp = layers.Input(shape=input_shape)
    x = shallow_cnn_block(inp, 32, 3, 2)
    x = shallow_cnn_block(x, 64, 3, 2)
    x = shallow_cnn_block(x, 128, 3, 2)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inp, out, name='shallow_cnn')
    return model

# ---- GimLensNet: custom model which looks original and is practical ----
def se_block(inp, reduction=8):
    ch = int(inp.shape[-1])
    x = layers.GlobalAveragePooling2D()(inp)
    x = layers.Dense(ch//reduction, activation='relu')(x)
    x = layers.Dense(ch, activation='sigmoid')(x)
    x = layers.Reshape((1,1,ch))(x)
    return layers.Multiply()([inp, x])

def spatial_attention(inp):
    """
    Spatial attention implemented using Keras layers / Lambda to operate on KerasTensors safely.
    Produces a spatial attention map from channel-wise avg and max pooling, then multiplies it.
    """
    # channel-wise average pooling -> (H, W, 1)
    avg_pool = layers.Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(inp)
    # channel-wise max pooling -> (H, W, 1)
    max_pool = layers.Lambda(lambda x: K.max(x, axis=-1, keepdims=True))(inp)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])  # (H, W, 2)
    attn = layers.Conv2D(1, 7, padding='same', activation='sigmoid', kernel_initializer='he_normal')(concat)
    return layers.Multiply()([inp, attn])

def depthwise_sep_conv(x, filters, kernel=3, stride=1):
    x = layers.DepthwiseConv2D(kernel, strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    return x

def build_gimlensnet(num_classes, input_shape=(224,224,3)):
    inp = layers.Input(shape=input_shape)
    # Stem
    x = layers.Conv2D(32, 3, strides=2, padding='same', kernel_initializer='he_normal')(inp)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)

    # Block 1: small mobile-like stack
    x = depthwise_sep_conv(x, 64, kernel=3, stride=1)
    x = se_block(x)
    x = spatial_attention(x)

    # Block 2: downsample + multi-scale fusion
    x1 = depthwise_sep_conv(x, 96, kernel=3, stride=2)  # lower res
    x2 = depthwise_sep_conv(x, 96, kernel=5, stride=2)  # different receptive field
    x = layers.Concatenate()([x1, x2])
    x = layers.Conv2D(128, 1, padding='same', kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization()(x); x = layers.ReLU()(x)
    x = se_block(x)

    # Block 3: lightweight deeper features
    x = depthwise_sep_conv(x, 160, kernel=3, stride=2)
    x = depthwise_sep_conv(x, 192, kernel=3, stride=1)
    x = se_block(x)

    # Head
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inp, out, name='gimlensnet')
    return model
