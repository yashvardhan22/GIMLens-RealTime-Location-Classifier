# Replace the existing build_mobilenetv2 in models_custom.py with this function (exact same signature)
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.regularizers import l2
from tensorflow.keras import backend as K

def _se_block(x, reduction=8):
    ch = int(x.shape[-1])
    y = layers.GlobalAveragePooling2D()(x)
    y = layers.Dense(max(4, ch//reduction), activation='relu', kernel_regularizer=l2(1e-5))(y)
    y = layers.Dense(ch, activation='sigmoid', kernel_regularizer=l2(1e-5))(y)
    y = layers.Reshape((1,1,ch))(y)
    return layers.Multiply()([x, y])

def _spatial_attention(x):
    # Keras-safe spatial attention (avg + max pooling across channels -> conv -> sigmoid)
    avg_pool = layers.Lambda(lambda z: K.mean(z, axis=-1, keepdims=True))(x)
    max_pool = layers.Lambda(lambda z: K.max(z, axis=-1, keepdims=True))(x)
    concat = layers.Concatenate(axis=-1)([avg_pool, max_pool])
    attn = layers.Conv2D(1, 7, padding='same', activation='sigmoid', kernel_regularizer=l2(1e-5))(concat)
    return layers.Multiply()([x, attn])

def _multi_scale_head(x, out_filters=128):
    # small, cheap multi-scale head to complement global features
    p1 = layers.Conv2D(out_filters//2, 1, padding='same', use_bias=False, kernel_regularizer=l2(1e-5))(x)
    p1 = layers.BatchNormalization()(p1); p1 = layers.ReLU()(p1)

    p2 = layers.DepthwiseConv2D(3, padding='same', use_bias=False, depthwise_regularizer=l2(1e-5))(x)
    p2 = layers.Conv2D(out_filters//2, 1, padding='same', use_bias=False, kernel_regularizer=l2(1e-5))(p2)
    p2 = layers.BatchNormalization()(p2); p2 = layers.ReLU()(p2)

    merged = layers.Concatenate()([p1, p2])
    merged = layers.Conv2D(out_filters, 1, padding='same', use_bias=False, kernel_regularizer=l2(1e-5))(merged)
    merged = layers.BatchNormalization()(merged)
    merged = layers.ReLU()(merged)
    return merged

def build_mobilenetv2(num_classes, input_shape=(224,224,3), unfreeze_last=40, base_weights='imagenet'):
    """
    Enhanced MobileNetV2:
      - keeps original MobileNetV2 backbone,
      - unfreezes last `unfreeze_last` layers for fine-tuning,
      - adds a light multi-scale conv head, SE block, and spatial attention,
      - small dense head with dropout and L2 to reduce overfitting.
    Signature kept identical to original so training scripts don't change.
    """
    base = MobileNetV2(weights=base_weights, include_top=False, input_shape=input_shape)
    # Freeze all then unfreeze last `unfreeze_last` layers
    for layer in base.layers[:-unfreeze_last]:
        layer.trainable = False
    for layer in base.layers[-unfreeze_last:]:
        layer.trainable = True

    x = base.output  # (H', W', Cb)
    # Small multi-scale head
    x = _multi_scale_head(x, out_filters=128)
    # SE + Spatial Attention
    x = _se_block(x, reduction=8)
    x = _spatial_attention(x)
    # Global pooling -> classifier head with light regularization
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=l2(1e-4))(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation='softmax')(x)

    model = Model(base.input, out, name='mobilenetv2_enhanced')
    return model
