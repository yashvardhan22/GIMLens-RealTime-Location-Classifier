# train_eval_all.py
import os, argparse, json, time, math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score
from models_custom import build_mobilenetv2, build_resnet50, build_inceptionv3, build_shallowcnn, build_gimlensnet

def get_generators(data_dir, target_size=(224,224), batch_size=32, val_split=0.2, seed=123):
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=val_split,
        rotation_range=20,
        zoom_range=0.2,
        width_shift_range=0.15,
        height_shift_range=0.15,
        brightness_range=[0.8,1.2],
        horizontal_flip=True
    )
    val_datagen = ImageDataGenerator(rescale=1./255, validation_split=val_split)
    train_gen = train_datagen.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size,
                                                  class_mode='categorical', subset='training', shuffle=True, seed=seed)
    val_gen = val_datagen.flow_from_directory(data_dir, target_size=target_size, batch_size=batch_size,
                                              class_mode='categorical', subset='validation', shuffle=False, seed=seed)
    return train_gen, val_gen

# metrics helpers
def top_k_acc(probs, labels, k=3):
    topk = np.argsort(probs, axis=1)[:, -k:]
    return np.mean([labels[i] in topk[i] for i in range(len(labels))])

def compute_ece(probs, labels, n_bins=15):
    confidences = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    bin_edges = np.linspace(0,1,n_bins+1)
    ece = 0.0
    for i in range(n_bins):
        low, high = bin_edges[i], bin_edges[i+1]
        mask = (confidences > low) & (confidences <= high)
        if mask.sum() == 0:
            continue
        acc = (preds[mask] == labels[mask]).mean()
        conf = confidences[mask].mean()
        ece += (mask.sum()/len(labels)) * abs(acc - conf)
    return float(ece)

def evaluate_model_keras(model, gen):
    steps = math.ceil(gen.samples / gen.batch_size)
    preds = []
    probs = []
    labels = []
    gen.reset()
    for _ in range(steps):
        x,y = next(gen)
        p = model.predict(x, verbose=0)
        preds.extend(np.argmax(p, axis=1).tolist())
        probs.extend(p.tolist())
        labels.extend(np.argmax(y, axis=1).tolist())
    probs = np.array(probs); preds = np.array(preds); labels = np.array(labels)
    acc = accuracy_score(labels, preds)
    top3 = top_k_acc(probs, labels, k=3)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(labels, preds, average='macro', zero_division=0)
    cm = confusion_matrix(labels, preds)
    ece = compute_ece(probs, labels)
    return {'accuracy':float(acc),'top3':float(top3),'precision_macro':float(p_macro),'recall_macro':float(r_macro),
            'f1_macro':float(f1_macro),'confusion_matrix':cm.tolist(),'ece':ece, 'probs':probs.tolist(), 'labels':labels.tolist()}

def measure_latency(model, gen, steps=50):
    # use predict on random batches from generator
    import time
    gen.reset()
    # warmup
    for _ in range(5):
        x,y = next(gen)
        _ = model.predict(x, verbose=0)
    tot = 0.0; count = 0
    for i in range(steps):
        x,y = next(gen)
        t0 = time.time()
        _ = model.predict(x, verbose=0)
        t1 = time.time()
        tot += (t1 - t0)
        count += x.shape[0]
    per_img = tot / count
    fps = count / tot
    return per_img, fps

def save_json(obj, path):
    with open(path,'w') as f:
        json.dump(obj, f, indent=2)

def main(args):
    train_gen, val_gen = get_generators(args.data_dir, target_size=(args.img_size,args.img_size), batch_size=args.batch_size, val_split=0.2)
    num_classes = train_gen.num_classes
    class_indices = train_gen.class_indices
    print("Classes:", num_classes, class_indices)

    # Build model
    if args.model == 'mobilenetv2':
        model = build_mobilenetv2(num_classes, input_shape=(args.img_size,args.img_size,3), unfreeze_last=20)
    elif args.model == 'resnet50':
        model = build_resnet50(num_classes, input_shape=(args.img_size,args.img_size,3))
    elif args.model == 'inceptionv3':
        model = build_inceptionv3(num_classes, input_shape=(args.img_size,args.img_size,3))
    elif args.model == 'shallow':
        model = build_shallowcnn(num_classes, input_shape=(args.img_size,args.img_size,3))
    elif args.model == 'gimlensnet':
        model = build_gimlensnet(num_classes, input_shape=(args.img_size,args.img_size,3))
    else:
        raise ValueError("Unknown model")

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    ckpt_path = f"best_{args.model}.keras"
    ckpt = ModelCheckpoint(ckpt_path, monitor='val_loss', save_best_only=True, verbose=1)
    early = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    steps_per_epoch = max(1, train_gen.samples // train_gen.batch_size)
    val_steps = max(1, val_gen.samples // val_gen.batch_size)

    history = model.fit(train_gen, validation_data=val_gen, epochs=args.epochs, steps_per_epoch=steps_per_epoch,
                        validation_steps=val_steps, callbacks=[ckpt, early], verbose=1)

    # try to load best saved
    try:
        model = tf.keras.models.load_model(ckpt_path)
    except Exception:
        print("Could not reload saved model; using current weights.")

    metrics = evaluate_model_keras(model, val_gen)
    per_img, fps = measure_latency(model, val_gen, steps=40)
    metrics['per_image_sec'] = per_img
    metrics['throughput_fps'] = fps
    metrics['params'] = model.count_params()
    try:
        # attempt to get size on disk
        import tempfile, shutil
        tmp = tempfile.mkdtemp()
        model.save(tmp)
        total = 0
        for root,_,files in os.walk(tmp):
            for fn in files:
                total += os.path.getsize(os.path.join(root,fn))
        shutil.rmtree(tmp)
        metrics['size_MB'] = total/(1024*1024)
    except Exception:
        metrics['size_MB'] = None

    # save outputs
    out_json = f"results_{args.model}.json"
    save_json(metrics, out_json)
    pd.DataFrame(history.history).to_csv(f"history_{args.model}.csv", index=False)
    print("Saved", out_json, "and history CSV.")
    print("Metrics summary:", {k:metrics[k] for k in ['accuracy','f1_macro','per_image_sec','throughput_fps','params','size_MB','ece']})

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='D:/ML/dataset_classes')
    parser.add_argument('--model', type=str, choices=['mobilenetv2','resnet50','inceptionv3','shallow','gimlensnet'], default='mobilenetv2')
    parser.add_argument('--img_size', type=int, default=224)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--lr', type=float, default=1e-4)
    args = parser.parse_args()
    main(args)
