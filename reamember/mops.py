import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import classification_report
from .eam.associative import AssociativeMemory

def rsize_recall(recall, msize, min_value, max_value):
    if (msize == 1):
        return (recall.astype(dtype=float) + 1.0)*(max_value - min_value)/2
    else:
        return (max_value - min_value) * recall.astype(dtype=float) \
            / (msize - 1.0) + min_value

def memorize(eam, dataset, filling_percent=1.0):
    """
    Create and fill memory registering features from the dataset.
    """

    features = dataset.data
    min_value = features.min()
    max_value = features.max()
    m = eam.m
    features_rounded = torch.round((features - min_value) / (max_value - min_value) * (m - 1)).to(torch.int16)

    if filling_percent < 1.0:
        n_features = int(len(features_rounded) * filling_percent)
        features_rounded = features_rounded[:n_features]

    print(f"[INFO] Memorizing {len(features_rounded)} features with shape {features_rounded.shape}...")
    for features in tqdm(features_rounded):
        eam.register(features)

    return eam

def remember(cfg, eam, dataset):
    """
    Remember features from the dataset.
    """

    features = dataset.data
    min_value = features.min()
    max_value = features.max()
    m = eam.m
    features_rounded = torch.round((features - min_value) / (max_value - min_value) * (m - 1)).to(torch.int16)

    memories_features = []
    memories_recognition = []
    memories_weights = []

    print(f"[INFO] Remembering {len(features_rounded)} features with shape {features_rounded.shape}...")
    for feature in tqdm(features_rounded):
        memory, recognized, weight = eam.recall(feature)
        memory = memory.cpu().numpy()
        recognized = recognized.cpu().numpy()
        weight = weight.cpu().numpy()

        memories_features.append(memory)
        memories_recognition.append(recognized)
        memories_weights.append(weight)

    memories_features = np.array(memories_features, dtype=float)
    memories_features = rsize_recall(memories_features, cfg.neural.latent_dim, min_value, max_value)
    memories_recognition = np.array(memories_recognition, dtype=int)
    memories_weights = np.array(memories_weights, dtype=float)

    return memories_features, memories_recognition, memories_weights

def evalm(eam, classifier, dataset):
    """
    Evaluate the memory on the dataset.
    """

    features = dataset.data
    min_value = features.min()
    max_value = features.max()
    m = eam.m
    features_rounded = torch.round((features - min_value) / (max_value - min_value) * (m - 1)).to(torch.int16)
    labels = dataset.targets.cpu().numpy()

    print(f"[INFO] Evaluating {len(features_rounded)} features with shape {features_rounded.shape}...")

    answers = []
    
    for feature in tqdm(features_rounded):
        memory, recognized, weight = eam.recall(feature)
        print(memory, recognized, weight)
        if recognized:
            memory = memory.cpu().numpy()
            memory = rsize_recall(memory, eam.size(), min_value, max_value)
            with torch.no_grad():
                prediction = classifier.predict(memory)
        else:
            prediction = None # No prediction if not recognized

        answers.append(prediction)

    # Results

    answers = np.array(answers, dtype=object)
    recognized_percentage = np.sum(answers != None) / len(answers) * 100
    predictions = answers[answers != None]
    labels = labels[answers != None]
    report = classification_report(labels, predictions, output_dict=True)

    return recognized_percentage, report

