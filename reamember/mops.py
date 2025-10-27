import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score

def rsize_recall(recall, msize, min_value, max_value):
    if not torch.is_tensor(recall):
        min_value = min_value.item() if torch.is_tensor(min_value) else min_value
        max_value = max_value.item() if torch.is_tensor(max_value) else max_value
    if (msize == 1):
        return (recall.astype(dtype=float) + 1.0)*(max_value - min_value)/2
    else:
        #print(f"[DEBUG] Resizing recall from {recall.shape} to msize {msize} with min {min_value} and max {max_value}")
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

    from omegaconf import ListConfig

    features = dataset.data
    min_value = features.min()
    max_value = features.max()
    m = eam.m
    features_rounded = torch.round((features - min_value) / (max_value - min_value) * (m - 1)).to(torch.int16)

    memories_features = []
    memories_recognition = []
    memories_weights = []

    latent = int(cfg.neural.latent_dim[0]) if isinstance(cfg.neural.latent_dim, ListConfig) else int(cfg.neural.latent_dim)

    for feature in tqdm(features_rounded):
        memory, recognized, weight = eam.recall(feature)
        memory = memory.cpu().numpy() if torch.is_tensor(memory) else memory
        recognized = recognized.cpu().numpy() if torch.is_tensor(recognized) else recognized
        weight = weight.cpu().numpy() if torch.is_tensor(weight) else weight

        memories_features.append(memory)
        memories_recognition.append(recognized)
        memories_weights.append(weight)

    memories_features = np.array(memories_features, dtype=float)
    memories_features = rsize_recall(memories_features, latent, min_value, max_value)
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
        if recognized:
            memory = memory.cpu().numpy() if torch.is_tensor(memory) else memory
            memory = rsize_recall(memory, eam.m, min_value, max_value)
            with torch.no_grad():
                memory = torch.tensor(memory, dtype=torch.float32, device=classifier.device).unsqueeze(0) if not isinstance(memory, torch.Tensor) else memory.unsqueeze(0)
                prediction = classifier.predict(memory).cpu().numpy()[0]
        else:
            prediction = None # No prediction if not recognized

        answers.append(prediction)

    # Results

    answers = np.array(answers)
    recognized = answers != None
    predictions = answers[recognized]
    labels = labels[recognized]

    true_positive = 0
    error_count = 0
    if len(predictions) == len(labels) and len(predictions) > 0:
        for i in range(len(predictions)):
            if predictions[i] == labels[i]:
                true_positive += 1
            else:
                error_count += 1

    recognized_count = np.sum(recognized)
    unrecognized_count = len(answers) - recognized_count
    recognized_percentage = recognized_count / len(answers)
    unrecognized_percentage = unrecognized_count / len(answers)
    correct_count_percentage = true_positive / len(predictions)
    incorrect_count_percentage = error_count / len(predictions)

    recall = true_positive / len(answers)
    precision = true_positive / (len(answers) - unrecognized_count) if (len(answers) - unrecognized_count) > 0 else 0.0

    return [recognized_percentage, unrecognized_percentage, correct_count_percentage, incorrect_count_percentage], recall, precision