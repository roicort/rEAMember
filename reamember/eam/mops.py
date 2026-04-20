import numpy as np
import torch
from tqdm import tqdm


def _get_dataset_features(dataset):
    if hasattr(dataset, "data"):
        return dataset.data
    return dataset


def rsize_recall(recall, msize, min_value, max_value):
    if not torch.is_tensor(recall):
        min_value = min_value.item() if torch.is_tensor(min_value) else min_value
        max_value = max_value.item() if torch.is_tensor(max_value) else max_value
    if msize == 1:
        return (recall.astype(dtype=float) + 1.0) * (max_value - min_value) / 2
    else:
        # print(f"[DEBUG] Resizing recall from {recall.shape} to msize {msize} with min {min_value} and max {max_value}")
        return (max_value - min_value) * recall.astype(dtype=float) / (
            msize - 1.0
        ) + min_value


def _quantize_features(features, eam, min_value, max_value):
    scale = max_value - min_value
    if torch.is_tensor(scale):
        eps = torch.finfo(features.dtype).eps
        scale = torch.clamp(scale, min=eps)
    elif scale == 0:
        scale = 1.0

    return torch.round((features - min_value) / scale * (eam.m - 1)).to(torch.int16)


def memorize(eam, dataset, quantize_min, quantize_max, filling_percent=1.0):
    """
    Create and fill memory registering features from the dataset.
    """

    features = _get_dataset_features(dataset)
    features_rounded = _quantize_features(features, eam, quantize_min, quantize_max)

    if filling_percent < 1.0:
        n_features = int(len(features_rounded) * filling_percent)
        features_rounded = features_rounded[:n_features]

    print(
        f"[INFO] Memorizing {len(features_rounded)} features with shape {features_rounded.shape}..."
    )
    for features in tqdm(features_rounded):
        eam.register(features)

    return eam


def remember(cfg, eam, dataset, dequantize_min, dequantize_max):
    """
    Remember features from the dataset.
    """

    features = _get_dataset_features(dataset)
    features_rounded = _quantize_features(features, eam, dequantize_min, dequantize_max)

    memories_features = []
    memories_recognition = []
    memories_weights = []

    for feature in tqdm(features_rounded):
        memory, recognized, weight = eam.recall(feature)
        memory = memory.cpu().numpy() if torch.is_tensor(memory) else memory
        recognized = (
            recognized.cpu().numpy() if torch.is_tensor(recognized) else recognized
        )
        weight = weight.cpu().numpy() if torch.is_tensor(weight) else weight

        memories_features.append(memory)
        memories_recognition.append(recognized)
        memories_weights.append(weight)

    memories_features = np.array(memories_features, dtype=float)
    memories_features = rsize_recall(memories_features, eam.m, dequantize_min, dequantize_max)
    memories_recognition = np.array(memories_recognition, dtype=int)
    memories_weights = np.array(memories_weights, dtype=float)

    return memories_features, memories_recognition, memories_weights


def evalm(eam, classifier, dataset, quantize_min, quantize_max):
    """
    Evaluate the memory on the dataset.
    """

    features = _get_dataset_features(dataset)
    features_rounded = _quantize_features(features, eam, quantize_min, quantize_max)
    labels = dataset.targets.cpu().numpy()

    print(
        f"[INFO] Evaluating {len(features_rounded)} features with shape {features_rounded.shape}..."
    )

    answers = []

    for feature in tqdm(features_rounded):
        memory, recognized, weight = eam.recall(feature)
        if recognized:
            memory = memory.cpu().numpy() if torch.is_tensor(memory) else memory
            memory = rsize_recall(memory, eam.m, quantize_min, quantize_max)
            with torch.no_grad():
                memory = (
                    torch.tensor(
                        memory, dtype=torch.float32, device=classifier.device
                    ).unsqueeze(0)
                    if not isinstance(memory, torch.Tensor)
                    else memory.unsqueeze(0)
                )
                prediction = classifier.predict(memory).cpu().numpy()[0]
        else:
            prediction = None  # No prediction if not recognized

        answers.append(prediction)

    # Results

    answers = np.array(answers)
    recognized = np.array([answer is not None for answer in answers], dtype=bool)
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
    correct_count_percentage = true_positive / len(answers)
    incorrect_count_percentage = error_count / len(answers)

    recall = true_positive / len(answers)
    precision = (
        true_positive / (len(answers) - unrecognized_count)
        if (len(answers) - unrecognized_count) > 0
        else 0.0
    )

    return (
        [
            recognized_percentage,
            unrecognized_percentage,
            correct_count_percentage,
            incorrect_count_percentage,
        ],
        recall,
        precision,
    )


def evalm_text(eam, dataset, quantize_min, quantize_max):
    """
    Evaluate the memory on a text-embedding dataset.
    """

    features_rounded = _quantize_features(dataset.data, eam, quantize_min, quantize_max)

    print(
        f"[INFO] Evaluating {len(features_rounded)} text features with shape {features_rounded.shape}..."
    )

    memories_features = []
    recognitions = []
    weights = []

    for feature in tqdm(features_rounded):
        memory, recognized, weight = eam.recall(feature)
        memory = memory.cpu().numpy() if torch.is_tensor(memory) else memory
        recognized = bool(recognized.cpu().item()) if torch.is_tensor(recognized) else bool(recognized)
        weight = weight.cpu().numpy() if torch.is_tensor(weight) else weight

        memories_features.append(memory)
        recognitions.append(recognized)
        weights.append(weight)

    memories_features = np.array(memories_features, dtype=float)
    memories_features = rsize_recall(memories_features, eam.m, quantize_min, quantize_max)
    recognitions = np.array(recognitions, dtype=bool)
    weights = np.array(weights, dtype=float)

    recognized_count = np.sum(recognitions)
    total = len(recognitions)
    recognized_percentage = recognized_count / total if total > 0 else 0.0
    unrecognized_percentage = 1.0 - recognized_percentage if total > 0 else 0.0

    return memories_features, recognitions, weights, recognized_percentage, unrecognized_percentage


def evalm_text_confusion(
    eam,
    seen_dataset,
    unseen_dataset,
    quantize_min,
    quantize_max,
    test_dataset=None,
):
    """
    Build a binary confusion matrix for text memory recognition.
    Rows: seen, unseen, optional test.
    Columns: recognized, unrecognized.
    """

    # Eam is filled with only the seen_dataset (first half of the training dataset)

    # Evaluate on both seen and unseen datasets
    _, seen_recognitions, seen_weights, seen_recognized, seen_unrecognized = evalm_text(
        eam, seen_dataset, quantize_min, quantize_max
    )
    # Evaluate on the unseen dataset
    _, unseen_recognitions, unseen_weights, unseen_recognized, unseen_unrecognized = evalm_text(
        eam, unseen_dataset, quantize_min, quantize_max
    )

    row_labels = ["seen", "unseen"]
    matrix_rows = []
    counts = {}
    rates = {
        "seen_recognized_rate": float(seen_recognized),
        "seen_unrecognized_rate": float(seen_unrecognized),
        "unseen_recognized_rate": float(unseen_recognized),
        "unseen_unrecognized_rate": float(unseen_unrecognized),
    }

    seen_total = len(seen_recognitions)
    unseen_total = len(unseen_recognitions)
    matrix_rows.append(
        [int(np.sum(seen_recognitions)), int(seen_total - np.sum(seen_recognitions))]
    )
    matrix_rows.append(
        [
            int(np.sum(unseen_recognitions)),
            int(unseen_total - np.sum(unseen_recognitions)),
        ]
    )
    counts["seen_total"] = int(seen_total)
    counts["unseen_total"] = int(unseen_total)

    if test_dataset is not None:
        _, test_recognitions, _, test_recognized, test_unrecognized = evalm_text(
            eam, test_dataset, quantize_min, quantize_max
        )
        test_total = len(test_recognitions)
        row_labels.append("test")
        matrix_rows.append(
            [int(np.sum(test_recognitions)), int(test_total - np.sum(test_recognitions))]
        )
        counts["test_total"] = int(test_total)
        rates["test_recognized_rate"] = float(test_recognized)
        rates["test_unrecognized_rate"] = float(test_unrecognized)

    matrix = np.array(matrix_rows, dtype=int)

    return {
        "labels": {
            "rows": row_labels,
            "columns": ["recognized", "unrecognized"],
        },
        "matrix": matrix,
        "counts": counts,
        "rates": rates,
    }
