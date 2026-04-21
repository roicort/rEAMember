import numpy as np
import torch
from tqdm import tqdm


def _get_dataset_features(dataset):
    if hasattr(dataset, "data"):
        return dataset.data
    return dataset


def _to_numpy(value):
    if torch.is_tensor(value):
        return value.cpu().numpy()
    return np.asarray(value)


def _iter_batches(values, batch_size=None):
    total = len(values)
    if batch_size is None or batch_size <= 0 or batch_size >= total:
        yield values
        return

    for start in range(0, total, batch_size):
        yield values[start : start + batch_size]


def _batch_count(values, batch_size=None):
    total = len(values)
    if total == 0:
        return 0
    if batch_size is None or batch_size <= 0 or batch_size >= total:
        return 1
    return (total + batch_size - 1) // batch_size


def memorize(eam, dataset, quantizer, filling_percent=1.0, batch_size=None):
    """
    Create and fill memory registering features from the dataset.
    """
    batch_register = getattr(eam, "batch_register", None)

    features = _to_numpy(_get_dataset_features(dataset))
    features_rounded = quantizer.quantize(features, eam.m)

    if filling_percent < 1.0:
        n_features = int(len(features_rounded) * filling_percent)
        features_rounded = features_rounded[:n_features]

    print(
        f"[INFO] Memorizing {len(features_rounded)} features with shape {features_rounded.shape}..."
    )

    if callable(batch_register):
        for batch in tqdm(
            _iter_batches(features_rounded, batch_size),
            total=_batch_count(features_rounded, batch_size),
        ):
            batch_register(_to_numpy(batch))
    else:
        for features in tqdm(features_rounded):
            eam.register(features)
    return eam


def remember(cfg, eam, dataset, quantizer, batch_size=None):
    """
    Remember features from the dataset.
    """

    features = _to_numpy(_get_dataset_features(dataset))
    features_rounded = quantizer.quantize(features, eam.m)

    memories_features = []
    memories_recognition = []
    memories_weights = []

    batch_recall = getattr(eam, "batch_recall", None)

    if callable(batch_recall):
        memories_features = []
        memories_recognition = []
        memories_weights = []

        for batch in tqdm(
            _iter_batches(features_rounded, batch_size),
            total=_batch_count(features_rounded, batch_size),
        ):
            batch_memories, batch_recognition, batch_weights = batch_recall(
                _to_numpy(batch)
            )
            memories_features.append(_to_numpy(batch_memories))
            memories_recognition.append(_to_numpy(batch_recognition))
            memories_weights.append(_to_numpy(batch_weights))

        memories_features = np.concatenate(memories_features, axis=0).astype(float)
        memories_features = quantizer.dequantize(memories_features, eam.m)
        memories_recognition = np.concatenate(memories_recognition, axis=0).astype(int)
        memories_weights = np.concatenate(memories_weights, axis=0).astype(float)
        return memories_features, memories_recognition, memories_weights

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
    memories_features = quantizer.dequantize(memories_features, eam.m)
    memories_recognition = np.array(memories_recognition, dtype=int)
    memories_weights = np.array(memories_weights, dtype=float)

    return memories_features, memories_recognition, memories_weights


def evalm(eam, classifier, dataset, quantizer, batch_size=None):
    """
    Evaluate the memory on the dataset.
    """

    features = _to_numpy(_get_dataset_features(dataset))
    features_rounded = quantizer.quantize(features, eam.m)
    labels = dataset.targets.cpu().numpy()

    print(
        f"[INFO] Evaluating {len(features_rounded)} features with shape {features_rounded.shape}..."
    )

    batch_recall = getattr(eam, "batch_recall", None)

    if callable(batch_recall):
        answers = []

        for batch in tqdm(
            _iter_batches(features_rounded, batch_size),
            total=_batch_count(features_rounded, batch_size),
        ):
            memories, recognized, _ = batch_recall(_to_numpy(batch))
            memories = _to_numpy(memories)
            recognized = _to_numpy(recognized).astype(bool)
            batch_answers = np.full(len(recognized), None, dtype=object)

            if np.any(recognized):
                recognized_memories = quantizer.dequantize(
                    memories[recognized].astype(float), eam.m
                )
                with torch.no_grad():
                    prediction_batch = torch.as_tensor(
                        recognized_memories,
                        dtype=torch.float32,
                        device=classifier.device,
                    )
                    predictions = classifier.predict(prediction_batch).cpu().numpy()
                batch_answers[recognized] = predictions.tolist()

            answers.append(batch_answers)

        answers = np.concatenate(answers, axis=0)
    else:
        answers = []

        for feature in tqdm(features_rounded):
            memory, recognized, weight = eam.recall(feature)
            if recognized:
                memory = memory.cpu().numpy() if torch.is_tensor(memory) else memory
                memory = quantizer.dequantize(memory, eam.m)
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
                prediction = None

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


def evalm_text(eam, dataset, quantizer, batch_size=None):
    """
    Evaluate the memory on a text-embedding dataset.
    """

    features_rounded = quantizer.quantize(_to_numpy(dataset.data), eam.m)

    print(
        f"[INFO] Evaluating {len(features_rounded)} text features with shape {features_rounded.shape}..."
    )

    memories_features = []
    recognitions = []
    weights = []

    batch_recall = getattr(eam, "batch_recall", None)

    if callable(batch_recall):
        memories_features = []
        recognitions = []
        weights = []

        for batch in tqdm(
            _iter_batches(features_rounded, batch_size),
            total=_batch_count(features_rounded, batch_size),
        ):
            batch_memories, batch_recognitions, batch_weights = batch_recall(
                _to_numpy(batch)
            )
            memories_features.append(_to_numpy(batch_memories))
            recognitions.append(_to_numpy(batch_recognitions))
            weights.append(_to_numpy(batch_weights))

        memories_features = np.concatenate(memories_features, axis=0).astype(float)
        memories_features = quantizer.dequantize(memories_features, eam.m)
        recognitions = np.concatenate(recognitions, axis=0).astype(bool)
        weights = np.concatenate(weights, axis=0).astype(float)

        recognized_count = np.sum(recognitions)
        total = len(recognitions)
        recognized_percentage = recognized_count / total if total > 0 else 0.0
        unrecognized_percentage = 1.0 - recognized_percentage if total > 0 else 0.0

        return (
            memories_features,
            recognitions,
            weights,
            recognized_percentage,
            unrecognized_percentage,
        )

    for feature in tqdm(features_rounded):
        memory, recognized, weight = eam.recall(feature)
        memory = memory.cpu().numpy() if torch.is_tensor(memory) else memory
        recognized = bool(recognized.cpu().item()) if torch.is_tensor(recognized) else bool(recognized)
        weight = weight.cpu().numpy() if torch.is_tensor(weight) else weight

        memories_features.append(memory)
        recognitions.append(recognized)
        weights.append(weight)

    memories_features = np.array(memories_features, dtype=float)
    memories_features = quantizer.dequantize(memories_features, eam.m)
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
    quantizer,
    test_dataset=None,
    batch_size=None,
):
    """
    Build a binary confusion matrix for text memory recognition.
    Rows: seen, unseen, optional test.
    Columns: recognized, unrecognized.
    """

    # Eam is filled with only the seen_dataset (first half of the training dataset)

    # Evaluate on both seen and unseen datasets
    _, seen_recognitions, seen_weights, seen_recognized, seen_unrecognized = evalm_text(
        eam, seen_dataset, quantizer, batch_size=batch_size
    )
    # Evaluate on the unseen dataset
    _, unseen_recognitions, unseen_weights, unseen_recognized, unseen_unrecognized = evalm_text(
        eam, unseen_dataset, quantizer, batch_size=batch_size
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
            eam, test_dataset, quantizer, batch_size=batch_size
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
