import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Union
import torch
import librosa
from datasets import Dataset
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2CTCTokenizer
)
import librosa
import json
from jiwer import wer, cer

def create_vocab(df, vocab_path):
    """
    Creates a character-level vocabulary from the dataset and saves it as a JSON file.

    Args:
        df (DataFrame): Data containing a 'text' column.
        vocab_path (str): Path to save the vocabulary JSON.
    
    Returns:
        dict: Vocabulary dictionary.
    """
    vocab_dict = {}
    for text in df['text']:
        for char in text:
            if char.strip():
                vocab_dict[char] = vocab_dict.get(char, 0) + 1
    
    sorted_chars = sorted(vocab_dict.items(), key=lambda x: x[1], reverse=True)

    vocab_dict = {}
    vocab_dict["[PAD]"] = 0
    vocab_dict["[UNK]"] = 1
    vocab_dict["|"] = 2
    
    for char, _ in sorted_chars:
        if char not in vocab_dict:
            vocab_dict[char] = len(vocab_dict)
    
    with open(vocab_path, 'w', encoding='utf-8') as vocab_file:
        json.dump(vocab_dict, vocab_file, ensure_ascii=False, indent=2)
    
    return vocab_dict

def load_audio(audio_path):
    """
    Loads audio file and resamples to 16kHz.

    Args:
        audio_path (str): Path to the audio file.
    
    Returns:
        np.ndarray: Audio signal.
    """
    audio, sr = librosa.load(audio_path, sr=16000)
    return audio

@dataclass
class DataCollatorCTCWithPadding:
    """
    Data collator that pads input values and labels for CTC training.

    Args:
        processor (Wav2Vec2Processor): HuggingFace processor.
        padding (bool or str): Padding strategy.
    """
    processor: Wav2Vec2Processor
    padding: Union[bool, str] = True

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        input_features = [{"input_values": feature["input_values"]} for feature in features]
        label_features = [feature["labels"] for feature in features]

        batch = self.processor.pad(
            input_features,
            padding=self.padding,
            return_tensors="pt",
        )

        label_lengths = [len(label) for label in label_features]
        max_label_len = max(label_lengths)
        
        labels = []
        for label in label_features:
            remainder = [-100] * (max_label_len - len(label))
            labels.append(label + remainder)
        
        batch["labels"] = torch.tensor(labels)

        return batch

def prepare_split(df, mapper, processor):
    """
    Splits dataset into train and eval sets, applies mapper, and returns collator.

    Args:
        df (DataFrame): Dataset with audio paths.
        mapper (function): Function to process examples.
        processor (Wav2Vec2Processor): Processor for padding and tokenizing.
    
    Returns:
        tuple: (train_dataset, eval_dataset, data_collator)
    """
    dataset = Dataset.from_pandas(df)
    dataset = dataset.map(mapper, remove_columns=["audio_path"])

    train_size = int(0.9 * len(dataset))
    train_dataset = dataset.select(range(train_size))
    eval_dataset = dataset.select(range(train_size, len(dataset)))

    data_collator = DataCollatorCTCWithPadding(processor=processor, padding=True)
    return train_dataset, eval_dataset, data_collator

def get_tokenizer(path):
    """
    Loads Wav2Vec2 tokenizer from a vocabulary file.

    Args:
        path (str): Path to vocabulary JSON.
    
    Returns:
        Wav2Vec2CTCTokenizer: Loaded tokenizer.
    """
    return Wav2Vec2CTCTokenizer(
        path, 
        unk_token="[UNK]", 
        pad_token="[PAD]", 
        word_delimiter_token="|",
        do_lower_case=False,
        strip_accents=False
    )

def create_compute_metrics(processor):
    """
    Returns a function to compute WER and CER from model predictions.

    Args:
        processor (Wav2Vec2Processor): Processor for decoding predictions.
    
    Returns:
        function: Metric computation function.
    """
    def compute_metrics(pred):
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        
        labels = pred.label_ids.copy()
        labels[labels == -100] = processor.tokenizer.pad_token_id
        
        pred_str = processor.batch_decode(pred_ids)
        label_str = processor.batch_decode(labels, group_tokens=False)
        
        wer_score = wer(label_str, pred_str)
        cer_score = cer(label_str, pred_str)
        
        return {
            "wer": wer_score,
            "cer": cer_score
        }
    
    return compute_metrics