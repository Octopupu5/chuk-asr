import asr_utils
from transformers import (
    Wav2Vec2Processor,
    Wav2Vec2ForCTC
)

class ChukchiASR:
    """
    ASR wrapper class for transcription using Wav2Vec2 model.
    
    Methods:
        transcribe: Transcribes raw audio.
        transcribe_file: Transcribes audio from file.
        transcribe_batch: Transcribes list of audio paths.
    """
    def __init__(self, model_path):
        self.device = asr_utils.torch.device("cuda" if asr_utils.torch.cuda.is_available() else "cpu")
        
        self.processor = Wav2Vec2Processor.from_pretrained(model_path)
        self.model = Wav2Vec2ForCTC.from_pretrained(model_path)
        self.model.to(self.device)
        self.model.eval()
    
    def transcribe(self, audio):
        inputs = self.processor(audio, sampling_rate=16000, return_tensors="pt", padding=True)
        input_values = inputs.input_values.to(self.device)
        
        with asr_utils.torch.no_grad():
            logits = self.model(input_values).logits
            predicted_ids = asr_utils.torch.argmax(logits, dim=-1)
        
        transcription = self.processor.batch_decode(predicted_ids)[0]
        return transcription
    
    def transcribe_file(self, audio_path):
        audio = asr_utils.load_audio(audio_path)
        return self.transcribe(audio)
    
    def transcribe_batch(self, audio_paths):
        results = []
        for i, audio_path in enumerate(audio_paths):
            transcription = self.transcribe_file(audio_path)
            results.append(transcription)
        return results

def calculate_metrics(predictions, references):
    """
    Calculates WER and CER for prediction-reference pairs.

    Args:
        predictions (list): Model predictions.
        references (list): Ground truth texts.
    
    Returns:
        dict: Dictionary with 'wer' and 'cer'.
    """
    valid_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
    
    if not valid_pairs:
        return {"wer": 1.0, "cer": 1.0}
    
    predictions_clean, references_clean = zip(*valid_pairs)
    
    predictions_list = list(predictions_clean)
    references_list = list(references_clean)
    
    word_error_rate = asr_utils.wer(references_list, predictions_list)
    char_error_rate = asr_utils.cer(references_list, predictions_list)
    
    return {
        "wer": word_error_rate,
        "cer": char_error_rate
    }

def evaluate_model(eval_dataset, model_path, output_file):
    """
    Evaluates ASR model and writes results to a file.

    Args:
        eval_dataset (Dataset): Dataset with input and label fields.
        model_path (str): Path to the trained model.
        output_file (str): File to save metrics and samples.
    
    Returns:
        tuple: (metrics dict, predictions list, references list)
    """
    asr_model = ChukchiASR(model_path)
    
    predictions = []
    references = []
    
    for i in range(len(eval_dataset)):
        example = eval_dataset[i]
        
        audio = example["input_values"]
        pred = asr_model.transcribe(audio)
        predictions.append(pred)
        
        reference = asr_model.processor.decode([l for l in example["labels"] if l != -100])
        references.append(reference)
    
    metrics = calculate_metrics(predictions, references)
    
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"WER: {metrics['wer']:.3f}\n")
        f.write(f"CER: {metrics['cer']:.3f}\n\n")

        print(f"WER: {metrics['wer']:.3f}")
        print(f"CER: {metrics['cer']:.3f}")
        
        for i in range(len(predictions)):
            f.write(f"True: '{references[i]}'\n")
            f.write(f"Pred: '{predictions[i]}'\n\n")

def simple_transcribe(audio_path, model_path):
    """
    Performs quick transcription of a single audio file.

    Args:
        audio_path (str): Path to audio file.
        model_path (str): Path to the trained model.
    
    Returns:
        str: Transcribed text.
    """
    asr_model = ChukchiASR(model_path)
    transcription = asr_model.transcribe_file(audio_path)
    print(f"Audio: {audio_path}")
    print(f"Transcription: '{transcription}'")
    return transcription