#!/usr/bin/env python3
"""
Inference script for Neural Machine Translation model
"""

import torch
import argparse
import os
import pickle
from typing import List
import time

import config
from src.model import TransformerNMT, BeamSearchDecoder, create_model
from src.dataset import DataPreprocessor
from evaluation import BLEUEvaluator, TranslationEvaluator


class NMTInference:
    """Neural Machine Translation Inference Engine"""
    
    def __init__(self, model_path: str, device: str = 'auto'):
        """Initialize the inference engine"""
        self.device = self._setup_device(device)
        self.config = config
        self.src_vocab, self.tgt_vocab = self._load_vocabularies(os.path.dirname(model_path))
        self.model = self._load_model(model_path)
        
        self.beam_decoder = BeamSearchDecoder(
            self.model,
            beam_size=5,
            max_len=self.config.MAX_LEN,
            sos_idx=self.tgt_vocab.SOS_IDX,
            eos_idx=self.tgt_vocab.EOS_IDX,
            pad_idx=self.tgt_vocab.PAD_IDX
        )
        
        print(f"Model loaded on {self.device}")
        print(f"Source vocabulary size: {len(self.src_vocab)}")
        print(f"Target vocabulary size: {len(self.tgt_vocab)}")
    
    def _setup_device(self, device: str) -> torch.device:
        """Setup computation device"""
        if device == 'auto':
            if torch.cuda.is_available():
                device = 'cuda'
                print(f"Using GPU: {torch.cuda.get_device_name()}")
            else:
                device = 'cpu'
                print("Using CPU")
        
        return torch.device(device)

    def _load_vocabularies(self, path: str) -> tuple:
        """Load source and target vocabularies"""
        src_vocab_path = os.path.join(path, 'src_vocab.pkl')
        tgt_vocab_path = os.path.join(path, 'tgt_vocab.pkl')
        
        if not os.path.exists(src_vocab_path) or not os.path.exists(tgt_vocab_path):
            raise FileNotFoundError("Vocabulary files (src_vocab.pkl, tgt_vocab.pkl) not found.")
            
        with open(src_vocab_path, 'rb') as f:
            src_vocab = pickle.load(f)
        with open(tgt_vocab_path, 'rb') as f:
            tgt_vocab = pickle.load(f)
            
        return src_vocab, tgt_vocab

    def _load_model(self, model_path: str) -> TransformerNMT:
        """Load trained model"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        model = create_model(
            src_vocab_size=len(self.src_vocab),
            tgt_vocab_size=len(self.tgt_vocab),
            d_model=self.config.D_MODEL,
            nhead=self.config.NHEAD,
            num_encoder_layers=self.config.NUM_ENCODER_LAYERS,
            num_decoder_layers=self.config.NUM_DECODER_LAYERS,
            dim_feedforward=self.config.DIM_FEEDFORWARD,
            dropout=0.0,  # No dropout during inference
            max_len=self.config.MAX_LEN,
            device=self.device
        )
        
        # Load state dict
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"Model loaded from epoch {checkpoint['epoch']}")
        print(f"Best BLEU score: {checkpoint.get('bleu_score', 'N/A')}")
        
        return model
    
    def preprocess_text(self, text: str, is_chinese: bool = True) -> str:
        """Preprocess input text"""
        # Basic text cleaning
        text = text.strip()
        
        if is_chinese:
            # For Chinese text, assume it's already segmented
            # In practice, you might want to use jieba or other segmentation tools
            pass
        else:
            # For English text, convert to lowercase
            text = text.lower()
        
        return text
    
    def translate_sentence(self, sentence: str, use_beam_search: bool = True) -> str:
        """Translate a single sentence"""
        # Preprocess
        preprocessor = DataPreprocessor('') # Dummy preprocessor for tokenization
        src_tokens = preprocessor.tokenize_english(sentence)

        # Convert to indices
        src_indices = self.src_vocab.tokens_to_indices(src_tokens)
        src_tensor = torch.tensor(src_indices, dtype=torch.long, device=self.device).unsqueeze(1)
        
        with torch.no_grad():
            if use_beam_search:
                # Use beam search decoding
                src_key_padding_mask = self.model.create_padding_mask(src_tensor, self.src_vocab.PAD_IDX)
                results = self.beam_decoder.decode(src_tensor, src_key_padding_mask=src_key_padding_mask)
                
                if results:
                    translated_indices = results[0].cpu().tolist()
                    translated_text = self.tgt_vocab.indices_to_sentence(translated_indices)
                else:
                    translated_text = ""
            else:
                # Use greedy decoding
                translated_text = self._greedy_decode(src_tensor)
        
        return translated_text
    
    def _greedy_decode(self, src: torch.Tensor) -> str:
        """Greedy decoding for single sentence"""
        batch_size = src.size(1)
        max_len = self.config.MAX_LEN
        
        # Initialize decoder input
        decoder_input = torch.full((1, batch_size), self.tgt_vocab.SOS_IDX, dtype=torch.long, device=self.device)
        
        for step in range(max_len):
            # Create masks
            tgt_mask = self.model.generate_square_subsequent_mask(decoder_input.size(0)).to(self.device)
            src_key_padding_mask = self.model.create_padding_mask(src, self.src_vocab.PAD_IDX)
            tgt_key_padding_mask = self.model.create_padding_mask(decoder_input, self.tgt_vocab.PAD_IDX)
            
            # Forward pass
            output = self.model(
                src=src,
                tgt=decoder_input,
                tgt_mask=tgt_mask,
                src_key_padding_mask=src_key_padding_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=src_key_padding_mask
            )
            
            # Get next token
            next_token = torch.argmax(output[-1, :, :], dim=-1, keepdim=True)
            decoder_input = torch.cat([decoder_input, next_token.unsqueeze(0)], dim=0)
            
            # Check for EOS
            if next_token.item() == self.tgt_vocab.EOS_IDX:
                break
        
        # Convert to text
        translated_indices = decoder_input[1:, 0].cpu().tolist()  # Skip SOS token
        return self.tgt_vocab.indices_to_sentence(translated_indices)
    
    def translate_batch(self, sentences: List[str], use_beam_search: bool = True) -> List[str]:
        """Translate a batch of sentences"""
        results = []
        
        for sentence in sentences:
            translated = self.translate_sentence(sentence, use_beam_search)
            results.append(translated)
        
        return results
    
    def evaluate_on_test_set(self, test_file: str, reference_file: str = None) -> dict:
        """Evaluate model on test set"""
        # Load test sentences
        with open(test_file, 'r', encoding='utf-8') as f:
            test_sentences = [line.strip() for line in f.readlines()]
        
        print(f"Translating {len(test_sentences)} test sentences...")
        
        # Translate
        start_time = time.time()
        translations = []
        
        for i, sentence in enumerate(test_sentences):
            if i % 100 == 0:
                print(f"Translated {i}/{len(test_sentences)} sentences")
            
            translated = self.translate_sentence(sentence)
            translations.append(translated)
        
        translation_time = time.time() - start_time
        
        print(f"Translation completed in {translation_time:.2f} seconds")
        print(f"Average time per sentence: {translation_time/len(test_sentences):.3f} seconds")
        
        # Evaluate if reference is provided
        results = {
            'num_sentences': len(test_sentences),
            'translation_time': translation_time,
            'avg_time_per_sentence': translation_time / len(test_sentences)
        }
        
        if reference_file and os.path.exists(reference_file):
            with open(reference_file, 'r', encoding='utf-8') as f:
                references = [line.strip() for line in f.readlines()]
            
            if len(references) == len(translations):
                evaluator = TranslationEvaluator()
                eval_results = evaluator.evaluate(references, translations)
                results.update(eval_results)
                
                print(f"BLEU-4 Score: {eval_results['bleu_4']:.4f}")
            else:
                print(f"Warning: Number of references ({len(references)}) != translations ({len(translations)})")
        
        return results, translations
    
    def interactive_translation(self):
        """Interactive translation mode"""
        print("=" * 60)
        print("INTERACTIVE TRANSLATION MODE")
        print("Enter Chinese sentences to translate to English")
        print("Type 'quit' or 'exit' to stop")
        print("=" * 60)
        
        while True:
            try:
                # Get input
                chinese_text = input("\n中文输入: ").strip()
                
                if chinese_text.lower() in ['quit', 'exit', '退出']:
                    break
                
                if not chinese_text:
                    continue
                
                # Translate
                start_time = time.time()
                english_text = self.translate_sentence(chinese_text)
                translation_time = time.time() - start_time
                
                # Display result
                print(f"English: {english_text}")
                print(f"Time: {translation_time:.3f}s")
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Translation error: {e}")
        
        print("\nGoodbye!")


def main():
    parser = argparse.ArgumentParser(description='Neural Machine Translation Inference')
    
    parser.add_argument(
        '--model_path', 
        type=str, 
        default='checkpoints/checkpoint_best.pth',
        help='Path to the trained model checkpoint'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='auto',
        choices=['cpu', 'cuda', 'auto'],
        help='Device to use for inference'
    )
    parser.add_argument(
        '--input_text',
        type=str,
        default=None,
        help='A single sentence to translate'
    )
    parser.add_argument(
        '--input_file',
        type=str,
        default=None,
        help='Path to a text file with sentences to translate (one per line)'
    )
    parser.add_argument(
        '--output_file',
        type=str,
        default='translations.txt',
        help='Path to save the translations'
    )
    parser.add_argument(
        '--reference_file',
        type=str,
        default=None,
        help='Path to the reference translations for evaluation'
    )
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode'
    )
    
    args = parser.parse_args()
    
    try:
        inference_engine = NMTInference(args.model_path, args.device)
        
        if args.interactive:
            inference_engine.interactive_translation()
        elif args.input_text:
            translation = inference_engine.translate_sentence(args.input_text)
            print(f"Input:    {args.input_text}")
            print(f"Translated: {translation}")
        elif args.input_file:
            with open(args.input_file, 'r', encoding='utf-8') as f:
                sentences = [line.strip() for line in f]
            
            translations = inference_engine.translate_batch(sentences)
            
            with open(args.output_file, 'w', encoding='utf-8') as f:
                for t in translations:
                    f.write(t + '\n')
            
            print(f"Translations saved to {args.output_file}")
            
            if args.reference_file:
                results, _ = inference_engine.evaluate_on_test_set(args.input_file, args.reference_file)
                print("\nEvaluation Results:")
                for key, value in results.items():
                    if isinstance(value, float):
                        print(f"  {key}: {value:.4f}")
                    else:
                        print(f"  {key}: {value}")

        else:
            print("No action specified. Use --interactive, --input_text, or --input_file.")
            parser.print_help()

    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please ensure the model path and vocabulary files are correct.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main() 