import nltk
import numpy as np
from collections import Counter
from typing import List, Dict, Tuple
import torch


class BLEUEvaluator:
    """BLEU score evaluator for machine translation"""
    
    def __init__(self):
        # Download NLTK data if needed
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
    
    def sentence_bleu(self, reference: List[str], candidate: str, n_gram=4) -> float:
        """Calculate BLEU score for a single sentence"""
        reference_tokens = [ref.split() for ref in reference]
        candidate_tokens = candidate.split()
        
        return self._compute_bleu(reference_tokens, candidate_tokens, n_gram)
    
    def corpus_bleu(self, references: List[List[str]], candidates: List[str], n_gram=4) -> float:
        """Calculate BLEU score for a corpus"""
        if len(references) != len(candidates):
            raise ValueError("Number of references and candidates must match")
        
        total_score = 0.0
        valid_sentences = 0
        
        for ref_list, cand in zip(references, candidates):
            if cand.strip():  # Skip empty candidates
                score = self.sentence_bleu(ref_list, cand, n_gram)
                total_score += score
                valid_sentences += 1
        
        return total_score / valid_sentences if valid_sentences > 0 else 0.0
    
    def _compute_bleu(self, references: List[List[str]], candidate: List[str], n_gram: int) -> float:
        """Compute BLEU score using modified precision and brevity penalty"""
        if not candidate:
            return 0.0
        
        # Calculate precision for each n-gram
        precisions = []
        
        for n in range(1, n_gram + 1):
            # Get n-grams from candidate
            candidate_ngrams = self._get_ngrams(candidate, n)
            if not candidate_ngrams:
                precisions.append(0.0)
                continue
            
            # Get n-grams from all references
            reference_ngrams_list = [self._get_ngrams(ref, n) for ref in references]
            
            # Calculate modified precision
            matches = 0
            total_candidate_ngrams = len(candidate_ngrams)
            
            for ngram in candidate_ngrams:
                # Find maximum count of this n-gram in any reference
                max_ref_count = max([ref_ngrams.get(ngram, 0) for ref_ngrams in reference_ngrams_list])
                # Count in candidate
                candidate_count = candidate_ngrams[ngram]
                # Clipped count
                matches += min(candidate_count, max_ref_count)
            
            precision = matches / total_candidate_ngrams if total_candidate_ngrams > 0 else 0.0
            precisions.append(precision)
        
        # Calculate geometric mean of precisions
        if any(p == 0 for p in precisions):
            return 0.0
        
        geo_mean = np.exp(np.mean([np.log(p) for p in precisions if p > 0]))
        
        # Calculate brevity penalty
        candidate_length = len(candidate)
        closest_ref_length = min([len(ref) for ref in references], 
                                key=lambda x: abs(x - candidate_length))
        
        if candidate_length > closest_ref_length:
            brevity_penalty = 1.0
        else:
            brevity_penalty = np.exp(1 - closest_ref_length / candidate_length)
        
        return brevity_penalty * geo_mean
    
    def _get_ngrams(self, tokens: List[str], n: int) -> Counter:
        """Extract n-grams from token list"""
        ngrams = []
        for i in range(len(tokens) - n + 1):
            ngram = tuple(tokens[i:i+n])
            ngrams.append(ngram)
        return Counter(ngrams)


def compute_bleu_score(references: List[str], hypotheses: List[str]) -> float:
    """Convenient function to compute BLEU-4 score"""
    evaluator = BLEUEvaluator()
    
    # Convert to required format
    ref_lists = [[ref] for ref in references]
    
    return evaluator.corpus_bleu(ref_lists, hypotheses, n_gram=4)


class TranslationEvaluator:
    """Comprehensive translation evaluation including BLEU and other metrics"""
    
    def __init__(self):
        self.bleu_evaluator = BLEUEvaluator()
    
    def evaluate(self, references: List[str], hypotheses: List[str]) -> Dict[str, float]:
        """Evaluate translations using multiple metrics"""
        results = {}
        
        # BLEU scores
        results['bleu_1'] = self.bleu_evaluator.corpus_bleu([[ref] for ref in references], hypotheses, 1)
        results['bleu_2'] = self.bleu_evaluator.corpus_bleu([[ref] for ref in references], hypotheses, 2)
        results['bleu_3'] = self.bleu_evaluator.corpus_bleu([[ref] for ref in references], hypotheses, 3)
        results['bleu_4'] = self.bleu_evaluator.corpus_bleu([[ref] for ref in references], hypotheses, 4)
        
        # Additional metrics
        results['exact_match'] = self._exact_match_score(references, hypotheses)
        results['avg_length_ratio'] = self._avg_length_ratio(references, hypotheses)
        
        return results
    
    def _exact_match_score(self, references: List[str], hypotheses: List[str]) -> float:
        """Calculate exact match score"""
        matches = sum(1 for ref, hyp in zip(references, hypotheses) if ref.strip() == hyp.strip())
        return matches / len(references) if references else 0.0
    
    def _avg_length_ratio(self, references: List[str], hypotheses: List[str]) -> float:
        """Calculate average length ratio (hypothesis/reference)"""
        ratios = []
        for ref, hyp in zip(references, hypotheses):
            ref_len = len(ref.split())
            hyp_len = len(hyp.split())
            if ref_len > 0:
                ratios.append(hyp_len / ref_len)
        
        return np.mean(ratios) if ratios else 0.0


def evaluate_model_predictions(model, data_loader, src_vocab, tgt_vocab, device, max_samples=None):
    """Evaluate model predictions on a dataset"""
    model.eval()
    references = []
    hypotheses = []
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if max_samples and batch_idx * data_loader.batch_size >= max_samples:
                break
            
            src = batch['src'].to(device)
            tgt_output = batch['tgt_output'].to(device)
            
            # Generate predictions (simple greedy decoding)
            batch_size = src.size(1)
            max_len = 100
            
            # Initialize decoder input
            decoder_input = torch.full((1, batch_size), tgt_vocab.SOS_IDX, dtype=torch.long, device=device)
            
            for step in range(max_len):
                # Create masks
                tgt_mask = model.generate_square_subsequent_mask(decoder_input.size(0)).to(device)
                src_key_padding_mask = model.create_padding_mask(src, src_vocab.PAD_IDX)
                tgt_key_padding_mask = model.create_padding_mask(decoder_input, tgt_vocab.PAD_IDX)
                
                # Forward pass
                output = model(
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
                if torch.all(next_token == tgt_vocab.EOS_IDX):
                    break
            
            # Convert to text
            for i in range(batch_size):
                # Hypothesis
                hyp_tokens = decoder_input[1:, i].cpu().tolist()  # Skip SOS token
                hyp_text = tgt_vocab.indices_to_sentence(hyp_tokens)
                hypotheses.append(hyp_text)
                
                # Reference
                ref_tokens = tgt_output[:, i].cpu().tolist()
                ref_text = tgt_vocab.indices_to_sentence(ref_tokens)
                references.append(ref_text)
    
    # Evaluate
    evaluator = TranslationEvaluator()
    results = evaluator.evaluate(references, hypotheses)
    
    return results, references, hypotheses


if __name__ == "__main__":
    # Test BLEU evaluator
    evaluator = BLEUEvaluator()
    
    # Test examples
    references = [["the cat is on the mat"], ["there is a cat on the mat"]]
    candidates = ["the cat is on the mat", "a cat is on the mat"]
    
    # Test sentence BLEU
    score1 = evaluator.sentence_bleu(references[0], candidates[0])
    print(f"Sentence BLEU: {score1:.4f}")
    
    # Test corpus BLEU
    corpus_score = evaluator.corpus_bleu(references, candidates)
    print(f"Corpus BLEU: {corpus_score:.4f}")
    
    # Test comprehensive evaluation
    comp_evaluator = TranslationEvaluator()
    ref_texts = ["the cat is on the mat", "there is a cat on the mat"]
    hyp_texts = ["the cat is on the mat", "a cat is on the mat"]
    
    results = comp_evaluator.evaluate(ref_texts, hyp_texts)
    print("Evaluation results:")
    for metric, score in results.items():
        print(f"  {metric}: {score:.4f}") 