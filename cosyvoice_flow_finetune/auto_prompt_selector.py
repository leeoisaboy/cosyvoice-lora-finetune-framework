#!/usr/bin/env python3
"""
Auto Prompt Selector - Intelligent Reference Audio Selection

Automatically selects the best reference audio based on:
1. Semantic similarity (meaning)
2. Length similarity (character count)
3. Rhythm similarity (pinyin patterns)

Usage:
    from auto_prompt_selector import AutoPromptSelector

    selector = AutoPromptSelector('raw_audio')
    best = selector.select_best("床前明月光")
    print(f"Best match: {best['path']} -> {best['text']}")
"""

import os
import re
import json
import math
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, asdict
from functools import lru_cache

# Optional imports - graceful degradation if not available
try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

try:
    from pypinyin import pinyin, Style
    HAS_PYPINYIN = True
except ImportError:
    HAS_PYPINYIN = False
    print("[AutoPromptSelector] pypinyin not installed, rhythm matching disabled")
    print("  Install with: pip install pypinyin")


@dataclass
class PromptInfo:
    """Reference audio information"""
    path: str           # Full path to wav file
    text: str           # Transcription text
    char_count: int     # Character count
    duration: float     # Audio duration in seconds (estimated)
    pinyin: str         # Pinyin representation (for rhythm matching)
    source: str         # Source poem/text name


class AutoPromptSelector:
    """
    Intelligent reference audio selector for CosyVoice TTS.

    Selects the best matching reference audio based on multiple criteria:
    - Semantic similarity (using sentence embeddings if available)
    - Length matching (character count)
    - Rhythm matching (pinyin patterns)
    """

    def __init__(
        self,
        audio_dir: str,
        metadata_path: Optional[str] = None,
        use_semantic: bool = True,
        cache_embeddings: bool = True,
        min_duration: float = 1.0,
        max_duration: float = 10.0,
    ):
        """
        Initialize the selector.

        Args:
            audio_dir: Directory containing reference audio files
            metadata_path: Path to metadata JSON (auto-generated if None)
            use_semantic: Whether to use semantic similarity (requires sentence-transformers)
            cache_embeddings: Cache sentence embeddings for faster lookup
            min_duration: Minimum audio duration to consider (seconds)
            max_duration: Maximum audio duration to consider (seconds)
        """
        self.audio_dir = Path(audio_dir)
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.use_semantic = use_semantic
        self.cache_embeddings = cache_embeddings

        # Load or generate metadata
        if metadata_path and os.path.exists(metadata_path):
            self.prompts = self._load_metadata(metadata_path)
        else:
            self.prompts = self._scan_audio_files()
            # Auto-save metadata for future use
            auto_metadata_path = self.audio_dir / 'prompt_metadata.json'
            self._save_metadata(auto_metadata_path)
            print(f"[AutoPromptSelector] Saved metadata to {auto_metadata_path}")

        print(f"[AutoPromptSelector] Loaded {len(self.prompts)} reference audios")

        # Initialize semantic model if requested
        self.semantic_model = None
        self.prompt_embeddings = None
        if use_semantic:
            self._init_semantic_model()

    def _parse_filename(self, filepath: Path) -> Optional[PromptInfo]:
        """
        Parse audio filename to extract text content.

        Expected formats:
        - 《诗名》作者_文本内容_序号.wav
        - 文本内容.wav
        - prefix_文本内容_suffix.wav
        """
        filename = filepath.stem

        # Try to extract text from filename patterns
        text = None
        source = ""

        # Pattern 1: 《xxx》author（xxx）_text_number
        match = re.match(r'《(.+?)》.+?[_）](.+?)(?:_\d+)?$', filename)
        if match:
            source = match.group(1)
            text = match.group(2)
            # Clean up text
            text = re.sub(r'_\d+$', '', text)  # Remove trailing numbers

        # Pattern 2: Simple text_number format
        if not text:
            parts = filename.split('_')
            # Find the longest part that looks like text (contains Chinese)
            chinese_parts = [p for p in parts if re.search(r'[\u4e00-\u9fff]', p)]
            if chinese_parts:
                text = max(chinese_parts, key=len)

        # Pattern 3: Just use filename if nothing else works
        if not text:
            text = re.sub(r'[_\d]+$', '', filename)

        if not text or len(text) < 2:
            return None

        # Get pinyin for rhythm matching
        pinyin_str = self._get_pinyin(text) if HAS_PYPINYIN else ""

        # Estimate duration from file size (rough estimate: ~32KB per second for 16kHz mono)
        try:
            file_size = filepath.stat().st_size
            estimated_duration = file_size / 32000  # Very rough estimate
        except:
            estimated_duration = 3.0  # Default estimate

        return PromptInfo(
            path=str(filepath),
            text=text,
            char_count=len(text),
            duration=estimated_duration,
            pinyin=pinyin_str,
            source=source,
        )

    def _get_pinyin(self, text: str) -> str:
        """Get pinyin representation of text for rhythm matching."""
        if not HAS_PYPINYIN:
            return ""
        try:
            # Get pinyin with tone numbers
            py = pinyin(text, style=Style.TONE3, heteronym=False)
            return ' '.join([p[0] for p in py if p])
        except:
            return ""

    def _scan_audio_files(self) -> List[PromptInfo]:
        """Scan audio directory and build metadata."""
        prompts = []

        wav_files = list(self.audio_dir.glob('*.wav'))
        print(f"[AutoPromptSelector] Scanning {len(wav_files)} audio files...")

        for filepath in wav_files:
            info = self._parse_filename(filepath)
            if info:
                prompts.append(info)

        # Sort by text length for consistent ordering
        prompts.sort(key=lambda x: x.char_count)

        return prompts

    def _load_metadata(self, path: str) -> List[PromptInfo]:
        """Load metadata from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return [PromptInfo(**item) for item in data]

    def _save_metadata(self, path: Path):
        """Save metadata to JSON file."""
        data = [asdict(p) for p in self.prompts]
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

    def _init_semantic_model(self):
        """Initialize sentence embedding model for semantic similarity."""
        try:
            from sentence_transformers import SentenceTransformer

            # Use a lightweight Chinese model
            model_name = 'shibing624/text2vec-base-chinese'
            print(f"[AutoPromptSelector] Loading semantic model: {model_name}")

            self.semantic_model = SentenceTransformer(model_name)

            # Pre-compute embeddings for all prompts
            if self.cache_embeddings and self.prompts:
                print("[AutoPromptSelector] Pre-computing embeddings...")
                texts = [p.text for p in self.prompts]
                self.prompt_embeddings = self.semantic_model.encode(
                    texts,
                    convert_to_tensor=HAS_TORCH,
                    show_progress_bar=False
                )
                print(f"[AutoPromptSelector] Cached {len(texts)} embeddings")

        except ImportError:
            print("[AutoPromptSelector] sentence-transformers not installed")
            print("  Semantic similarity disabled. Install with:")
            print("  pip install sentence-transformers")
            self.use_semantic = False
        except Exception as e:
            print(f"[AutoPromptSelector] Failed to load semantic model: {e}")
            self.use_semantic = False

    def _compute_length_score(self, query_len: int, prompt_len: int) -> float:
        """
        Compute length similarity score.

        Prefers prompts with similar character count.
        Score range: 0.0 - 1.0
        """
        diff = abs(query_len - prompt_len)
        # Gaussian-like decay: score = exp(-diff^2 / (2 * sigma^2))
        sigma = max(query_len * 0.3, 3)  # Allow 30% variance
        score = math.exp(-(diff ** 2) / (2 * sigma ** 2))
        return score

    def _compute_rhythm_score(self, query_pinyin: str, prompt_pinyin: str) -> float:
        """
        Compute rhythm similarity based on pinyin patterns.

        Considers:
        - Syllable count similarity
        - Tone pattern similarity

        Score range: 0.0 - 1.0
        """
        if not query_pinyin or not prompt_pinyin:
            return 0.5  # Neutral score if pinyin not available

        query_parts = query_pinyin.split()
        prompt_parts = prompt_pinyin.split()

        # Syllable count similarity
        len_ratio = min(len(query_parts), len(prompt_parts)) / max(len(query_parts), len(prompt_parts), 1)

        # Tone pattern similarity (compare tone numbers)
        query_tones = ''.join([p[-1] if p[-1].isdigit() else '0' for p in query_parts])
        prompt_tones = ''.join([p[-1] if p[-1].isdigit() else '0' for p in prompt_parts])

        # Simple edit distance ratio for tones
        if query_tones and prompt_tones:
            # Levenshtein distance approximation
            min_len = min(len(query_tones), len(prompt_tones))
            max_len = max(len(query_tones), len(prompt_tones))
            matches = sum(1 for i in range(min_len) if query_tones[i] == prompt_tones[i])
            tone_score = matches / max_len if max_len > 0 else 0
        else:
            tone_score = 0.5

        # Combined score
        return 0.6 * len_ratio + 0.4 * tone_score

    def _compute_semantic_score(self, query_embedding, prompt_idx: int) -> float:
        """
        Compute semantic similarity using cosine similarity.

        Score range: 0.0 - 1.0
        """
        if self.prompt_embeddings is None:
            return 0.5

        if HAS_TORCH:
            from sentence_transformers import util
            score = util.cos_sim(query_embedding, self.prompt_embeddings[prompt_idx])
            return float(score[0][0])
        else:
            # Numpy fallback
            import numpy as np
            prompt_emb = self.prompt_embeddings[prompt_idx]
            cos_sim = np.dot(query_embedding, prompt_emb) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(prompt_emb)
            )
            return float(cos_sim)

    def select_best(
        self,
        text: str,
        top_k: int = 1,
        length_weight: float = 0.3,
        rhythm_weight: float = 0.2,
        semantic_weight: float = 0.5,
        min_chars: Optional[int] = None,
        max_chars: Optional[int] = None,
        verbose: bool = False,
    ) -> Dict:
        """
        Select the best matching reference audio for the given text.

        Args:
            text: Input text to synthesize
            top_k: Number of top matches to return (1 = single best)
            length_weight: Weight for length similarity (0-1)
            rhythm_weight: Weight for rhythm similarity (0-1)
            semantic_weight: Weight for semantic similarity (0-1)
            min_chars: Minimum character count for candidates
            max_chars: Maximum character count for candidates
            verbose: Print detailed scoring information

        Returns:
            Dict with 'path', 'text', 'score', and other metadata
            If top_k > 1, returns list of dicts
        """
        if not self.prompts:
            raise ValueError("No prompts available in the library")

        query_len = len(text)
        query_pinyin = self._get_pinyin(text) if HAS_PYPINYIN else ""

        # Get query embedding if semantic model available
        query_embedding = None
        if self.use_semantic and self.semantic_model:
            query_embedding = self.semantic_model.encode(
                text,
                convert_to_tensor=HAS_TORCH,
                show_progress_bar=False
            )

        # Normalize weights
        total_weight = length_weight + rhythm_weight
        if self.use_semantic:
            total_weight += semantic_weight
        else:
            # Redistribute semantic weight if not available
            length_weight += semantic_weight * 0.6
            rhythm_weight += semantic_weight * 0.4
            total_weight = length_weight + rhythm_weight

        length_weight /= total_weight
        rhythm_weight /= total_weight
        semantic_weight = semantic_weight / total_weight if self.use_semantic else 0

        # Filter candidates
        candidates = []
        for i, prompt in enumerate(self.prompts):
            # Apply character count filters
            if min_chars and prompt.char_count < min_chars:
                continue
            if max_chars and prompt.char_count > max_chars:
                continue
            candidates.append((i, prompt))

        if not candidates:
            # Fallback: use all prompts if filters too strict
            candidates = list(enumerate(self.prompts))

        # Score each candidate
        scores = []
        for idx, prompt in candidates:
            # Length score
            len_score = self._compute_length_score(query_len, prompt.char_count)

            # Rhythm score
            rhythm_score = self._compute_rhythm_score(query_pinyin, prompt.pinyin)

            # Semantic score
            if self.use_semantic and query_embedding is not None:
                sem_score = self._compute_semantic_score(query_embedding, idx)
            else:
                sem_score = 0.5

            # Combined score
            final_score = (
                length_weight * len_score +
                rhythm_weight * rhythm_score +
                semantic_weight * sem_score
            )

            scores.append({
                'index': idx,
                'prompt': prompt,
                'score': final_score,
                'length_score': len_score,
                'rhythm_score': rhythm_score,
                'semantic_score': sem_score,
            })

        # Sort by score descending
        scores.sort(key=lambda x: x['score'], reverse=True)

        if verbose:
            print(f"\n[AutoPromptSelector] Query: '{text}' ({query_len} chars)")
            print(f"  Weights: length={length_weight:.2f}, rhythm={rhythm_weight:.2f}, semantic={semantic_weight:.2f}")
            print(f"  Top {min(5, len(scores))} matches:")
            for i, s in enumerate(scores[:5]):
                p = s['prompt']
                print(f"    {i+1}. [{s['score']:.3f}] '{p.text}' ({p.char_count} chars)")
                print(f"       L={s['length_score']:.2f}, R={s['rhythm_score']:.2f}, S={s['semantic_score']:.2f}")

        # Return results
        def format_result(score_info):
            p = score_info['prompt']
            return {
                'path': p.path,
                'text': p.text,
                'score': score_info['score'],
                'char_count': p.char_count,
                'source': p.source,
                'scores': {
                    'length': score_info['length_score'],
                    'rhythm': score_info['rhythm_score'],
                    'semantic': score_info['semantic_score'],
                }
            }

        if top_k == 1:
            return format_result(scores[0])
        else:
            return [format_result(s) for s in scores[:top_k]]

    def get_random(self, n: int = 1) -> List[Dict]:
        """Get random prompts from the library."""
        import random
        selected = random.sample(self.prompts, min(n, len(self.prompts)))
        return [{'path': p.path, 'text': p.text, 'char_count': p.char_count} for p in selected]

    def get_by_length(self, target_len: int, tolerance: int = 3) -> List[Dict]:
        """Get prompts with similar character count."""
        matches = [
            p for p in self.prompts
            if abs(p.char_count - target_len) <= tolerance
        ]
        return [{'path': p.path, 'text': p.text, 'char_count': p.char_count} for p in matches]


# ============================================================
# Command-line interface
# ============================================================

def main():
    """CLI for testing the selector."""
    import argparse

    parser = argparse.ArgumentParser(description='Auto Prompt Selector')
    parser.add_argument('--audio-dir', '-d', type=str, default='raw_audio',
                        help='Directory containing reference audio files')
    parser.add_argument('--text', '-t', type=str, required=True,
                        help='Input text to match')
    parser.add_argument('--top-k', '-k', type=int, default=5,
                        help='Number of top matches to show')
    parser.add_argument('--no-semantic', action='store_true',
                        help='Disable semantic similarity')
    parser.add_argument('--verbose', '-v', action='store_true',
                        help='Show detailed scoring')

    args = parser.parse_args()

    # Initialize selector
    selector = AutoPromptSelector(
        args.audio_dir,
        use_semantic=not args.no_semantic,
    )

    # Select best matches
    results = selector.select_best(
        args.text,
        top_k=args.top_k,
        verbose=args.verbose,
    )

    print(f"\n{'='*60}")
    print(f"Input: '{args.text}'")
    print(f"{'='*60}")

    if isinstance(results, list):
        for i, r in enumerate(results):
            print(f"\n{i+1}. Score: {r['score']:.3f}")
            print(f"   Text: {r['text']}")
            print(f"   Path: {r['path']}")
            print(f"   Chars: {r['char_count']}, Source: {r['source']}")
    else:
        print(f"\nBest match (score: {results['score']:.3f}):")
        print(f"  Text: {results['text']}")
        print(f"  Path: {results['path']}")


if __name__ == '__main__':
    main()
