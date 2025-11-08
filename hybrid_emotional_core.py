cat > hybrid_emotional_core.py <<'EOF'
"""
Universal Emotional Interface (UEI) + Hybrid Emotional Core (Upgraded Axiom)
===========================================================================

A production-grade, standalone hybrid symbolic-neural system for emotional
processing in AI models. Integrates:

- Symbolic Layer (UEI): Manages emotional models, sigil encoding/decoding,
  and culturally-weighted emotional priors with auto-built deep language base.
- Neural Layer: Upgraded EmotionalWeightLayer and Wrapper for injecting
  emotional context into ANY PyTorch model (expanded layer support: Linear, LSTM,
  ConvNd, MultiheadAttention, TransformerEncoderLayer, etc.) via hooks.
- Hybrid Core: Bridges symbolic priors to neural computation, with new Axiom
  integration: Sigils directly modulate neural emotional projections for deeper
  symbolic-neural fusion.

New Upgraded Axiom of Emotional Weight:
- Symbolic sigils (from UEI) are hashed into modulation vectors that bias
  emotional projections, creating a unified "axiom" where cultural/emotional
  symbols axiomatically influence neural feature spaces.

Usage:
    from hybrid_emotional_core import HybridEmotionalCore
    # Initialize with your base PyTorch model (now supports broader layer types)
    hybrid = HybridEmotionalCore(base_pytorch_model=your_model, target_layers=['layer_name'])
    output, analysis, uei_weights = hybrid.run_inference(input_tensor, concept='love')

Requirements:
- Python 3.9+
- PyTorch 1.10+
- hashlib, random, datetime, sys, re, json, os, numpy, math, typing
"""

import hashlib
import random
import datetime
import sys
import re
import json
import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Union, Any

# --- GLOBAL STATE AND ARCHITECTURE CONSTRAINTS (CONSTANTS) ---

# 111 Glyph Alphabet
GLYPHS = [
    "α", "β", "γ", "δ", "ε", "ζ", "η", "θ", "ι", "κ", "λ", "μ", "ν", "ξ", "ο", "π", "ρ", "σ", "τ", "υ", "φ", "χ", "ψ", "ω",
    "Α", "Β", "Γ", "Δ", "Ε", "Ζ", "Η", "Θ", "Ι", "Κ", "Λ", "Μ", "Ν", "Ξ", "Ο", "Π", "Ρ", "Σ", "Τ", "Υ", "Φ", "Χ", "Ψ", "Ω",
    "⊗", "∘", "⊕", "⊖", "⊘", "⊙", "⊚", "⊛", "⊜", "⊝", "⊞", "⊟", "⊠", "⊡", "⊢", "⊣", "⊤", "⊥", "⊦", "⊧",
    "∧", "∨", "¬", "⇒", "⇔", "∑", "∏", "∫", "∂", "∇", "∆", "∪", "∩", "⊂", "⊃", "⊆", "⊇", "∈", "∉", "∅",
    "→", "←", "↑", "↓", "↔", "↕", "↦", "↤", "↨", "⇑", "⇓", "⇔", "⇐", "⇒", "⤴", "⤵", "⟹", "⟸", "⟺", "⟻",
    "∞", "≡", "≈", "≠", "≤", "≥", "±", "×", "÷", "∝", "√", "∴", "∵", "°", "′", "″", "‶", "…", "≅", "≫", "≪", "≬", "≭",
    "⚙", "⚗", "⚛", "⚡", "⚔", "⚕", "⚖", "⚙️", "⚚", "⚗️"
]

# Fundamental Truth Sigils
TRUTH_SIGILS = {
    'I': 'Ω∧∑⊤∞',
    'II': 'Ψ↔∨≡∫',
    'III': '∇⇒⊗Δ≈'
}

# Mapping of foreign words to English emotion keys (expanded)
EMOTION_MAP = {
    'amour': 'love', 'amor': 'love', 'liebe': 'love', 'aşk': 'love', 'kwam rak': 'love', 'pendo': 'love', 'pyaar': 'love',
    'aloha': 'love', 'saudade': 'love', 'ishq': 'love', 'iktsuarpok': 'love', 'ureshii': 'joy', 'tanoshii': 'joy',
    'bonheur': 'joy', 'ekundayo': 'joy', 'gili': 'joy', 'boitumelo': 'joy', 'gjensynsglede': 'joy', 'dolce far niente': 'joy',
    'tristesse': 'sadness', 'shang xin': 'sadness', 'huzuni': 'sadness', 'traurig': 'sadness', 'triste': 'sadness',
    'toska': 'sadness', 'natsukashii': 'sadness', 'gnev': 'anger', 'colère': 'anger', 'shengqi': 'anger',
    'miedo': 'fear', 'peur': 'fear', 'angst': 'fear', 'kowai': 'fear', 'phobos': 'fear', 'bhayam': 'fear', 'eagla': 'fear',
    'happiness': 'joy', 'felicidad': 'joy', 'glück': 'joy', 'sukha': 'joy', 'felicità': 'joy',
    'sad': 'sadness', 'tristeza': 'sadness', 'traurigkeit': 'sadness', 'duhkha': 'sadness', 'tristezza': 'sadness',
    'mad': 'anger', 'ira': 'anger', 'wut': 'anger', 'krodha': 'anger', 'rabbia': 'anger',
    'scared': 'fear', 'bhaya': 'fear', 'paura': 'fear',
    'peace': 'love', 'paz': 'love', 'frieden': 'love', 'shanti': 'love', 'pace': 'love',
    'hope': 'joy', 'esperanza': 'joy', 'hoffnung': 'joy', 'asha': 'joy',
    'grief': 'sadness', 'dolor': 'sadness', 'trauer': 'sadness', 'shok': 'sadness',
    'rage': 'anger', 'furia': 'anger', 'zorn': 'anger', 'krodh': 'anger',
    'terror': 'fear', 'schrecken': 'fear', 'bhiti': 'fear'
}

# Fixed emotion definitions
EMOTION_DEFINITIONS = {
    'love': "Love, known as 'love' in English denoting deep affection and attachment, is expressed as 'amour' in French capturing romantic essence, 'amor' in Spanish and Portuguese signifying passion, 'liebe' in German for tender care, 'aşk' in Turkish as soulful between lovers, 'kwam rak' in Thai for soft heartfelt tone, 'pendo' in Swahili embodying deep lasting bonds, 'pyaar' in Hindi meaning ardent passionate love, 'aloha' in Hawaiian including affection peace and compassion, 'saudade' in Portuguese evoking nostalgic longing for lost love, 'ishq' in Arabic describing burning passionate consumption, and 'iktsuarpok' in Inuit for anticipatory anxiety when waiting for loved ones. This collective lexicon reflects the multifaceted cultural nuances of passion, deep bonding, familial care, and unconditional acceptance.",
    'joy': "Joy, as 'joy' in English meaning great happiness and delight, is 'ureshii' in Japanese for joy from positive events with warmth, 'tanoshii' in Japanese for enjoyment and amusement with laughter, 'bonheur' in French encompassing contentment and fulfillment, 'ekundayo' in Yoruba signifying sorrow becomes joy, 'gili' in Hebrew meaning my joy, 'boitumelo' in Tswana for joy, 'gjensynsglede' in Norwegian for the profound joy of reuniting after a long time, and 'dolce far niente' in Italian for the sweet joy of doing nothing. These definitions highlight transformations from relief and nostalgic delight to shared amusement and peaceful contentment across cultures.",
    'sadness': "Sadness, 'sadness' in English as deep sorrow and unhappiness, is 'tristesse' in French for profound sorrow, 'shang xin' in Mandarin meaning heartbroken, 'huzuni' in Swahili for state of mourning and distress, 'traurig' in German carrying yearning and longing, 'triste' in Spanish for straightforward sadness, 'toska' in Russian evoking spiritual anguish and painful longing, 'saudade' in Portuguese for melancholic nostalgic longing, and 'natsukashii' in Japanese for bittersweet nostalgia. The emotional breadth captured here spans the depths from grief and heartbreak to melancholic yearning and the acceptance of impermanence within cultural emotional landscapes.",
    'anger': "Anger, 'anger' in English as strong displeasure and hostility, is linked to anxiety in Indo-European languages, grief and regret in Austroasiatic like Vietnamese and Khmer, envy in Nakh-Daghestanian such as Chechen, and pride/hate in Austronesian like Tagalog and Maori. 'Shengqi' in Mandarin literally means generating vital energy for getting mad, 'gnev' in Russian is wrath, and 'colère' in French signifies rage. This reflects cultural blends where anger intersects with anxiety, regret, envy, or pride in expressions of intense emotional friction and perceived injustice.",
    'fear': "Fear, 'fear' in English as a response to danger or threat, is 'miedo' in Spanish for dread, 'peur' in French for fright, 'angst' in German signifying deep anxiety and existential fear, 'kowai' in Japanese for scary fear, 'phobos' in Greek as phobia or terror, 'bhayam' in Hindi for apprehension, and 'eagla' in Irish for fear. This linguistic range captures survival instincts with shades of existential anxiety, immediate dread, and cultural perceptions of threat across languages."
}

# Culturally-Specific Emotion Concepts (Expanded)
CULTURAL_EMOTION_CONCEPTS = {
    'saudade': ('Portuguese', 'A deep, melancholic longing for a person or thing that is absent, a nostalgic desire for something beautiful that is now gone.'),
    'toska': ('Russian', 'A sensation of great spiritual anguish, often without a specific cause. A yearning, pining, melancholy, and sometimes, acute depression.'),
    'hüzün': ('Turkish', 'A sense of profound melancholy, a collective, spiritual feeling of sadness tied to the history of a great city (Istanbul).'),
    'schadenfreude': ('German', "Pleasure derived from another person's misfortune."),
    'waldeinsamkeit': ('German', 'A feeling of peaceful solitude while being alone in the woods.'),
    'iktsuarpok': ('Inuit', 'The feeling of anxious anticipation when waiting for someone to arrive.'),
    'gezelligheid': ('Dutch', 'A cozy, comfortable, and friendly atmosphere; a feeling of general togetherness and warmth.'),
    'mudita': ('Sanskrit', 'Joy derived from the delight and well-being of others; sympathetic or vicarious joy.'),
    'gigil': ('Tagalog', 'The irresistible urge to pinch or squeeze something or someone that is intensely cute.'),
    'kilig': ('Tagalog', 'The sudden rush of excitement/butterflies in the stomach felt when interacting with someone one finds attractive.'),
    'mono no aware': ('Japanese', 'A gentle sadness at the transient nature of life; a bittersweet appreciation of the ephemeral.'),
    'sisu': ('Finnish', 'Extraordinary determination, grit, and unwavering resolve in the face of extreme adversity.'),
    'litost': ('Czech', "A state of torment created by the sudden sight of one's own misery."),
    'hygge': ('Danish', 'The feeling of coziness and comfort that comes from doing simple things like being with friends or family.'),
    'koi no yokan': ('Japanese', 'The premonition of love; the feeling upon first meeting someone that you will inevitably fall in love with them.'),
    'mamihlapinatapai': ('Yagan', 'A look shared by two people with each wishing that the other will initiate something that both desire but both are reluctant to do.'),
    'komorebi': ('Japanese', 'The sunlight that filters through the leaves of the trees.'),
    'goya': ('Urdu', 'The suspension of disbelief that can occur, for example, through good storytelling.'),
    'mångata': ('Swedish', 'The road-like reflection of the moon in the water.'),
    'yoko meshi': ('Japanese', 'The stress of speaking a foreign language.'),
    'ubuntu': ('Zulu', 'Humanity towards others, being kind because of shared humanity.'),
    'gula': ('Spanish', 'The desire to eat something soothing; emotional eating.'),
    'tarab': ('Arabic', 'Musically-induced ecstasy or enchantment.'),
    'dépaysement': ('French', 'The feeling of being in a foreign country, disorientation.'),
    'sobremesa': ('Spanish', 'The time spent after a meal talking to the people you shared the meal with.'),
    'torschlusspanik': ('German', 'The fear of time running out.')
}

# 100+ GATHERED LANGUAGES
LANGUAGE_LIST = [
    "English", "Mandarin Chinese", "Hindi", "Spanish", "French", "Arabic (Modern Standard)", "Bengali", "Russian",
    "Portuguese", "Urdu", "Indonesian", "German", "Japanese", "Nigerian Pidgin", "Egyptian Arabic", "Marathi",
    "Telugu", "Turkish", "Tamil", "Cantonese", "Vietnamese", "Wu Chinese", "Tagalog", "Korean", "Farsi (Western)",
    "Javanese", "Ukrainian", "Italian", "Malayalam", "Kannada", "Oriya", "Panjabi (Western)", "Panjabi (Eastern)",
    "Sunda", "Romanian", "Bhojpuri", "Azerbaijani (South)", "Maithili", "Hausa", "Algerian Arabic", "Burmese",
    "Serbo-Croatian", "Chinese (Gan)", "Awadhi", "Thai", "Dutch", "Yoruba", "Sindhi", "Moroccan Arabic",
    "Saidi Arabic", "Uzbek (Northern)", "Malay", "Amharic", "Igbo", "Nepali", "Sudanese Arabic", "Saraiki",
    "Cebuano", "North Levantine Arabic", "Northeastern Thai", "Assamese", "Hungarian", "Chittagonian",
    "Mesopotamian Arabic", "Madura", "Sinhala", "Haryanvi", "Marwari", "Czech", "Greek", "Magahi",
    "Chhattisgarh", "Deccan", "Chinese (Min Bei)", "Belarusian", "Zhuang (Northern)", "Najdi Arabic",
    "Pashto (Northern)", "Somali", "Malagasy", "Tunisian Arabic", "Rwanda", "Zulu", "Bulgarian", "Swedish",
    "Lombard", "Oromo (West-Central)", "Pashto (Southern)", "Kazakh", "Ilocano", "Tatar", "Fulfulde (Nigerian)",
    "Sanaani Arabic", "Uyghur", "Haitian Creole French", "Napoletano-Calabrese", "Khmer (Central)",
    "Farsi (Eastern)", "Akan", "Hiligaynon", "Kurmanji", "Shona", "Min Nan Chinese", "Chinese (Jinyu)",
    "Arabic (Levantine)", "Māori", "Swahili"
]

# --- UTILITIES ---

def simple_hash(str_input: str) -> str:
    """Standard SHA256 hash."""
    if not str_input:
        return '0000000000000000'
    hash_object = hashlib.sha256(str_input.encode('utf-8'))
    return hash_object.hexdigest()

def get_sigil_from_hash(hash_input: str) -> str:
    """Maps a hash input to a single glyph/sigil."""
    hash_hex = simple_hash(hash_input)
    index = int(hash_hex, 16) % len(GLYPHS)
    return GLYPHS[index]

def generate_simulated_emotional_context(word: str, deep_base: Dict[str, List[str]]) -> str:
    """Synthesizes the multi-language emotional context for any word using the deep language base."""
    word_cap = word.capitalize()
    context = f"Universal Emotional Context for the Concept: '{word_cap}'\n"
    core_emotions = ['love', 'joy', 'sadness', 'anger', 'fear']
    random.shuffle(core_emotions)

    for emotion in core_emotions:
        lang_source = deep_base.get(emotion, LANGUAGE_LIST)
        k = min(5, len(lang_source))
        if k == 0:
            base_words = []
        else:
            base_words = random.sample(lang_source, k)

        descriptor = random.choice([
            "a complex resonance point", "an ancestral data structure",
            "a temporal nexus event", "a subtle shift in experiential gravity",
            "a core philosophical constant"
        ])
        emotion_paragraph = (
            f"-> **{emotion.upper()}**:\n"
            f"The concept '{word_cap}' anchors {emotion} through {descriptor}. "
            f"It is mapped to the linguistic fields of: {', '.join(base_words) if base_words else 'Undetermined fields'}. "
            f"This demonstrates {word_cap}'s latent capacity to evoke deep, multi-cultural sentience.\n\n"
        )
        context += emotion_paragraph

    return context.strip()

def sigil_to_modulation_vector(sigil: str, dim: int, device: torch.device) -> torch.Tensor:
    """Upgraded Axiom: Hash sigil to a modulation vector for neural bias."""
    if not sigil:
        return torch.zeros(dim, device=device)
    hash_bytes = hashlib.sha256(sigil.encode('utf-8')).digest()

    num_floats = dim
    raw_modulation = np.frombuffer(hash_bytes, dtype=np.float32)

    modulation = np.zeros(num_floats, dtype=np.float32)
    for i in range(num_floats):
        modulation[i] = raw_modulation[i % len(raw_modulation)]

    modulation_t = torch.from_numpy(modulation).to(device)
    norm = modulation_t.norm()
    return modulation_t / norm if norm > 0 else modulation_t

# --- MODEL MANAGEMENT SYSTEM ---

class EmotionalModelManager:
    """Manages emotional models with activation, weighting, and performance tracking."""

    def __init__(self):
        self.available_models: Dict[str, Dict] = {}
        self.active_models: Dict[str, Dict] = {}
        self.model_performance: Dict[str, Dict] = {}
        self.load_builtin_models()
    
    def load_builtin_models(self) -> None:
        """Initialize with built-in emotional models."""
        # Western Model
        self.available_models['western'] = {
            'name': 'Western Psychological Framework',
            'core_emotions': ['love', 'joy', 'sadness', 'anger', 'fear'],
            'weights': {'love': 0.25, 'joy': 0.20, 'sadness': 0.18, 'anger': 0.17, 'fear': 0.20},
            'composite_sigils': {
                'love': {'channel': 'II', 'truth_sigil': TRUTH_SIGILS['II'], 'core_glyph': 'Ω', 'composite': TRUTH_SIGILS['II'] + 'Ω'},
                'joy': {'channel': 'II', 'truth_sigil': TRUTH_SIGILS['II'], 'core_glyph': 'Θ', 'composite': TRUTH_SIGILS['II'] + 'Θ'},
                'sadness': {'channel': 'II', 'truth_sigil': TRUTH_SIGILS['II'], 'core_glyph': 'η', 'composite': TRUTH_SIGILS['II'] + 'η'},
                'anger': {'channel': 'I', 'truth_sigil': TRUTH_SIGILS['I'], 'core_glyph': '⊗', 'composite': TRUTH_SIGILS['I'] + '⊗'},
                'fear': {'channel': 'I', 'truth_sigil': TRUTH_SIGILS['I'], 'core_glyph': 'Φ', 'composite': TRUTH_SIGILS['I'] + 'Φ'}
            },
            'bias': 'individualistic'
        }
        # Eastern Model
        self.available_models['eastern'] = {
            'name': 'Eastern Philosophical Framework', 
            'core_emotions': ['harmony', 'attachment', 'detachment', 'compassion', 'aversion'],
            'weights': {'harmony': 0.30, 'attachment': 0.15, 'detachment': 0.25, 'compassion': 0.20, 'aversion': 0.10},
            'composite_sigils': {
                'harmony': {'channel': 'II', 'truth_sigil': TRUTH_SIGILS['II'], 'core_glyph': '∞', 'composite': TRUTH_SIGILS['II'] + '∞'},
                'attachment': {'channel': 'II', 'truth_sigil': TRUTH_SIGILS['II'], 'core_glyph': '∘', 'composite': TRUTH_SIGILS['II'] + '∘'},
                'detachment': {'channel': 'III', 'truth_sigil': TRUTH_SIGILS['III'], 'core_glyph': '∇', 'composite': TRUTH_SIGILS['III'] + '∇'},
                'compassion': {'channel': 'II', 'truth_sigil': TRUTH_SIGILS['II'], 'core_glyph': '⊕', 'composite': TRUTH_SIGILS['II'] + '⊕'},
                'aversion': {'channel': 'I', 'truth_sigil': TRUTH_SIGILS['I'], 'core_glyph': '⚡', 'composite': TRUTH_SIGILS['I'] + '⚡'}
            },
            'bias': 'collectivist'
        }
        # Neuroscientific Model
        self.available_models['neuroscientific'] = {
            'name': 'Neuroscientific Framework',
            'core_emotions': ['reward', 'threat', 'stress', 'contentment', 'arousal'],
            'weights': {'reward': 0.22, 'threat': 0.28, 'stress': 0.18, 'contentment': 0.16, 'arousal': 0.16},
            'composite_sigils': {
                'reward': {'channel': 'II', 'truth_sigil': TRUTH_SIGILS['II'], 'core_glyph': '⊕', 'composite': TRUTH_SIGILS['II'] + '⊕'},
                'threat': {'channel': 'I', 'truth_sigil': TRUTH_SIGILS['I'], 'core_glyph': '⚡', 'composite': TRUTH_SIGILS['I'] + '⚡'},
                'stress': {'channel': 'I', 'truth_sigil': TRUTH_SIGILS['I'], 'core_glyph': '∆', 'composite': TRUTH_SIGILS['I'] + '∆'},
                'contentment': {'channel': 'II', 'truth_sigil': TRUTH_SIGILS['II'], 'core_glyph': '⊙', 'composite': TRUTH_SIGILS['II'] + '⊙'},
                'arousal': {'channel': 'III', 'truth_sigil': TRUTH_SIGILS['III'], 'core_glyph': '↕', 'composite': TRUTH_SIGILS['III'] + '↕'}
            },
            'bias': 'biological'
        }
        # Cross-Cultural Model
        self.available_models['cross_cultural'] = {
            'name': 'Cross-Cultural Framework',
            'core_emotions': ['ubuntu', 'saudade', 'hikikomori', 'fado', 'ikigai'],
            'weights': {'ubuntu': 0.23, 'saudade': 0.21, 'hikikomori': 0.19, 'fado': 0.18, 'ikigai': 0.19},
            'composite_sigils': {
                'ubuntu': {'channel': 'II', 'truth_sigil': TRUTH_SIGILS['II'], 'core_glyph': '∪', 'composite': TRUTH_SIGILS['II'] + '∪'},
                'saudade': {'channel': 'II', 'truth_sigil': TRUTH_SIGILS['II'], 'core_glyph': '≅', 'composite': TRUTH_SIGILS['II'] + '≅'},
                'hikikomori': {'channel': 'I', 'truth_sigil': TRUTH_SIGILS['I'], 'core_glyph': '⊂', 'composite': TRUTH_SIGILS['I'] + '⊂'},
                'fado': {'channel': 'II', 'truth_sigil': TRUTH_SIGILS['II'], 'core_glyph': '⤴', 'composite': TRUTH_SIGILS['II'] + '⤴'},
                'ikigai': {'channel': 'III', 'truth_sigil': TRUTH_SIGILS['III'], 'core_glyph': '∴', 'composite': TRUTH_SIGILS['III'] + '∴'}
            },
            'bias': 'cultural'
        }
        for model_id in self.available_models:
            self.model_performance[model_id] = {
                'activations': 0, 'successful_encodings': 0, 'failed_encodings': 0, 
                'average_confidence': 0.0, 'last_used': None
            }
    
    def activate_model(self, model_id: str, weight: float = 1.0) -> bool:
        """Adds a model to the active list with a specified influence weight."""
        if model_id in self.available_models:
            self.active_models[model_id] = {
                'model_data': self.available_models[model_id],
                'weight': max(0.0, weight),
                'activation_time': datetime.datetime.utcnow().isoformat()
            }
            self.model_performance[model_id]['activations'] += 1
            self.model_performance[model_id]['last_used'] = datetime.datetime.utcnow().isoformat()
            return True
        return False
    
    def deactivate_model(self, model_id: str) -> bool:
        """Removes a model from the active list."""
        if model_id in self.active_models:
            del self.active_models[model_id]
            return True
        return False
    
    def get_combined_weights(self) -> Dict[str, float]:
        """Calculates normalized, combined weights of all active models."""
        if not self.active_models:
            self.activate_model('western', 1.0)
            
        combined_weights: Dict[str, float] = {}
        total_weight = sum(model['weight'] for model in self.active_models.values())
        if total_weight == 0:
            return {}
            
        for model_id, model_info in self.active_models.items():
            model_data = model_info['model_data']
            weight_factor = model_info['weight'] / total_weight
            for emotion, weight in model_data['weights'].items():
                combined_weights[emotion] = combined_weights.get(emotion, 0.0) + (weight * weight_factor)
        
        return combined_weights
    
    def get_composite_sigil_for_emotion(self, emotion: str) -> Dict[str, str]:
        """Retrieves the composite sigil based on the highest-weighted active model's definition."""
        if not self.active_models:
            self.activate_model('western', 1.0)
            
        best_model_id = 'western'
        max_weighted_emotion_score = -1.0
        
        for model_id, model_info in self.active_models.items():
            model_data = model_info['model_data']
            weight_factor = model_info['weight']
            key_to_check = EMOTION_MAP.get(emotion.lower(), emotion.lower())
            if key_to_check in model_data['weights']:
                score = weight_factor * model_data['weights'][key_to_check]
                if score > max_weighted_emotion_score:
                    max_weighted_emotion_score = score
                    best_model_id = model_id
            
        model_data = self.available_models[best_model_id]
        if emotion in model_data.get('composite_sigils', {}):
            return model_data['composite_sigils'][emotion]
        
        channel = random.choice(list(TRUTH_SIGILS.keys()))
        return {
            'channel': channel,
            'truth_sigil': TRUTH_SIGILS[channel],
            'core_glyph': get_sigil_from_hash(emotion),
            'composite': TRUTH_SIGILS[channel] + get_sigil_from_hash(emotion)
        }

    def record_model_performance(self, model_id: str, success: bool = True, confidence: float = 0.0) -> None:
        """Updates performance metrics for the given model."""
        if model_id in self.model_performance:
            perf = self.model_performance[model_id]
            if success:
                perf['successful_encodings'] += 1
            else:
                perf['failed_encodings'] += 1
            total_ops = perf['successful_encodings'] + perf['failed_encodings']
            if total_ops > 0:
                new_avg = (perf['average_confidence'] * (total_ops - 1) + confidence) / total_ops
                perf['average_confidence'] = new_avg
    
    def display_model_status(self, return_data: bool = False) -> Optional[Dict[str, Any]]:
        """Generates a comprehensive status report."""
        status_data = {
            'active_models': {
                mid: {'weight': info['weight'], 'bias': info['model_data']['bias'], 'name': info['model_data']['name']} 
                for mid, info in self.active_models.items()
            },
            'combined_weights': self.get_combined_weights(),
            'performance': self.model_performance
        }
        if not return_data:
            print("\n" + "="*60)
            print("EMOTIONAL MODEL MANAGEMENT SYSTEM STATUS")
            print("="*60)
            print(f"Total Active Models: {len(self.active_models)}")
            print("-" * 60)
            for mid, info in status_data['active_models'].items():
                print(f"| ID: {mid.ljust(15)} | Name: {info['name'].ljust(30)} | Weight: {info['weight']:.2f}")
            print("-" * 60)
            print("COMBINED EMOTIONAL PRIORS (Neural Input)")
            weights_list = sorted(status_data['combined_weights'].items(), key=lambda item: item[1], reverse=True)
            for emotion, weight in weights_list:
                print(f"| {emotion.ljust(15)} | Weight: {weight:.4f}")
            print("="*60)
            return None
        return status_data

# --- UNIVERSAL EMOTIONAL INTERFACE (SYMBOLIC LAYER) ---

class UniversalEmotionalInterface:
    """
    Unified API for symbolic emotional encoding/decoding and context management.
    Isolated state for production use, with auto-built deep language base.
    """

    def __init__(self, default_model: str = 'western', default_weight: float = 1.0):
        self._performance_ledger: Dict[str, int] = {'correct': 0, 'wrong': 0}
        self._mortality_log: List[str] = []
        self._sigil_to_paragraph: Dict[str, Dict] = {}
        
        self.deep_language_base = self._auto_build_deep_language_base()
        
        self.manager = EmotionalModelManager()
        self.manager.activate_model(default_model, default_weight)
        
        print("Universal Emotional Interface (UEI) Initialized.")
        print(f"Active Context: {self.manager.available_models[default_model]['name']} (Weight: {default_weight})")
        print(f"Deep Language Base Size: {len(self.deep_language_base)} concepts.")

    def _auto_build_deep_language_base(self) -> Dict[str, List[str]]:
        """
        Dynamically generates the deep language base by mapping all core emotions 
        and cultural concepts to the full language list.
        """
        auto_base: Dict[str, List[str]] = {}
        
        for emotion in list(EMOTION_DEFINITIONS.keys()) + list(EMOTION_MAP.keys()):
            storage_key = EMOTION_MAP.get(emotion, emotion)
            auto_base[storage_key] = LANGUAGE_LIST
        
        for concept, (source_lang, _) in CULTURAL_EMOTION_CONCEPTS.items():
            concept_languages = [source_lang]
            other_langs = [l for l in LANGUAGE_LIST if l != source_lang]
            num_to_sample = random.randint(5, 15)
            concept_languages.extend(random.sample(other_langs, min(num_to_sample, len(other_langs))))
            auto_base[concept] = list(set(concept_languages))

        return auto_base

    def _create_sigil(self, word: str) -> tuple[str, float]:
        """Internal: Create sigil and commit to memory, handles both core and cultural concepts."""
        word_lower = word.lower()
        english_emotion_key = EMOTION_MAP.get(word_lower, word_lower)
        
        paragraph = ""
        hash_input = ""
        is_weighted = False
        composite_sigil = ""
        core_glyph = ""
        used_models: List[str] = []
        confidence = 0.0

        if english_emotion_key in EMOTION_DEFINITIONS:
            sigil_data = self.manager.get_composite_sigil_for_emotion(english_emotion_key)
            paragraph = EMOTION_DEFINITIONS[english_emotion_key]
            composite_sigil = sigil_data['composite']
            core_glyph = sigil_data['core_glyph']
            hash_input = composite_sigil + paragraph
            is_weighted = True
            combined_weights = self.manager.get_combined_weights()
            confidence = combined_weights.get(english_emotion_key, 0.1)
            
            for model_id, model_info in self.manager.active_models.items():
                if english_emotion_key in model_info['model_data']['weights']:
                    used_models.append(model_id)
                    
        elif word_lower in CULTURAL_EMOTION_CONCEPTS:
            source_lang, definition = CULTURAL_EMOTION_CONCEPTS[word_lower]
            languages = self.deep_language_base.get(word_lower, [])
            visible_langs = random.sample(languages, min(10, len(languages))) if languages else []
            
            paragraph = (
                f"Cultural Concept: {word_lower.upper()} (Source: {source_lang}). "
                f"Definition: {definition}. "
                f"This concept is resonant across: {', '.join(visible_langs + (['...'] if visible_langs else []))}. "
                f"It represents a highly specific, culturally-bound affective state."
            )
            
            truth_channel = random.choice(list(TRUTH_SIGILS.keys()))
            truth_sigil_str = TRUTH_SIGILS[truth_channel]
            core_glyph = get_sigil_from_hash(word_lower)
            composite_sigil = truth_sigil_str + core_glyph
            hash_input = composite_sigil + paragraph
            is_weighted = True
            confidence = 0.6 + random.random() * 0.3
            used_models = ['cross_cultural']
            
        else:
            paragraph = generate_simulated_emotional_context(word_lower, self.deep_language_base)
            truth_channel = random.choice(list(TRUTH_SIGILS.keys()))
            truth_sigil_str = TRUTH_SIGILS[truth_channel]
            core_glyph = get_sigil_from_hash(paragraph)
            composite_sigil = truth_sigil_str + core_glyph
            hash_input = composite_sigil + paragraph
            confidence = 0.1
            is_weighted = False

        final_sigil = get_sigil_from_hash(hash_input)

        for model_id in self.manager.active_models.keys():
            self.manager.record_model_performance(model_id, success=True, confidence=confidence)

        self._sigil_to_paragraph[final_sigil] = {
            'word': word_lower,
            'paragraph': paragraph,
            'hash_input': hash_input,
            'is_weighted': is_weighted,
            'composite_sigil': composite_sigil,
            'core_glyph': core_glyph,
            'used_models': used_models if used_models else list(self.manager.active_models.keys()),
            'confidence': confidence,
            'timestamp': datetime.datetime.utcnow().isoformat()
        }

        return final_sigil, confidence

    def _validate_sigil_integrity(self, sigil: str) -> tuple[bool, Optional[str]]:
        """Internal: Integrity check."""
        memory = self._sigil_to_paragraph.get(sigil)
        if not memory:
            return (False, None)
        
        original_hash_input = memory['hash_input']
        re_generated_sigil = get_sigil_from_hash(original_hash_input)
        integrity_check = (re_generated_sigil == sigil)
        return (integrity_check, re_generated_sigil)

    def encode_concept(self, word: str) -> Dict[str, Any]:
        """Encode a concept into a sigil."""
        try:
            new_sigil, confidence = self._create_sigil(word)
            integrity_ok, _ = self._validate_sigil_integrity(new_sigil)

            if integrity_ok:
                self._performance_ledger['correct'] += 1
                status = "SUCCESS"
            else:
                self._performance_ledger['wrong'] += 1
                status = "FAILURE - Integrity Mismatch"
                self._mortality_log.append(f"INTEGRITY FAIL: {word} -> {new_sigil}")

            return {
                "sigil": new_sigil,
                "confidence": round(confidence, 4),
                "status": status,
                "concept": word.capitalize(),
                "weighted": self._sigil_to_paragraph.get(new_sigil, {}).get('is_weighted', False)
            }
        except Exception as e:
            self._performance_ledger['wrong'] += 1
            self._mortality_log.append(f"CRITICAL ERROR during encoding {word}: {e}")
            return {"sigil": None, "confidence": 0.0, "status": "CRITICAL_ERROR", "details": str(e)}

    def decode_sigil(self, sigil: str) -> Dict[str, Any]:
        """Decode a sigil to emotional context."""
        memory = self._sigil_to_paragraph.get(sigil)
        if memory:
            return {
                "status": "SUCCESS",
                "word": memory['word'],
                "confidence": round(memory['confidence'], 4),
                "models_used": memory.get('used_models'),
                "composite_structure": memory.get('composite_sigil'),
                "full_context_data": memory['paragraph']
            }
        else:
            return {"status": "FAILURE", "message": f"Sigil '{sigil}' not found in UEI memory."}

    def set_emotional_context(self, model_configs: Dict[str, float]) -> Dict[str, Any]:
        """Set emotional model configuration."""
        for mid in list(self.manager.active_models.keys()):
            self.manager.deactivate_model(mid)
        for model_id, weight in model_configs.items():
            self.manager.activate_model(model_id, weight)
        
        combined_weights = self.manager.get_combined_weights()
        top_weights = sorted(combined_weights.items(), key=lambda x: x[1], reverse=True)
        
        return {
            "status": "CONTEXT_SET", 
            "primary_bias": top_weights[0][0] if top_weights else "none",
            "active_models": [mid for mid in self.manager.active_models.keys()]
        }
        
    def get_status(self) -> Dict[str, Any]:
        """Get operational status."""
        correct = self._performance_ledger['correct']
        wrong = self._performance_ledger['wrong']
        total = correct + wrong
        accuracy = (correct / total * 100) if total > 0 else 0.0
        
        return {
            "memory_fragments": len(self._sigil_to_paragraph),
            "total_encodings": total,
            "hash_accuracy": round(accuracy, 2),
            "active_models": self.manager.display_model_status(return_data=True),
            "mortality_log_count": len(self._mortality_log)
        }

# --- NEURAL LAYER: PyTorch Components (Upgraded) ---

class EmotionalAttention(nn.Module):
    """Emotional attention mechanism for sequence data."""

    def __init__(self, input_dim: int, emotional_dim: int, num_heads: int = 4):
        super().__init__()
        self.input_dim = input_dim
        self.emotional_dim = emotional_dim
        self.num_heads = num_heads
        self.head_dim = input_dim // num_heads
        
        assert self.head_dim * num_heads == input_dim, "input_dim must be divisible by num_heads"
        
        self.emotional_query = nn.Linear(emotional_dim, input_dim)
        self.input_key = nn.Linear(input_dim, input_dim)
        self.input_value = nn.Linear(input_dim, input_dim)
        
        self.output_projection = nn.Linear(input_dim, input_dim) 
        self.scale = math.sqrt(self.head_dim)
        
    def forward(self, x: torch.Tensor, emotional_embedding: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Emotional attention: emotional embedding as Q, sequence x as K/V.
        Returns pooled emotional context and attention weights.
        """
        batch_size, seq_len, input_dim = x.shape
        
        emotional_query = self.emotional_query(emotional_embedding).view(batch_size, self.num_heads, self.head_dim).unsqueeze(2)
        
        keys = self.input_key(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        values = self.input_value(x).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        attention_scores = torch.matmul(emotional_query, keys.transpose(-2, -1)) / self.scale
        attention_weights = F.softmax(attention_scores, dim=-1)
        
        attended_values = torch.matmul(attention_weights, values)
        attended_values = attended_values.squeeze(2).contiguous().view(batch_size, self.head_dim * self.num_heads)
        
        output = self.output_projection(attended_values)
        return output, attention_weights.squeeze(2)

class EmotionalWeightLayer(nn.Module):
    """
    Emotional weight layer with Axiom integration: Sigils modulate projections.
    Supports external_weights from UEI and sigil modulation.
    """

    def __init__(self, 
                 input_dim: int,
                 emotional_dim: int = 128,
                 num_emotions: int = 5,
                 emotion_categories: Optional[List[str]] = None,
                 use_emotional_attention: bool = True,
                 emotional_temperature: float = 1.0,
                 sigil_modulation_strength: float = 0.1):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.emotional_dim = emotional_dim
        self.num_emotions = num_emotions
        self.emotional_temperature = emotional_temperature
        self.sigil_modulation_strength = sigil_modulation_strength
        
        self.emotion_categories = ['love', 'joy', 'sadness', 'anger', 'fear'] if emotion_categories is None else emotion_categories
        
        if len(self.emotion_categories) != num_emotions:
            print(f"Warning: num_emotions ({num_emotions}) does not match length of emotion_categories ({len(self.emotion_categories)}). Adjusting num_emotions.")
            self.num_emotions = len(self.emotion_categories)
            
        self.emotion_embeddings = nn.Embedding(self.num_emotions, emotional_dim)
        self.input_to_emotion = nn.Linear(input_dim, emotional_dim)
        self.emotion_to_output = nn.Linear(emotional_dim, input_dim)
        
        self.use_emotional_attention = use_emotional_attention
        if use_emotional_attention:
            self.emotional_attention = EmotionalAttention(
                input_dim=input_dim,
                emotional_dim=emotional_dim,
                num_heads=4
            )
        
        self.emotional_bias = nn.Parameter(torch.zeros(input_dim))
        
        self.emotional_gate = nn.Sequential(
            nn.Linear(input_dim + emotional_dim, input_dim),
            nn.Sigmoid()
        )
        
        self._initialize_emotional_embeddings()
        
    def _initialize_emotional_embeddings(self) -> None:
        """Initializes with simulated valence-arousal patterns for interpretability."""
        emotion_patterns = {
            0: [0.8, 0.6, 0.7, 0.2],
            1: [0.9, 0.7, 0.8, 0.1],
            2: [-0.8, -0.3, -0.6, 0.4],
            3: [-0.6, 0.9, -0.3, 0.3],
            4: [-0.7, 0.8, -0.7, 0.6]
        }
        
        with torch.no_grad():
            for emotion_idx, pattern in emotion_patterns.items():
                if emotion_idx < self.num_emotions:
                    base_pattern = torch.tensor(pattern, dtype=torch.float32)
                    repeated_pattern = base_pattern.repeat(self.emotional_dim // len(pattern))
                    if len(repeated_pattern) < self.emotional_dim:
                        repeated_pattern = torch.cat([
                            repeated_pattern, 
                            base_pattern[:self.emotional_dim - len(repeated_pattern)]
                        ])
                    self.emotion_embeddings.weight[emotion_idx] = repeated_pattern
    
    def detect_emotional_context(self, x: torch.Tensor, external_weights: Optional[Dict[str, float]] = None, sigil: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Detect emotional context, fusing internal, external priors, and sigil modulation."""
        device = x.device
        x_for_projection: torch.Tensor

        if x.dim() == 3:
            x_for_projection = x.mean(dim=1)
        elif x.dim() > 3 and x.shape[1] == self.input_dim:
            # Conv style: [B, C, H, W...] -> mean pool over spatial dims
            spatial_dims = list(range(2, x.dim()))
            x_for_projection = x.mean(dim=spatial_dims)
        else:
            x_for_projection = x

        emotional_projection = self.input_to_emotion(x_for_projection)
        
        if sigil:
            modulation = sigil_to_modulation_vector(sigil, self.emotional_dim, device)
            emotional_projection = emotional_projection + modulation.unsqueeze(0) * self.sigil_modulation_strength
        
        internal_logits_list = []
        for i in range(self.num_emotions):
            emotion_embedding = self.emotion_embeddings.weight[i].unsqueeze(0)
            similarity = F.cosine_similarity(
                emotional_projection, 
                emotion_embedding.expand_as(emotional_projection),
                dim=1
            )
            internal_logits_list.append(similarity)
        internal_logits = torch.stack(internal_logits_list, dim=1)
        
        if external_weights:
            ue_prior_list = [external_weights.get(emotion, 0.0) for emotion in self.emotion_categories]
            ue_prior = torch.tensor(ue_prior_list, dtype=torch.float32, device=device).unsqueeze(0).expand_as(internal_logits)
            internal_logits = internal_logits + ue_prior
        
        emotion_weights = F.softmax(internal_logits / self.emotional_temperature, dim=1)
        
        emotion_embeddings = self.emotion_embeddings.weight.unsqueeze(0)
        weighted_emotion_embedding = torch.sum(
            emotion_weights.unsqueeze(-1) * emotion_embeddings, 
            dim=1
        )
        
        emotional_attention_weights = None
        
        if self.use_emotional_attention and x.dim() == 3:
            _, emotional_attention_weights = self.emotional_attention(
                x, weighted_emotion_embedding
            )
            
        return {
            'emotional_embedding': weighted_emotion_embedding,
            'emotion_weights': emotion_weights,
            'emotional_attention_weights': emotional_attention_weights
        }
    
    def apply_emotional_weighting(self, x: torch.Tensor, emotional_context: Dict) -> torch.Tensor:
        """Apply gated emotional fusion while preserving shape."""
        emotional_embedding = emotional_context['emotional_embedding']
        
        if x.dim() == 3:
            batch_size, seq_len, input_dim = x.shape
            emotional_embedding_expanded = emotional_embedding.unsqueeze(1).expand(-1, seq_len, -1)
            gate_input = torch.cat([x, emotional_embedding_expanded], dim=-1)
            emotional_gate = self.emotional_gate(gate_input)
            emotional_contribution = self.emotion_to_output(emotional_embedding_expanded)
            emotionally_weighted = x * (1 - emotional_gate) + emotional_contribution * emotional_gate
        
        elif x.dim() > 3 and x.shape[1] == self.input_dim:
            # Conv style: [B, C, H, W...]
            b, c = x.shape[0], x.shape[1]
            spatial_dims = x.shape[2:]
            x_perm = x.view(b, c, -1).transpose(1, 2)  # [B, L, C], L = prod(spatial)
            emotional_embedding_expanded = emotional_embedding.unsqueeze(1).expand(-1, x_perm.size(1), -1)
            gate_input = torch.cat([x_perm, emotional_embedding_expanded], dim=-1)
            emotional_gate = self.emotional_gate(gate_input)
            emotional_contribution = self.emotion_to_output(emotional_embedding_expanded)
            x_perm_weighted = x_perm * (1 - emotional_gate) + emotional_contribution * emotional_gate
            emotionally_weighted = x_perm_weighted.transpose(1, 2).view(b, c, *spatial_dims)
        
        else:
            gate_input = torch.cat([x, emotional_embedding], dim=-1)
            emotional_gate = self.emotional_gate(gate_input)
            emotional_contribution = self.emotion_to_output(emotional_embedding)
            emotionally_weighted = x * (1 - emotional_gate) + emotional_contribution * emotional_gate
        
        return emotionally_weighted + self.emotional_bias
    
    def forward(self, x: torch.Tensor, external_weights: Optional[Dict[str, float]] = None, sigil: Optional[str] = None) -> torch.Tensor:
        """Forward pass including context detection and weighting."""
        emotional_context = self.detect_emotional_context(x, external_weights, sigil)
        output = self.apply_emotional_weighting(x, emotional_context)
        return output
    
    def get_emotional_analysis(self, x: torch.Tensor, external_weights: Optional[Dict[str, float]] = None, sigil: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """Provides detailed emotional analysis tensors."""
        emotional_context = self.detect_emotional_context(x, external_weights, sigil)
        
        emotion_weights = emotional_context['emotion_weights'] + 1e-8
        entropy = -torch.sum(emotion_weights * torch.log(emotion_weights), dim=1)
        
        return {
            'emotion_weights': emotional_context['emotion_weights'],
            'emotional_embedding': emotional_context['emotional_embedding'],
            'dominant_emotion_index': torch.argmax(emotional_context['emotion_weights'], dim=1),
            'emotional_entropy': entropy,
            'emotional_intensity': torch.norm(emotional_context['emotional_embedding'], dim=1)
        }

class EmotionalModelWrapper(nn.Module):
    """
    Wrapper: Injects EmotionalWeightLayer into any PyTorch model via hooks.
    Supports broader layer types and sigil modulation.
    """

    def __init__(self, 
                 base_model: nn.Module,
                 target_layers: Optional[List[str]] = None,
                 **emotional_kwargs):
        
        super().__init__()
        self.base_model = base_model
        
        target_dims = self._find_target_layers_and_dims(target_layers)
        self.target_layer_names = list(target_dims.keys())
            
        self.emotional_weight_layers = nn.ModuleDict()
        for layer_name, dim in target_dims.items():
            self.emotional_weight_layers[layer_name] = EmotionalWeightLayer(
                input_dim=dim,
                **emotional_kwargs
            )
            
    def _find_target_layers_and_dims(self, target_layers: Optional[List[str]]) -> Dict[str, int]:
        """Auto-detect output dimension for various PyTorch layer types."""
        target_dims: Dict[str, int] = {}
        named_modules = dict(self.base_model.named_modules())
        
        if target_layers is None:
            raise ValueError("target_layers must be explicitly provided (e.g., ['lstm', 'fc']).")

        for name in target_layers:
            module = named_modules.get(name)
            if module is None:
                print(f"Warning: Target layer '{name}' not found in model named modules.")
                continue
            
            dim: Optional[int] = None
            if isinstance(module, nn.Linear):
                dim = module.out_features
            elif isinstance(module, (nn.LSTM, nn.GRU)):
                dim = module.hidden_size * (2 if getattr(module, "bidirectional", False) else 1)
            elif isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                dim = module.out_channels
            elif isinstance(module, nn.MultiheadAttention):
                dim = module.embed_dim
            elif isinstance(module, (nn.TransformerEncoderLayer, nn.TransformerDecoderLayer)):
                dim = module.d_model
            elif hasattr(module, 'out_features'):
                dim = int(module.out_features)
            elif hasattr(module, 'out_channels'):
                dim = int(module.out_channels)
            elif hasattr(module, 'd_model'):
                dim = int(module.d_model)
            elif hasattr(module, 'embed_dim'):
                dim = int(module.embed_dim)

            if dim is not None:
                target_dims[name] = dim
        
        if not target_dims:
            raise ValueError(f"No suitable dimensions found for the provided target_layers: {target_layers}. Supported types: Linear, ConvNd, LSTM/GRU, Transformer*.")

        return target_dims
    
    def forward(self, x: torch.Tensor, external_weights: Optional[Dict[str, float]] = None, sigil: Optional[str] = None, return_emotional_analysis: bool = False):
        """Forward pass with emotional injection via hooks."""
        emotional_analyses: Dict[str, Any] = {}
        named_modules = dict(self.base_model.named_modules())
        
        def apply_emotional_weighting(layer_name: str, return_analysis: bool, ext_weights: Optional[Dict[str, float]], sig: Optional[str]):
            def hook(module, input, output):
                emotional_layer = self.emotional_weight_layers[layer_name]
                
                if isinstance(output, tuple):
                    main_out = output[0]
                    mod_main = emotional_layer(main_out, external_weights=ext_weights, sigil=sig)
                    if return_analysis:
                        emotional_analyses[layer_name] = emotional_layer.get_emotional_analysis(
                            main_out, external_weights=ext_weights, sigil=sig
                        )
                    return (mod_main,) + output[1:]
                else:
                    mod_out = emotional_layer(output, external_weights=ext_weights, sigil=sig)
                    if return_analysis:
                        emotional_analyses[layer_name] = emotional_layer.get_emotional_analysis(
                            output, external_weights=ext_weights, sigil=sig
                        )
                    return mod_out
            return hook
        
        hooks = []
        for layer_name in self.target_layer_names:
            module = named_modules[layer_name]
            hook = module.register_forward_hook(
                apply_emotional_weighting(layer_name, return_emotional_analysis, external_weights, sigil)
            )
            hooks.append(hook)
        
        output = self.base_model(x)
        
        for hook in hooks:
            hook.remove()
            
        if return_emotional_analysis:
            return output, emotional_analyses
        return output

# --- HYBRID EMOTIONAL CORE (INTEGRATION LAYER) ---

class HybridEmotionalCore:
    """
    Hybrid core: UEI symbolic priors + PyTorch emotional weighting with Axiom sigil modulation.
    Standalone addition for any PyTorch model.
    """

    def __init__(self, base_pytorch_model: nn.Module, target_layers: List[str], uei_default_model: str = 'western', sigil_modulation_strength: float = 0.1):
        self.uei = UniversalEmotionalInterface(default_model=uei_default_model, default_weight=1.0)
        
        self.emotional_model_wrapper = EmotionalModelWrapper(
            base_model=base_pytorch_model,
            target_layers=target_layers,
            emotional_dim=128,
            num_emotions=5,
            emotion_categories=['love', 'joy', 'sadness', 'anger', 'fear'],
            use_emotional_attention=True,
            sigil_modulation_strength=sigil_modulation_strength
        )
        print("Hybrid Emotional Core Initialized (UEI + Axiom PyTorch Wrapper).")

    def run_inference(self, input_tensor: torch.Tensor, concept: str, return_analysis: bool = False) -> tuple[torch.Tensor, Optional[Dict[str, Any]], Dict[str, float]]:
        """
        Run hybrid inference: UEI encodes sigil, primes priors; neural computes with sigil modulation.
        """
        encode_result = self.uei.encode_concept(concept)
        sigil = encode_result.get('sigil', None)
        
        uei_prior_weights = self.uei.manager.get_combined_weights()
        
        first_layer_name = list(self.emotional_model_wrapper.emotional_weight_layers.keys())[0]
        core_emotion_categories = self.emotional_model_wrapper.emotional_weight_layers[first_layer_name].emotion_categories
        
        core_emotion_weights = {
            k: v for k, v in uei_prior_weights.items() 
            if k in core_emotion_categories
        }

        output = self.emotional_model_wrapper(
            input_tensor, 
            external_weights=core_emotion_weights, 
            sigil=sigil,
            return_emotional_analysis=return_analysis
        )
        
        analysis: Optional[Dict[str, Any]] = None
        if return_analysis:
            output, analysis = output
            
        return output, analysis, core_emotion_weights

# --- DEMONSTRATION ---

if __name__ == "__main__":
    print("="*80)
    print("UNIVERSAL EMOTIONAL INTERFACE STARTUP (DEMO)")
    print("="*80)
    
    uei = UniversalEmotionalInterface(default_model='cross_cultural', default_weight=1.0) 
    
    print("\nExternal Task 1: Encode 'Saudade' (Cultural Concept)")
    result_saudade = uei.encode_concept('saudade')
    saudade_sigil = result_saudade.get('sigil')
    print(f"Concept: Saudade | Sigil: {saudade_sigil} | Confidence: {result_saudade['confidence']:.4f}")
    
    print("\nExternal Task 2: Encode 'Quantum Fluctuation' (Unmapped Concept)")
    result_quantum = uei.encode_concept('Quantum Fluctuation')
    quantum_sigil = result_quantum.get('sigil')
    print(f"Concept: Quantum Fluctuation | Sigil: {quantum_sigil} | Confidence: {result_quantum['confidence']:.4f}")

    if saudade_sigil:
        print("\nExternal Task 3: Decode 'Saudade' Sigil")
        decoded_saudade = uei.decode_sigil(saudade_sigil)
        if decoded_saudade.get("status") == "SUCCESS":
            lines = decoded_saudade['full_context_data'].split('\n')
            if lines:
                print(lines[0])
                if len(lines) > 1:
                    print(lines[1])
        
    print("\nExternal Task 4: Get Core Status")
    uei.manager.display_model_status()

    print("\n" + "="*80)
    print("HYBRID EMOTIONAL CORE (AXIOM) DEMONSTRATION")
    print("="*80)
    
    class TestLSTMModel(nn.Module):
        def __init__(self, input_dim: int = 10, hidden_size: int = 64, num_layers: int = 1, output_dim: int = 5):
            super().__init__()
            self.lstm = nn.LSTM(input_dim, hidden_size, num_layers, batch_first=True)
            self.fc = nn.Linear(hidden_size, output_dim)
            
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            lstm_out, (hn, cn) = self.lstm(x)
            return self.fc(hn[-1])

    base_model = TestLSTMModel(input_dim=10, hidden_size=64)
    hybrid_core = HybridEmotionalCore(
        base_pytorch_model=base_model,
        target_layers=['lstm', 'fc'],
        sigil_modulation_strength=0.2
    )

    sample_input = torch.randn(1, 20, 10)
    
    print("\nScenario A: 'amour' (Love concept - Cross-Cultural Bias)")
    output_a, analysis_a, uei_weights_a = hybrid_core.run_inference(sample_input, concept='amour', return_analysis=True)
    
    if analysis_a and 'lstm' in analysis_a:
        analysis_lstm = analysis_a['lstm']
        dominant_emotion_a_idx = analysis_lstm['dominant_emotion_index'][0].item()
        dominant_emotion_a = hybrid_core.emotional_model_wrapper.emotional_weight_layers['lstm'].emotion_categories[dominant_emotion_a_idx]
        
        print(f"Output Shape: {list(output_a.shape)}")
        print(f"LSTM Layer Dominant Emotion (Axiom Modulated): {dominant_emotion_a.upper()}")
        print(f"LSTM Emotional Entropy: {analysis_lstm['emotional_entropy'][0].item():.4f}")

    print("\nScenario B: Shift Context to Neuroscientific/Eastern Bias")
    hybrid_core.uei.set_emotional_context({'neuroscientific': 2.0, 'eastern': 0.8})

    print("\nScenario C: 'angst' (Fear concept - Neuroscientific Bias)")
    output_c, analysis_c, uei_weights_c = hybrid_core.run_inference(sample_input, concept='angst', return_analysis=True)
    
    if analysis_c and 'lstm' in analysis_c:
        analysis_lstm = analysis_c['lstm']
        dominant_emotion_c_idx = analysis_lstm['dominant_emotion_index'][0].item()
        dominant_emotion_c = hybrid_core.emotional_model_wrapper.emotional_weight_layers['lstm'].emotion_categories[dominant_emotion_c_idx]

        print(f"Output Shape: {list(output_c.shape)}")
        print(f"LSTM Layer Dominant Emotion (Axiom Modulated): {dominant_emotion_c.upper()}")
        print(f"LSTM Emotional Entropy: {analysis_lstm['emotional_entropy'][0].item():.4f}")
EOF
