#To Integrate Into Your Own Model
from hybrid_emotional_core import HybridEmotionalCore

hybrid = HybridEmotionalCore(
    base_pytorch_model=your_model,
    target_layers=['name_of_layer_in_your_model'],
    sigil_modulation_strength=0.2,
)

out, analysis, priors = hybrid.run_inference(input_tensor, concept='love', return_analysis=True)
