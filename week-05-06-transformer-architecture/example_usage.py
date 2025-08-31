"""
Example Usage of the Transformer Encoder Demonstration
====================================================

This script shows how to use the main demonstration to understand
Transformer encoder behavior and extract educational insights.
"""

from main_demo import main, format_analysis_results, demonstrate_educational_insights

def simple_usage_example():
    """Simple example showing basic usage."""
    
    # Run the complete demonstration
    results = main()
    
    # Extract key insights for quick overview
    formatted = results['formatted_analysis']
    insights = results['educational_insights']
    
    print("=== TRANSFORMER ENCODER DEMONSTRATION RESULTS ===")
    print()
    
    # 1. Classification Results
    if 'classification_summary' in formatted:
        cls_summary = formatted['classification_summary']
        config = cls_summary['model_config']
        
        print("1. TEXT CLASSIFICATION DEMO")
        print("-" * 40)
        print(f"Model: {config['d_model']}D, {config['num_layers']} layers, {config['num_heads']} heads")
        print(f"Vocabulary size: {config['vocab_size']:,} tokens")
        print()
        
        print("Sample predictions:")
        for i, pred in enumerate(cls_summary['predictions'][:3]):  # Show first 3
            text_preview = pred['text'][:50] + "..." if len(pred['text']) > 50 else pred['text']
            print(f"  Text: \"{text_preview}\"")
            print(f"  Prediction: {pred['predicted_class']} ({pred['technical_prob']:.3f} confidence)")
            print()
    
    # 2. Representation Evolution
    if 'representation_evolution_summary' in formatted:
        repr_summary = formatted['representation_evolution_summary']
        
        print("2. REPRESENTATION EVOLUTION")
        print("-" * 40)
        print(f"Analyzed text: \"{repr_summary['text']}\"")
        print(f"Tokens: {' | '.join(repr_summary['tokens'])}")
        print()
        
        print("Layer-to-layer similarities:")
        for i, sim in enumerate(repr_summary['layer_similarities']):
            print(f"  Layer {i} -> Layer {i+1}: {sim:.4f}")
        
        print(f"\nKey insights:")
        print(f"  - {repr_summary['insights']['most_similar_layers']}")
        print(f"  - {repr_summary['insights']['most_diverse_attention']}")
        print(f"  - {repr_summary['insights']['strongest_representations']}")
        print()
    
    # 3. Performance Analysis
    if 'performance_scaling_summary' in formatted:
        perf_summary = formatted['performance_scaling_summary']
        
        print("3. PERFORMANCE SCALING")
        print("-" * 40)
        
        for config_name, data in perf_summary.items():
            print(f"{config_name.upper()} model:")
            print(f"  Parameters: {data['parameters']:,}")
            print(f"  Inference time: {data['inference_time_ms']:.2f} ms")
            print(f"  Throughput: {data['throughput_tokens_per_sec']:.1f} tokens/sec")
            print()
    
    # 4. Educational Insights
    print("4. KEY EDUCATIONAL INSIGHTS")
    print("-" * 40)
    
    for finding in insights['key_findings']:
        print(f"• {finding}")
    
    print("\nArchitectural insights:")
    for insight in insights['architectural_insights'][:3]:  # Show first 3
        print(f"• {insight}")
    
    print("\nPractical recommendations:")
    for rec in insights['practical_recommendations'][:3]:  # Show first 3
        print(f"• {rec}")
    
    print("\nLearning objectives achieved:")
    for obj in insights['learning_objectives_met']:
        print(f"{obj}")

def detailed_analysis_example():
    """Example showing how to access detailed analysis data."""
    
    # Run demonstration
    results = main()
    raw_results = results['raw_results']
    
    print("\n=== DETAILED ANALYSIS ACCESS ===")
    print()
    
    # Access raw attention weights for analysis
    if 'attention_patterns' in raw_results:
        attention_data = raw_results['attention_patterns']
        layer_analyses = attention_data['layer_analyses']
        
        print("ATTENTION PATTERN ANALYSIS")
        print("-" * 30)
        
        # Show attention patterns from the first layer, first head
        first_layer = layer_analyses[0]
        first_head = first_layer['heads'][0]
        
        print(f"Layer 0, Head 0 - Most attended pairs:")
        for attention in first_head['max_attentions'][:5]:  # First 5
            query = attention['query_token']
            key = attention['key_token']
            score = attention['attention_score']
            print(f"  '{query}' -> '{key}' ({score:.3f})")
        print()
    
    # Access feature similarities
    if 'feature_extraction' in raw_results:
        feature_data = raw_results['feature_extraction']
        similarity_matrix = feature_data['similarity_matrix']
        texts = feature_data['texts']
        
        print("FEATURE SIMILARITY ANALYSIS")
        print("-" * 30)
        
        # Find most similar text pair
        max_sim = 0
        max_pair = (0, 1)
        for i in range(len(texts)):
            for j in range(i + 1, len(texts)):
                if similarity_matrix[i, j] > max_sim:
                    max_sim = similarity_matrix[i, j]
                    max_pair = (i, j)
        
        i, j = max_pair
        print(f"Most similar texts (similarity: {max_sim:.3f}):")
        print(f"  1. \"{texts[i]}\"")
        print(f"  2. \"{texts[j]}\"")

if __name__ == "__main__":
    # Run the simple example
    simple_usage_example()
    
    # Run detailed analysis example
    detailed_analysis_example()
    
    print("\n=== SUMMARY ===")
    print("This demonstration provides:")
    print("• Complete working Transformer encoder implementation")
    print("• Text classification example showing practical usage")
    print("• Detailed analysis of how representations evolve through layers")
    print("• Attention pattern visualization and interpretation")
    print("• Performance scaling analysis for different model sizes")
    print("• Feature extraction capabilities for downstream tasks")
    print("• Educational insights into Transformer architecture design")
    
    print("\nUse these results to:")
    print("• Understand how Transformers process information")
    print("• Compare different architectural choices")
    print("• Build intuition for hyperparameter selection")
    print("• Design downstream tasks and applications")
    print("• Optimize models for specific requirements")