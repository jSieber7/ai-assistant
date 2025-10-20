#!/usr/bin/env python3
"""
Test script to verify model filtering is working correctly.
"""
import sys
import os
sys.path.append('.')

# Mock gradio module to avoid import issues
class MockGradio:
    pass

sys.modules['gradio'] = MockGradio()

from app.core.config import get_available_models, initialize_llm_providers

def test_model_filtering():
    print("Testing model filtering...")
    
    # Initialize providers
    try:
        initialize_llm_providers()
        print("✓ Providers initialized")
    except Exception as e:
        print(f"✗ Failed to initialize providers: {e}")
        return
    
    # Get models
    try:
        models = get_available_models()
        print(f"Found {len(models)} models:")
        
        openrouter_models = []
        other_models = []
        
        for model in models:
            if hasattr(model, 'provider') and hasattr(model.provider, 'value'):
                if 'openrouter' in str(model.provider.value).lower() or 'openai_compatible' in str(model.provider.value).lower():
                    openrouter_models.append(model)
                else:
                    other_models.append(model)
        
        print(f"  OpenRouter/OpenAI-compatible models: {len(openrouter_models)}")
        for i, model in enumerate(openrouter_models[:5]):  # Show first 5
            print(f"    {i+1}. {model.name}")
        if len(openrouter_models) > 5:
            print(f"    ... and {len(openrouter_models) - 5} more")
        
        print(f"  Other provider models: {len(other_models)}")
        for i, model in enumerate(other_models[:5]):  # Show first 5
            print(f"    {i+1}. {model.name} (provider: {model.provider.value})")
        if len(other_models) > 5:
            print(f"    ... and {len(other_models) - 5} more")
            
        print("\n✓ Model filtering test completed successfully")
        print("Note: OpenRouter models should now be filtered to only show accessible models")
        
    except Exception as e:
        print(f"✗ Failed to get models: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_model_filtering()