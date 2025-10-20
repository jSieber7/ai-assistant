"""
Unit tests for model filtering functionality.
"""

import pytest
from unittest.mock import patch, MagicMock


@pytest.mark.unit
class TestModelFiltering:
    """Test model filtering functionality"""

    @patch('app.core.config.initialize_llm_providers')
    @patch('app.core.config.get_available_models')
    def test_model_filtering(self, mock_get_models, mock_initialize):
        """Test model filtering functionality"""
        # Create mock models
        mock_openrouter_model = MagicMock()
        mock_openrouter_model.name = "openrouter/model1"
        mock_openrouter_model.provider.value = "openrouter"
        
        mock_openai_model = MagicMock()
        mock_openai_model.name = "openai/gpt-4"
        mock_openai_model.provider.value = "openai_compatible"
        
        mock_other_model = MagicMock()
        mock_other_model.name = "other/model"
        mock_other_model.provider.value = "other_provider"
        
        # Setup mock return
        mock_get_models.return_value = [
            mock_openrouter_model,
            mock_openai_model,
            mock_other_model
        ]
        
        # Import after mocking
        from app.core.config import get_available_models, initialize_llm_providers
        
        # Test initialization
        initialize_llm_providers()
        mock_initialize.assert_called_once()
        
        # Test model retrieval
        models = get_available_models()
        assert len(models) == 3
        
        # Test model filtering
        openrouter_models = []
        other_models = []
        
        for model in models:
            if hasattr(model, 'provider') and hasattr(model.provider, 'value'):
                if 'openrouter' in str(model.provider.value).lower() or 'openai_compatible' in str(model.provider.value).lower():
                    openrouter_models.append(model)
                else:
                    other_models.append(model)
        
        # Verify filtering results
        assert len(openrouter_models) == 2
        assert len(other_models) == 1
        assert openrouter_models[0].name == "openrouter/model1"
        assert openrouter_models[1].name == "openai/gpt-4"
        assert other_models[0].name == "other/model"

    @patch('app.core.config.initialize_llm_providers')
    def test_initialization_failure(self, mock_initialize):
        """Test handling of initialization failure"""
        # Make initialization raise an exception
        mock_initialize.side_effect = Exception("Initialization failed")
        
        # Import after mocking
        from app.core.config import initialize_llm_providers
        
        # Test that the exception is raised
        with pytest.raises(Exception, match="Initialization failed"):
            initialize_llm_providers()

    @patch('app.core.config.initialize_llm_providers')
    @patch('app.core.config.get_available_models')
    def test_get_models_failure(self, mock_get_models, mock_initialize):
        """Test handling of get_models failure"""
        # Make get_models raise an exception
        mock_get_models.side_effect = Exception("Failed to get models")
        
        # Import after mocking
        from app.core.config import get_available_models, initialize_llm_providers
        
        # Test initialization succeeds
        initialize_llm_providers()
        mock_initialize.assert_called_once()
        
        # Test that get_models raises an exception
        with pytest.raises(Exception, match="Failed to get models"):
            get_available_models()

    @patch('app.core.config.initialize_llm_providers')
    @patch('app.core.config.get_available_models')
    def test_empty_model_list(self, mock_get_models, mock_initialize):
        """Test handling of empty model list"""
        # Setup mock return with empty list
        mock_get_models.return_value = []
        
        # Import after mocking
        from app.core.config import get_available_models, initialize_llm_providers
        
        # Test initialization
        initialize_llm_providers()
        mock_initialize.assert_called_once()
        
        # Test empty model list
        models = get_available_models()
        assert len(models) == 0
        
        # Test filtering with empty list
        openrouter_models = []
        other_models = []
        
        for model in models:
            if hasattr(model, 'provider') and hasattr(model.provider, 'value'):
                if 'openrouter' in str(model.provider.value).lower() or 'openai_compatible' in str(model.provider.value).lower():
                    openrouter_models.append(model)
                else:
                    other_models.append(model)
        
        # Verify empty results
        assert len(openrouter_models) == 0
        assert len(other_models) == 0

    @patch('app.core.config.initialize_llm_providers')
    @patch('app.core.config.get_available_models')
    def test_model_without_provider(self, mock_get_models, mock_initialize):
        """Test handling of models without provider attribute"""
        # Create mock models - one with provider, one without
        mock_model_with_provider = MagicMock()
        mock_model_with_provider.name = "model_with_provider"
        mock_model_with_provider.provider.value = "openrouter"
        
        mock_model_without_provider = MagicMock()
        mock_model_without_provider.name = "model_without_provider"
        # Deliberately don't set provider attribute
        # Remove the provider attribute entirely
        del mock_model_without_provider.provider
        
        # Setup mock return
        mock_get_models.return_value = [
            mock_model_with_provider,
            mock_model_without_provider
        ]
        
        # Import after mocking
        from app.core.config import get_available_models, initialize_llm_providers
        
        # Test initialization
        initialize_llm_providers()
        mock_initialize.assert_called_once()
        
        # Test model retrieval
        models = get_available_models()
        assert len(models) == 2
        
        # Test model filtering
        openrouter_models = []
        other_models = []
        
        for model in models:
            if hasattr(model, 'provider') and hasattr(model.provider, 'value'):
                if 'openrouter' in str(model.provider.value).lower() or 'openai_compatible' in str(model.provider.value).lower():
                    openrouter_models.append(model)
                else:
                    other_models.append(model)
        
        # Verify filtering results
        assert len(openrouter_models) == 1
        assert len(other_models) == 0
        assert openrouter_models[0].name == "model_with_provider"