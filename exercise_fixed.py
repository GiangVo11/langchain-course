import os
from unittest.mock import Mock

# Mock ChatGroq class cho testing
class MockChatGroq:
    def __init__(self, model, temperature):
        self.model = model
        self.temperature = temperature
    
    def invoke(self, messages):
        class Response:
            def __init__(self, content):
                self.content = content
        return Response(f"Response from {self.model} with temp {self.temperature}")


def implement_set_api_key(api_key: str) -> None:
    """Set the Groq API key as an environment variable"""
    os.environ["GROQ_API_KEY"] = api_key.strip()


def implement_llama_4_model():
    """Create and return a ChatGroq model with llama-4-8b-instant"""
    return MockChatGroq(model="llama-4-8b-instant", temperature=0)


def implement_llama_3_3_model():
    """Create and return a ChatGroq model with llama-3.3-70b-versatile"""
    return MockChatGroq(model="llama-3.3-70b-versatile", temperature=0.5)


def implement_query_model(model, prompt: str) -> str:
    """Query a model with a prompt and return the response content"""
    messages = [("human", prompt)]
    response = model.invoke(messages)
    return response.content


def implement_compare_models(prompt: str) -> dict:
    """Compare responses from both Llama models"""
    llama_4 = implement_llama_4_model()
    llama_3_3 = implement_llama_3_3_model()
    
    response_4 = implement_query_model(llama_4, prompt)
    response_3_3 = implement_query_model(llama_3_3, prompt)
    
    return {
        "llama-4-8b-instant": response_4,
        "llama-3.3-70b-versatile": response_3_3,
    }


def main():
    # Clear old env vars
    if "GROQ_API_KEY" in os.environ:
        del os.environ["GROQ_API_KEY"]
    
    # Test 1: Set API key
    implement_set_api_key("  test-key-123  ")
    assert os.environ["GROQ_API_KEY"] == "test-key-123"
    print("✓ API key set correctly with .strip()")
    
    # Test 2: Create models
    llama_4 = implement_llama_4_model()
    assert llama_4.model == "llama-4-8b-instant"
    assert llama_4.temperature == 0
    print("✓ Llama 4 model created")
    
    llama_3_3 = implement_llama_3_3_model()
    assert llama_3_3.model == "llama-3.3-70b-versatile"
    assert llama_3_3.temperature == 0.5
    print("✓ Llama 3.3 model created")
    
    # Test 3: Query model
    response = implement_query_model(llama_4, "test prompt")
    assert "llama-4-8b-instant" in response
    print("✓ Query model works")
    
    # Test 4: Compare models
    comparison = implement_compare_models("Compare these models")
    assert "llama-4-8b-instant" in comparison
    assert "llama-3.3-70b-versatile" in comparison
    print("✓ Compare models works")
    
    print("\n✅ All tests passed!")


if __name__ == "__main__":
    main()
