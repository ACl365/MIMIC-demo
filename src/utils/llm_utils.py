"""
Utilities for interacting with Large Language Models (LLMs) for explanations.
"""

import os
from typing import List, Optional, Tuple  # Remove Dict, add Optional

import numpy as np

# Attempt to import openai, handle if not installed
try:
    import openai

    OPENAI_AVAILABLE = True
    # Ensure API key is set (consider more robust key management for production)
    if not os.getenv("OPENAI_API_KEY"):
        print(
            "Warning: OPENAI_API_KEY environment variable not set. LLM explanation will fail."
        )
        # OPENAI_AVAILABLE = False # Optionally disable if key is mandatory
except ImportError:
    OPENAI_AVAILABLE = False
    print(
        "Warning: 'openai' library not installed. LLM explanation feature is disabled. Install with: pip install openai"
    )

from .logger import get_logger  # Assuming logger setup in utils

logger = get_logger(__name__)


def format_shap_prompt(
    top_features: List[Tuple[str, float]], prediction_prob: float
) -> str:
    """Formats a prompt for an LLM to explain SHAP contributions."""

    prompt = (
        "You are an AI assistant explaining machine learning model predictions to a clinician. "
        "The model predicts the probability of 30-day hospital readmission. "
        f"For this patient, the predicted probability is {prediction_prob:.2f}.\n\n"
        "The top factors influencing this prediction, according to SHAP analysis, are:\n"
    )
    for feature, shap_value in top_features:
        direction = "increases" if shap_value > 0 else "decreases"
        magnitude = (
            "significantly" if abs(shap_value) > 0.1 else "slightly"
        )  # Example threshold
        prompt += f"- {feature}: {direction} the predicted risk ({magnitude}, SHAP value: {shap_value:.3f})\n"

    prompt += "\nPlease provide a concise, one-sentence explanation summarizing the key drivers of this prediction for a clinician, avoiding technical jargon like 'SHAP value'."
    return prompt


def explain_shap_with_llm(
    shap_values_single: np.ndarray,
    feature_names: List[str],
    prediction_prob: float,
    top_n: int = 3,
    model_name: str = "gpt-3.5-turbo-instruct",  # Or another suitable completion model
) -> Optional[
    str
]:  # Type hint already uses Optional, no change needed here if import is added
    """
    Uses an LLM (currently OpenAI) to generate a human-readable explanation
    from SHAP values for a single prediction.

    Args:
        shap_values_single (np.ndarray): SHAP values for one prediction (1D array).
        feature_names (List[str]): List of feature names corresponding to shap_values.
        prediction_prob (float): The predicted probability for this instance.
        top_n (int): Number of top features to include in the explanation.
        model_name (str): The specific OpenAI model to use.

    Returns:
        Optional[str]: The explanation text from the LLM, or None if unavailable/error.
    """
    if not OPENAI_AVAILABLE:
        logger.warning(
            "OpenAI library not available or API key not set. Cannot generate LLM explanation."
        )
        return None
    if not os.getenv("OPENAI_API_KEY"):
        logger.warning("OPENAI_API_KEY not set. Cannot generate LLM explanation.")
        return None

    if len(shap_values_single) != len(feature_names):
        logger.error(
            f"Mismatch between SHAP values ({len(shap_values_single)}) and feature names ({len(feature_names)})."
        )
        return None

    try:
        # Calculate absolute SHAP values and get indices of top N features
        abs_shap = np.abs(shap_values_single)
        top_indices = np.argsort(abs_shap)[-top_n:][::-1]  # Indices of top N features

        # Get feature names and their corresponding SHAP values
        top_features = [(feature_names[i], shap_values_single[i]) for i in top_indices]

        # Format the prompt
        prompt = format_shap_prompt(top_features, prediction_prob)
        logger.debug(f"Generated LLM Prompt:\n{prompt}")

        # Call OpenAI API (using Completion endpoint for instruct models)
        # Note: For chat models like gpt-3.5-turbo or gpt-4, use openai.ChatCompletion.create
        client = openai.OpenAI()  # Uses OPENAI_API_KEY from environment by default
        response = client.completions.create(
            model=model_name,
            prompt=prompt,
            max_tokens=100,  # Limit response length
            temperature=0.5,  # Adjust creativity
            n=1,
            stop=None,  # Let the model decide when to stop
        )

        explanation = response.choices[0].text.strip()
        logger.info("Successfully generated LLM explanation.")
        return explanation

    except openai.AuthenticationError:
        logger.error(
            "OpenAI API key invalid or missing. Cannot generate LLM explanation."
        )
        return None
    except Exception as e:
        logger.error(f"Error calling OpenAI API: {e}", exc_info=True)
        return None
