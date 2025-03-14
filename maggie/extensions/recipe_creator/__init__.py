"""
Maggie AI Extension - Recipe Creator
===================================

Speech-to-document recipe creation extension for Maggie AI Assistant.

This extension provides a streamlined workflow for creating recipe documents
from speech input, enabling users to dictate recipes and have them
automatically formatted and saved as Word documents.
"""

from maggie.extensions.recipe_creator import RecipeCreator, RecipeState, RecipeData
__all__ = ['RecipeCreator', 'RecipeState', 'RecipeData']