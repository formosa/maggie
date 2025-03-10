"""
Maggie AI Assistant - Recipe Creator Utility
=========================================
Utility module for creating recipes.
Handles multi-step interactions and document creation.
"""

import os
import time
import threading
from enum import Enum, auto
from dataclasses import dataclass
from typing import Dict, Any, Optional, List, Callable
import docx
from loguru import logger

# Parent class import with corrected path
from utils.utility_base import UtilityBase


class RecipeState(Enum):
    """States for the recipe creation process."""
    INITIAL = auto()
    NAME_PROMPT = auto()
    NAME_CONFIRM = auto()
    DESCRIPTION_PROMPT = auto()
    RECORDING = auto()
    PROCESSING = auto()
    DOCUMENT_CREATION = auto()
    COMPLETED = auto()
    CANCELLED = auto()


@dataclass
class RecipeData:
    """Data structure for recipe information."""
    name: str = ""
    description: str = ""
    ingredients: List[str] = None
    steps: List[str] = None
    
    def __post_init__(self):
        """Initialize empty lists."""
        if self.ingredients is None:
            self.ingredients = []
        if self.steps is None:
            self.steps = []


class RecipeCreator(UtilityBase):
    """
    Recipe Creator utility for creating and formatting recipes.
    
    Parameters
    ----------
    event_bus : EventBus
        Reference to the central event bus
    config : Dict[str, Any]
        Configuration for the utility
    """
    
    def __init__(self, event_bus, config):
        """
        Initialize the Recipe Creator utility.
        
        Parameters
        ----------
        event_bus : EventBus
            Reference to the central event bus
        config : Dict[str, Any]
            Configuration for the utility
        """
        super().__init__(event_bus, config)
        
        self.state = RecipeState.INITIAL
        self.recipe_data = RecipeData()
        
        self.output_dir = config.get("output_dir", "recipes")
        self.template_path = config.get("template_path", "templates/recipe_template.docx")
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create template if it doesn't exist
        if not os.path.exists(self.template_path):
            template_dir = os.path.dirname(self.template_path)
            os.makedirs(template_dir, exist_ok=True)
            self._create_template()
        
        # Register event handlers
        self.event_bus.subscribe("speech_recognized", self._handle_speech)
        self.event_bus.subscribe("recipe_command", self._handle_command)
        
        # Reference to speech processor and LLM processor will be set during start
        self.speech_processor = None
        self.llm_processor = None
        self._processing_thread = None
        
    def _initialize_resources(self) -> bool:
        """
        Initialize resources needed by this utility.
        
        Returns
        -------
        bool
            True if resources were initialized successfully
        """
        try:
            # Get references to required components from event_bus subscribers
            main_app = next((sub for sub in self.event_bus.subscribers.get("main_app", []) 
                          if hasattr(sub, "speech_processor") and hasattr(sub, "llm_processor")), None)
                          
            if main_app:
                self.speech_processor = main_app.speech_processor
                self.llm_processor = main_app.llm_processor
                return True
            else:
                logger.error("Failed to get speech and LLM processors")
                return False
        except Exception as e:
            logger.error(f"Error initializing RecipeCreator resources: {e}")
            return False
        
    def _create_template(self):
        """Create a basic recipe template file."""
        try:
            doc = docx.Document()
            doc.add_heading("Recipe Name", level=1)
            doc.add_heading("Ingredients", level=2)
            doc.add_paragraph("• Ingredient 1", style='ListBullet')
            doc.add_paragraph("• Ingredient 2", style='ListBullet')
            doc.add_heading("Instructions", level=2)
            doc.add_paragraph("1. Step 1")
            doc.add_paragraph("2. Step 2")
            doc.add_heading("Notes", level=2)
            doc.add_paragraph("Add any additional notes here.")
            
            os.makedirs(os.path.dirname(self.template_path), exist_ok=True)
            doc.save(self.template_path)
            logger.info(f"Created recipe template at {self.template_path}")
        except Exception as e:
            logger.error(f"Failed to create recipe template: {e}")
        
    def start(self) -> bool:
        """
        Start the Recipe Creator utility.
        
        Returns
        -------
        bool
            True if started successfully
        """
        try:
            # Initialize if needed
            if not self._initialized and not self.initialize():
                return False
                
            # Reset the state and data
            self.state = RecipeState.INITIAL
            self.recipe_data = RecipeData()
            
            # Start the recipe creation workflow in a thread
            self._processing_thread = threading.Thread(
                target=self._recipe_workflow,
                name="RecipeCreatorThread"
            )
            self._processing_thread.daemon = True
            self._processing_thread.start()
            
            self.running = True
            return True
            
        except Exception as e:
            logger.error(f"Error starting Recipe Creator: {e}")
            return False
            
    def stop(self) -> bool:
        """
        Stop the Recipe Creator utility.
        
        Returns
        -------
        bool
            True if stopped successfully
        """
        try:
            # Cancel if still running
            if self.running:
                self.state = RecipeState.CANCELLED
                
            self.running = False
            
            # Wait for processing thread to finish
            if self._processing_thread and self._processing_thread.is_alive():
                self._processing_thread.join(timeout=5.0)
                
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Recipe Creator: {e}")
            return False
            
    def process_command(self, command: str) -> bool:
        """
        Process a command directed to this utility.
        
        Parameters
        ----------
        command : str
            The command string to process
            
        Returns
        -------
        bool
            True if command was processed, False if not applicable
        """
        if not self.running:
            return False
            
        # Check for cancel command
        if "cancel" in command.lower():
            self.state = RecipeState.CANCELLED
            return True
            
        # Publish command to event bus for handling based on current state
        self.event_bus.publish("recipe_command", command)
        return True
        
    def _recipe_workflow(self):
        """Main workflow for recipe creation."""
        try:
            # Welcome message
            self.speech_processor.speak("Starting recipe creator. Let's create a new recipe.")
            
            # Transition to name prompt
            self.state = RecipeState.NAME_PROMPT
            self.speech_processor.speak("What would you like to name this recipe?")
            
            # Main state loop
            while self.running and self.state != RecipeState.COMPLETED and self.state != RecipeState.CANCELLED:
                # Process based on current state
                if self.state == RecipeState.NAME_PROMPT:
                    # Waiting for input handled by _handle_speech
                    time.sleep(0.1)
                    
                elif self.state == RecipeState.NAME_CONFIRM:
                    self.speech_processor.speak(f"I heard {self.recipe_data.name}. Is that correct?")
                    # Waiting for confirmation handled by _handle_command
                    time.sleep(0.1)
                    
                elif self.state == RecipeState.DESCRIPTION_PROMPT:
                    self.speech_processor.speak("Please describe the recipe, including ingredients and steps.")
                    self.state = RecipeState.RECORDING
                    
                elif self.state == RecipeState.RECORDING:
                    # Record audio for recipe description
                    success, description = self.speech_processor.recognize_speech(timeout=30.0)
                    
                    if success and description:
                        self.recipe_data.description = description
                        self.speech_processor.speak("Got it. Processing your recipe.")
                        self.state = RecipeState.PROCESSING
                    else:
                        self.speech_processor.speak("I didn't catch that. Let's try again.")
                        self.state = RecipeState.DESCRIPTION_PROMPT
                        
                elif self.state == RecipeState.PROCESSING:
                    # Process the recipe with LLM
                    self._process_recipe_with_llm()
                    self.state = RecipeState.DOCUMENT_CREATION
                    
                elif self.state == RecipeState.DOCUMENT_CREATION:
                    # Create the document
                    success = self._create_recipe_document()
                    
                    if success:
                        self.speech_processor.speak("Recipe document created successfully.")
                        self.state = RecipeState.COMPLETED
                    else:
                        self.speech_processor.speak("There was an error creating the document.")
                        self.state = RecipeState.CANCELLED
                        
                else:
                    # Unknown state
                    logger.error(f"Unknown recipe state: {self.state}")
                    self.state = RecipeState.CANCELLED
                    
            # Finalize
            if self.state == RecipeState.COMPLETED:
                self.speech_processor.speak(f"Recipe {self.recipe_data.name} has been created and saved.")
                
            elif self.state == RecipeState.CANCELLED:
                self.speech_processor.speak("Recipe creation cancelled.")
                
            # Signal completion to main app
            self.event_bus.publish("utility_completed", "recipe_creator")
            
        except Exception as e:
            logger.error(f"Error in recipe workflow: {e}")
            self.speech_processor.speak("An error occurred during recipe creation.")
            self.event_bus.publish("utility_error", {"utility": "recipe_creator", "error": str(e)})
            
        finally:
            self.running = False
            
    def _handle_speech(self, speech_data):
        """
        Handle speech recognition results.
        
        Parameters
        ----------
        speech_data : dict
            Dictionary containing recognized speech
        """
        if not self.running:
            return
            
        text = speech_data.get("text", "")
        
        if self.state == RecipeState.NAME_PROMPT and text:
            self.recipe_data.name = text
            self.state = RecipeState.NAME_CONFIRM
            
    def _handle_command(self, command):
        """
        Handle commands during recipe creation.
        
        Parameters
        ----------
        command : str
            Command string
        """
        if not self.running:
            return
            
        command = command.lower()
        
        if self.state == RecipeState.NAME_CONFIRM:
            if any(word in command for word in ["yes", "correct", "right", "yeah"]):
                self.state = RecipeState.DESCRIPTION_PROMPT
            elif any(word in command for word in ["no", "wrong", "incorrect", "nope"]):
                self.recipe_data.name = ""
                self.state = RecipeState.NAME_PROMPT
                self.speech_processor.speak("Let's try again. What would you like to name this recipe?")
                
        elif "cancel" in command:
            self.state = RecipeState.CANCELLED
            
    def _process_recipe_with_llm(self):
        """Process the recipe description with the LLM."""
        try:
            prompt = f"""
            Parse the following recipe description into a structured format.
            Extract:
            1. A list of ingredients with quantities
            2. A list of numbered steps for preparation
            
            Recipe: {self.recipe_data.description}
            
            Format your answer as:
            INGREDIENTS:
            - ingredient 1
            - ingredient 2
            ...
            
            STEPS:
            1. step 1
            2. step 2
            ...
            """
            
            response = self.llm_processor.generate_text(prompt)
            
            # Parse the response
            ingredients = []
            steps = []
            
            current_section = None
            for line in response.split('\n'):
                line = line.strip()
                
                if line == "INGREDIENTS:":
                    current_section = "ingredients"
                elif line == "STEPS:":
                    current_section = "steps"
                elif current_section == "ingredients" and line.startswith("-"):
                    ingredients.append(line[1:].strip())
                elif current_section == "steps" and (line[0].isdigit() and line[1] == "."):
                    steps.append(line[2:].strip())
                    
            self.recipe_data.ingredients = ingredients
            self.recipe_data.steps = steps
            
            logger.info(f"Processed recipe: {len(ingredients)} ingredients, {len(steps)} steps")
            
        except Exception as e:
            logger.error(f"Error processing recipe with LLM: {e}")
            # Use raw description if processing fails
            self.recipe_data.ingredients = ["Failed to process ingredients"]
            self.recipe_data.steps = [self.recipe_data.description]
            
    def _create_recipe_document(self):
        """
        Create a recipe document using python-docx.
        
        Returns
        -------
        bool
            True if document was created successfully
        """
        try:
            # Start with template if exists, otherwise create new
            if os.path.exists(self.template_path):
                doc = docx.Document(self.template_path)
            else:
                doc = docx.Document()
                
            # Add title
            doc.add_heading(self.recipe_data.name, level=1)
            
            # Add ingredients section
            doc.add_heading("Ingredients", level=2)
            for ingredient in self.recipe_data.ingredients:
                doc.add_paragraph(ingredient, style='ListBullet')
                
            # Add steps section
            doc.add_heading("Instructions", level=2)
            for i, step in enumerate(self.recipe_data.steps, 1):
                doc.add_paragraph(f"{i}. {step}")
                
            # Create safe filename
            safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in self.recipe_data.name)
            filename = f"{safe_name}_{int(time.time())}.docx"
            filepath = os.path.join(self.output_dir, filename)
            
            # Save the document
            doc.save(filepath)
            logger.info(f"Recipe document saved to {filepath}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating recipe document: {e}")
            return False
