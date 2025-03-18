"""
Maggie AI Assistant - Recipe Creator Extension
=========================================
Speech-to-document recipe creation extension.

This module provides a streamlined workflow for creating recipe documents
from speech input, with specific optimizations for AMD Ryzen 9 5900X
and NVIDIA GeForce RTX 3080 hardware. It implements a state machine approach to
guide users through the recipe creation process:

1. Recipe name collection with confirmation
2. Extended recipe description recording
3. Processing with LLM to extract structured recipe data
   - Ingredient list with quantities
   - Ordered preparation steps
   - Additional notes and tips
4. Document generation with proper formatting
5. Saving to Microsoft Word (.docx) format

The extension leverages thread-safe design, efficient speech processing,
and optimized LLM inference to provide a responsive user experience.

Examples
--------
>>> from maggie.extensions.recipe_creator import RecipeCreator
>>> from maggie.core import EventBus
>>> event_bus = EventBus()
>>> config = {"output_dir": "recipes", "template_path": "templates/recipe_template.docx"}
>>> recipe_creator = RecipeCreator(event_bus, config)
>>> recipe_creator.initialize()
>>> recipe_creator.start()
>>> # This will start an interactive workflow to create a recipe
>>> # Later, to stop
>>> recipe_creator.stop()
"""

# Standard library imports
import os
import time
import threading
import urllib.request
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union

# Third-party imports
import docx
from loguru import logger

# Local imports
from maggie.extensions.base import ExtensionBase

__all__ = ['RecipeState', 'RecipeData', 'RecipeCreator']

class RecipeState(Enum):
    """
    States for the recipe creation process.
    
    Defines the possible states in the recipe creation workflow,
    supporting a structured step-by-step process.
    
    Attributes
    ----------
    INITIAL : enum
        Initial state before starting, no data collected
    NAME_INPUT : enum
        Getting recipe name from user with speech recognition and confirmation
    DESCRIPTION : enum
        Getting detailed recipe description from user (ingredients, steps, tips)
    PROCESSING : enum
        Processing description with LLM to extract structured recipe data
    CREATING : enum
        Creating and formatting document with extracted recipe information
    COMPLETED : enum
        Process completed successfully, document saved to output directory
    CANCELLED : enum
        Process cancelled by user or due to error
    ERROR : enum
        Error state for handling recoverable failures
        
    Examples
    --------
    >>> state = RecipeState.INITIAL
    >>> state = RecipeState.NAME_INPUT
    >>> print(f"Current state: {state.name}")
    Current state: NAME_INPUT
    >>> state = RecipeState.PROCESSING
    >>> print(f"Now processing: {state.name}")
    Now processing: PROCESSING
    """
    INITIAL = auto()      # Initial state
    NAME_INPUT = auto()   # Getting recipe name
    DESCRIPTION = auto()  # Getting recipe description
    PROCESSING = auto()   # Processing with LLM
    CREATING = auto()     # Creating document
    COMPLETED = auto()    # Process completed
    CANCELLED = auto()    # Process cancelled
    ERROR = auto()        # Error state

@dataclass
class RecipeData:
    """
    Data structure for recipe information.
    
    Parameters
    ----------
    name : str
        Recipe name collected from user via speech recognition
    description : str
        Raw recipe description collected from user via speech recognition
    ingredients : List[str]
        Parsed list of ingredients with quantities extracted by LLM
    steps : List[str]
        Parsed list of preparation steps in order extracted by LLM
    notes : str
        Additional notes, tips, or variations extracted by LLM
        
    Examples
    --------
    >>> recipe = RecipeData(
    ...     name="Classic Chocolate Chip Cookies",
    ...     description="Mix flour, sugar, and chocolate chips...",
    ...     ingredients=["2 cups all-purpose flour", "1 cup chocolate chips"],
    ...     steps=["Preheat oven to 375°F", "Mix dry ingredients"],
    ...     notes="For softer cookies, reduce baking time by 2 minutes"
    ... )
    >>> print(f"Recipe: {recipe.name}")
    Recipe: Classic Chocolate Chip Cookies
    >>> print(f"First ingredient: {recipe.ingredients[0]}")
    First ingredient: 2 cups all-purpose flour
    """
    name: str = ""
    description: str = ""
    ingredients: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    notes: str = ""

class RecipeCreator(ExtensionBase):
    """
    Recipe Creator extension for creating and formatting recipes from speech.
    
    A streamlined extension for creating recipe documents from speech input,
    with specific optimizations for processing speed and document formatting.
    
    Parameters
    ----------
    event_bus : EventBus
        Reference to the central event bus
    config : Dict[str, Any]
        Configuration dictionary with recipe creator settings
        
    Attributes
    ----------
    state : RecipeState
        Current state in the recipe creation workflow
    recipe_data : RecipeData
        Current recipe data being processed
    output_dir : str
        Directory to save recipe documents
    template_path : str
        Path to the recipe document template
    speech_processor : Any
        Reference to the speech processor component
    llm_processor : Any
        Reference to the LLM processor component
    _workflow_thread : Optional[threading.Thread]
        Thread for running the recipe creation workflow
    _retry_count : int
        Counter for retry attempts on speech recognition
    _max_retries : int
        Maximum number of retries for error recovery
    """
    
    def __init__(self, event_bus, config: Dict[str, Any]):
        """
        Initialize the Recipe Creator extension.
        
        Parameters
        ----------
        event_bus : EventBus
            Reference to the central event bus
        config : Dict[str, Any]
            Configuration dictionary with recipe creator settings
            
        Notes
        -----
        Sets up initial state, configures paths, and ensures required directories
        exist for operation. Establishes recipe data tracking and prepares for
        the creation workflow.
        """
        super().__init__(event_bus, config)
        
        # Initialize attributes
        self.state = RecipeState.INITIAL
        self.recipe_data = RecipeData()
        self.output_dir = config.get("output_dir", "recipes")
        self.template_path = config.get("template_path", "templates/recipe_template.docx")
        
        # Processing thread
        self._workflow_thread = None
        
        # Component references - will be set during initialize()
        self.speech_processor = None
        self.llm_processor = None
        
        # Error handling and retry logic
        self._retry_count = 0
        self._max_retries = config.get("max_retries", 3)
        
        # Speech recognition settings
        self.speech_timeout = config.get("speech_timeout", 30.0)
        
        # Ensure directories exist
        self._ensure_directories()
        
    def _ensure_directories(self) -> None:
        """
        Ensure required directories exist.
        
        Creates output directory and template directory if they don't exist.
        Also creates template if it doesn't exist and verifies TTS model availability.
        
        Returns
        -------
        None
        
        Notes
        -----
        Handles potential IOError and other exceptions when creating directories,
        logging appropriate error messages if directories cannot be created.
        """
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.dirname(self.template_path), exist_ok=True)
            
            # Create template if it doesn't exist
            if not os.path.exists(self.template_path):
                self._create_template()
                
            # Verify TTS model exists - attempt to download if missing
            self._verify_tts_model()
                
        except IOError as io_error:
            logger.error(f"IO error creating directories: {io_error}")
        except Exception as general_error:
            logger.error(f"Error creating directories: {general_error}")
            
    def _verify_tts_model(self) -> bool:
        """
        Verify that the required TTS voice model is available.
        
        Checks for the TTS voice model file and attempts to download it if missing.
        Supports various path combinations to handle case sensitivity and alternative
        directory structures.
        
        Returns
        -------
        bool
            True if model is available or successfully downloaded, False otherwise
            
        Notes
        -----
        This is a preventive measure to ensure the TTS voice model is available
        before attempting to use it, avoiding runtime errors.
        """
        # Expected model locations (handle multiple possible paths)
        model_name = "af_heart.pt"
        possible_paths = [
            os.path.join("maggie", "models", "tts", model_name),
            os.path.join("maggie", "models", "TTS", model_name),
            os.path.join("models", "tts", model_name),
            os.path.join("models", "TTS", model_name),
        ]
        
        # Check if model exists in any of the possible locations
        model_exists = False
        for path in possible_paths:
            if os.path.exists(path):
                logger.info(f"TTS model found at {path}")
                model_exists = True
                break
        
        # If model doesn't exist, try to download it
        if not model_exists:
            logger.warning(f"TTS model not found. Attempting to download...")
            return self._download_tts_model()
            
        return model_exists
    
    def _download_tts_model(self) -> bool:
        """
        Download the TTS voice model if it's missing.
        
        Downloads the af_heart.pt model from the Hugging Face repository and
        saves it to the expected locations.
        
        Returns
        -------
        bool
            True if download was successful, False otherwise
            
        Notes
        -----
        Creates necessary directories and handles download failures gracefully.
        Adds robustness by downloading to multiple potential locations to handle
        case sensitivity issues.
        """
        model_url = "https://huggingface.co/hexgrad/Kokoro-82M/resolve/main/voices/af_heart.pt"
        download_paths = [
            os.path.join("maggie", "models", "tts", "af_heart.pt"),
            os.path.join("maggie", "models", "TTS", "af_heart.pt"),
        ]
        
        try:
            # Ensure directories exist
            for path in download_paths:
                os.makedirs(os.path.dirname(path), exist_ok=True)
            
            # Download to first path
            logger.info(f"Downloading TTS model from {model_url}...")
            urllib.request.urlretrieve(model_url, download_paths[0])
            
            # Copy to second path for redundancy
            if os.path.exists(download_paths[0]):
                import shutil
                shutil.copy2(download_paths[0], download_paths[1])
                logger.info(f"TTS model downloaded successfully")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to download TTS model: {e}")
            logger.error(f"Voice synthesis may not work. Please download manually to {download_paths[0]}")
            return False
    
    def get_trigger(self) -> str:
        """
        Get the trigger phrase for this extension.
        
        Returns
        -------
        str
            Trigger phrase that activates this extension
            
        Notes
        -----
        This phrase is what the user needs to say to activate the recipe creator
        utility when the system is in the READY state.
        """
        return "new recipe"
    
    def initialize(self) -> bool:
        """
        Initialize the Recipe Creator.
        
        Acquires references to required components like speech processor and LLM,
        and verifies that all prerequisites are met for operation.
        
        Returns
        -------
        bool
            True if initialization successful, False otherwise
            
        Notes
        -----
        Uses the service locator pattern to acquire component references,
        performing validation to ensure they are correctly obtained.
        """
        if self._initialized:
            return True
            
        try:
            # Find the main app with speech and LLM processors
            success = self._acquire_component_references()
            
            # Check if components were found
            if not success:
                logger.error("Failed to acquire speech or LLM processor references")
                return False
            
            # Verify TTS model
            if not self._verify_tts_model():
                logger.warning("TTS model verification failed - voice output may be unavailable")
                # Continue anyway as this is non-critical
            
            self._initialized = True
            logger.info("Recipe Creator initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing Recipe Creator: {e}")
            return False
    
    def _acquire_component_references(self) -> bool:
        """
        Acquire references to required components using ServiceLocator.
        
        Retrieves speech processor and LLM processor references from the service
        locator to enable voice interaction and language model processing.
        
        Returns
        -------
        bool
            True if all required component references were acquired, False otherwise
            
        Notes
        -----
        Critical for functionality - both components must be available for the
        recipe creator to work correctly.
        """
        # Get speech processor from service locator
        self.speech_processor = self.get_service("speech_processor")
        
        # Get LLM processor from service locator
        self.llm_processor = self.get_service("llm_processor")
        
        # Check if both services were found
        return self.speech_processor is not None and self.llm_processor is not None

    def start(self) -> bool:
        """
        Start the Recipe Creator workflow.
        
        Initiates the recipe creation workflow in a separate thread, beginning the
        step-by-step process of creating a recipe document from voice input.
        
        Returns
        -------
        bool
            True if started successfully, False otherwise
            
        Notes
        -----
        Thread-based approach allows the main application to remain responsive
        during the recipe creation process.
        """
        try:
            # Reset state and data
            self.state = RecipeState.INITIAL
            self.recipe_data = RecipeData()
            
            # Check if already running
            if self.running:
                logger.warning("Recipe Creator already running")
                return False
            
            # Initialize components if needed
            if not self.initialized and not self.initialize():
                logger.error("Failed to initialize Recipe Creator")
                return False
            
            # Start workflow thread
            self._workflow_thread = threading.Thread(
                target=self._workflow,
                name="RecipeWorkflow"
            )
            self._workflow_thread.daemon = True
            self._workflow_thread.start()
            
            self.running = True
            logger.info("Recipe Creator started")
            return True
            
        except Exception as e:
            logger.error(f"Error starting Recipe Creator: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the Recipe Creator.
        
        Cancels the recipe creation process and cleans up resources,
        transitioning to the CANCELLED state.
        
        Returns
        -------
        bool
            True if stopped successfully, False otherwise
            
        Notes
        -----
        Gracefully terminates the workflow thread and releases resources.
        """
        if not self.running:
            return True
            
        try:
            # Set state to CANCELLED to stop workflow
            self.state = RecipeState.CANCELLED
            self.running = False
            
            # Wait for thread to finish
            if self._workflow_thread and self._workflow_thread.is_alive():
                self._workflow_thread.join(timeout=2.0)
            
            logger.info("Recipe Creator stopped")
            return True
            
        except Exception as e:
            logger.error(f"Error stopping Recipe Creator: {e}")
            return False
    
    def process_command(self, command: str) -> bool:
        """
        Process a command directed to this extension.
        
        Handles user commands during the recipe creation workflow,
        particularly for confirmation and cancellation.
        
        Parameters
        ----------
        command : str
            Command string to process
            
        Returns
        -------
        bool
            True if command processed, False otherwise
            
        Notes
        -----
        Supports cancel commands at any state and affirmation/negation
        during the NAME_INPUT state for confirmation.
        """
        if not self.running:
            return False
            
        command = command.lower().strip()
        
        # Handle cancel command
        if "cancel" in command or command in ["stop", "quit", "exit"]:
            logger.info("Recipe creation cancelled by user")
            self.state = RecipeState.CANCELLED
            return True
            
        # Handle name confirmation
        if self.state == RecipeState.NAME_INPUT:
            if command in ["yes", "correct", "right", "yeah", "yep", "sure", "okay"]:
                logger.info(f"Recipe name confirmed: {self.recipe_data.name}")
                self.state = RecipeState.DESCRIPTION
                return True
            elif command in ["no", "wrong", "incorrect", "nope"]:
                logger.info("Recipe name rejected, asking again")
                self.recipe_data.name = ""
                return True
                
        return False
    
    def _workflow(self) -> None:
        """
        Main recipe creation workflow.
        
        Manages the state machine for the recipe creation process,
        guiding the user through each step of creating a recipe.
        
        Returns
        -------
        None
        
        Notes
        -----
        Implements a robust state machine that handles transitions between
        states, error recovery, and resource management throughout the process.
        """
        try:
            # Welcome message
            self._speak_safely("Starting recipe creator. Let's create a new recipe.")
            
            # State machine
            while self.running and self.state not in [RecipeState.COMPLETED, RecipeState.CANCELLED]:
                # Process current state
                self._process_current_state()
                
                # Brief pause to prevent tight loop
                time.sleep(0.1)
            
            # Finalize
            self._finalize_workflow()
            
        except Exception as e:
            logger.error(f"Error in recipe workflow: {e}")
            self._speak_safely("An error occurred while creating the recipe.")
            self.event_bus.publish("extension_error", "recipe_creator")
            
        finally:
            # Ensure speech processor is stopped if we started it
            if hasattr(self, 'speech_processor') and self.speech_processor:
                try:
                    self.speech_processor.stop_listening()
                except:
                    pass
                    
            self.running = False
    
    def _speak_safely(self, text: str) -> bool:
        """
        Safely use the speech processor to speak text with error handling.
        
        Wraps the speech processor's speak method with error handling to prevent
        crashes when the TTS model is unavailable or other issues occur.
        
        Parameters
        ----------
        text : str
            Text to be spoken
            
        Returns
        -------
        bool
            True if speech was successful, False if an error occurred
            
        Notes
        -----
        Falls back to logging the message if speech fails, ensuring the workflow
        can continue even without voice output.
        """
        try:
            if self.speech_processor:
                success = self.speech_processor.speak(text)
                if not success:
                    logger.warning(f"TTS failed, fallback to log: {text}")
                return success
            else:
                logger.warning(f"Speech processor unavailable: {text}")
                return False
        except Exception as e:
            logger.error(f"Error in speech synthesis: {e}")
            logger.info(f"Would have said: {text}")
            return False
    
    def _process_current_state(self) -> None:
        """
        Process the current workflow state.
        
        Handles the appropriate action based on the current state
        in the recipe creation workflow.
        
        Returns
        -------
        None
        
        Notes
        -----
        Central dispatcher for the state machine, directing execution to the
        appropriate handler for each state.
        """
        if self.state == RecipeState.INITIAL:
            # Start getting the recipe name
            self.state = RecipeState.NAME_INPUT
            
        elif self.state == RecipeState.NAME_INPUT:
            # Get recipe name
            self._process_name_input()
            
        elif self.state == RecipeState.DESCRIPTION:
            # Get recipe description
            self._process_description_input()
            
        elif self.state == RecipeState.PROCESSING:
            # Process with LLM
            success = self._process_with_llm()
            
            if success:
                self.state = RecipeState.CREATING
            else:
                # Handle error with retry or state transition
                self._retry_count += 1
                if self._retry_count <= self._max_retries:
                    self._speak_safely("There was an error processing your recipe. Let's try describing it again.")
                    self.state = RecipeState.DESCRIPTION
                else:
                    self._speak_safely("I'm having trouble processing the recipe after multiple attempts. Let's cancel and try again later.")
                    self.state = RecipeState.CANCELLED
            
        elif self.state == RecipeState.CREATING:
            # Create document
            success = self._create_document()
            
            if success:
                self._speak_safely(f"Recipe '{self.recipe_data.name}' has been created and saved.")
                self.state = RecipeState.COMPLETED
            else:
                self._speak_safely("There was an error creating the document.")
                self.state = RecipeState.CANCELLED
                
        elif self.state == RecipeState.ERROR:
            # Handle error state with retry mechanism
            self._handle_error_state()
    
    def _process_name_input(self) -> None:
        """
        Process the name input state.
        
        Gets the recipe name from the user and asks for confirmation.
        
        Returns
        -------
        None
        
        Notes
        -----
        Ensures speech processor is listening before attempting to recognize speech,
        improving reliability and preventing errors.
        """
        # Ensure we're listening before trying to recognize speech
        if self.speech_processor and not getattr(self.speech_processor, 'listening', False):
            self.speech_processor.start_listening()
            logger.info("Started speech processor listening")
            
        if not self.recipe_data.name:
            self._speak_safely("What would you like to name this recipe?")
            success, name = self._recognize_speech_safely(timeout=10.0)
            
            if success and name:
                self.recipe_data.name = name
                self._speak_safely(f"I heard {name}. Is that correct?")
            else:
                self._speak_safely("I didn't catch that. Let's try again.")
    
    def _recognize_speech_safely(self, timeout: float = 10.0) -> Tuple[bool, str]:
        """
        Safely recognize speech with error handling and recovery.
        
        Wraps the speech recognition process with additional error handling and
        retry capabilities to improve reliability.
        
        Parameters
        ----------
        timeout : float, optional
            Maximum time to listen for speech, by default 10.0 seconds
            
        Returns
        -------
        Tuple[bool, str]
            Success status and recognized text (empty string if failed)
            
        Notes
        -----
        Implements retries, error logging, and auto-starts listening if needed,
        creating a more robust speech recognition process.
        """
        try:
            # Ensure listening is active
            if self.speech_processor and not getattr(self.speech_processor, 'listening', False):
                try:
                    self.speech_processor.start_listening()
                    time.sleep(0.5)  # Brief pause to ensure listening is fully started
                    logger.info("Started speech processor listening")
                except Exception as listen_error:
                    logger.error(f"Error starting listening: {listen_error}")
            
            # Attempt speech recognition
            if self.speech_processor:
                return self.speech_processor.recognize_speech(timeout=timeout)
            else:
                logger.error("Speech processor not available")
                return False, ""
                
        except Exception as e:
            logger.error(f"Error recognizing speech: {e}")
            return False, ""
    
    def _process_description_input(self) -> None:
        """
        Process the description input state.
        
        Gets the recipe description from the user with comprehensive instructions
        and enhanced timeout for detailed descriptions.
        
        Returns
        -------
        None
        
        Notes
        -----
        Uses longer timeout value to allow for detailed recipe descriptions and
        provides clear instructions to improve input quality.
        """
        # Give detailed instructions for better results
        self._speak_safely(
            "Please describe the recipe in detail, including ingredients with quantities, "
            "preparation steps in order, and any tips or variations. I'll listen for up to "
            "30 seconds, so take your time."
        )
        
        # Use longer timeout for detailed descriptions
        success, description = self._recognize_speech_safely(timeout=self.speech_timeout)
        
        if success and description:
            self.recipe_data.description = description
            self._speak_safely("Got it. Processing your recipe.")
            self.state = RecipeState.PROCESSING
        else:
            # Handle failure with retry logic
            self._retry_count += 1
            if self._retry_count <= self._max_retries:
                self._speak_safely("I didn't catch that. Let's try again.")
            else:
                self._speak_safely("I'm having trouble understanding. Let's try again later.")
                self.state = RecipeState.CANCELLED
    
    def _handle_error_state(self) -> None:
        """
        Handle error state with appropriate recovery actions.
        
        Implements retry logic and appropriate error messaging when the workflow
        encounters recoverable errors.
        
        Returns
        -------
        None
        
        Notes
        -----
        Uses retry count to determine whether to attempt recovery or transition
        to a cancelled state.
        """
        # Increment retry counter
        self._retry_count += 1
        
        # Check if we should retry or give up
        if self._retry_count <= self._max_retries:
            self._speak_safely("I encountered an issue. Let's try again.")
            
            # Return to previous state or description state for retry
            self.state = RecipeState.DESCRIPTION
        else:
            self._speak_safely("I'm having trouble completing this recipe after several attempts. Let's try again later.")
            self.state = RecipeState.CANCELLED
    
    def _finalize_workflow(self) -> None:
        """
        Finalize the recipe creation workflow.
        
        Handles final messages and events when the workflow completes or is cancelled.
        
        Returns
        -------
        None
        
        Notes
        -----
        Ensures appropriate feedback is given to the user and the correct completion
        events are published to the event bus.
        """
        if self.state == RecipeState.CANCELLED:
            self._speak_safely("Recipe creation cancelled.")
        elif self.state == RecipeState.COMPLETED:
            self._speak_safely("Recipe creation completed successfully.")
            
        # Publish completion event
        self.event_bus.publish("extension_completed", "recipe_creator")
    
    def _process_with_llm(self) -> bool:
        """
        Process recipe description with LLM.
        
        Uses the language model to extract structured recipe information
        from the user's natural language description.
        
        Returns
        -------
        bool
            True if processing successful, False otherwise
            
        Notes
        -----
        Formats a specialized prompt for the LLM that encourages extraction of
        structured data from unstructured user input.
        """
        try:
            # Create prompt for LLM
            prompt = self._create_llm_prompt()
            
            # Process with LLM
            response = self.llm_processor.generate_text(prompt, max_tokens=1024)
            
            if not response:
                logger.error("Empty response from LLM")
                return False
                
            # Parse response
            self._parse_llm_response(response)
            
            logger.info(f"Recipe processed: {len(self.recipe_data.ingredients)} ingredients, {len(self.recipe_data.steps)} steps")
            return True
            
        except Exception as e:
            logger.error(f"Error processing recipe with LLM: {e}")
            return False
    
    def _create_llm_prompt(self) -> str:
        """
        Create prompt for the language model.
        
        Formats an optimized prompt that instructs the LLM how to parse the recipe
        description into structured format.
        
        Returns
        -------
        str
            Formatted prompt for LLM to process recipe description
            
        Notes
        -----
        Uses specific formatting instructions that help the LLM provide well-structured
        output that can be easily parsed.
        """
        return f"""
        Parse the following recipe description into structured format:
        
        Recipe: {self.recipe_data.description}
        
        Extract and format as follows:
        
        INGREDIENTS:
        - [ingredient with quantity]
        - [ingredient with quantity]
        ...
        
        STEPS:
        1. [step 1]
        2. [step 2]
        ...
        
        NOTES:
        [any additional notes or tips]
        """
    
    def _parse_llm_response(self, response: str) -> None:
        """
        Parse the LLM response into structured recipe data.
        
        Analyzes the LLM's output to extract ingredients, steps, and notes,
        populating the recipe data structure.
        
        Parameters
        ----------
        response : str
            Text response from LLM
            
        Returns
        -------
        None
        
        Notes
        -----
        Uses a robust parsing approach that handles variations in the LLM's output
        format while correctly categorizing content.
        """
        current_section = None
        ingredients = []
        steps = []
        notes = []
        
        for line in response.strip().split('\n'):
            line = line.strip()
            
            if not line:
                continue
                
            if line == "INGREDIENTS:" or line.startswith("INGREDIENTS:"):
                current_section = "ingredients"
                continue
            elif line == "STEPS:" or line.startswith("STEPS:"):
                current_section = "steps"
                continue
            elif line == "NOTES:" or line.startswith("NOTES:"):
                current_section = "notes"
                continue
                
            if current_section == "ingredients" and (line.startswith("-") or line.startswith("•")):
                ingredients.append(line[1:].strip())
            elif current_section == "steps" and (line[0].isdigit() and "." in line[:3]):
                steps.append(line[line.find(".")+1:].strip())
            elif current_section == "steps" and line.startswith("Step "):
                # Handle "Step X:" format
                steps.append(line[line.find(" ")+1:].strip())
            elif current_section == "notes":
                notes.append(line)
        
        # Update recipe data
        self.recipe_data.ingredients = ingredients
        self.recipe_data.steps = steps
        self.recipe_data.notes = "\n".join(notes)
        
        # Fallback for empty sections
        if not ingredients:
            # Try extracting ingredients using alternate parsing
            for line in response.strip().split('\n'):
                if "cup" in line or "tbsp" in line or "tsp" in line or "gram" in line or "oz" in line:
                    if line not in ingredients:
                        ingredients.append(line.strip())
            
            self.recipe_data.ingredients = ingredients
    
    def _create_document(self) -> bool:
        """
        Create recipe document.
        
        Generates a formatted Microsoft Word document using python-docx
        based on the processed recipe data.
        
        Returns
        -------
        bool
            True if document created successfully, False otherwise
            
        Notes
        -----
        Implements robust error handling to manage docx creation failures and
        ensures proper template existence.
        """
        try:
            # Load template or create new document
            if os.path.exists(self.template_path):
                doc = docx.Document(self.template_path)
            else:
                self._create_template()
                doc = docx.Document(self.template_path)
            
            # Clear template content and add recipe content
            self._populate_document(doc)
            
            # Create safe filename and save document
            filepath = self._save_document(doc)
            
            logger.info(f"Recipe document saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating recipe document: {e}")
            return False
    
    def _populate_document(self, doc: docx.Document) -> None:
        """
        Populate the document with recipe content.
        
        Parameters
        ----------
        doc : docx.Document
            Document to populate with the recipe data
            
        Returns
        -------
        None
        
        Notes
        -----
        Carefully formats the recipe content using appropriate styles and section
        headings for a professional document appearance.
        """
        # Clear template content
        for paragraph in doc.paragraphs:
            if paragraph.text and paragraph.text.strip():
                paragraph.text = ""
        
        # Add recipe title
        doc.add_heading(self.recipe_data.name, level=1)
        
        # Add ingredients section
        doc.add_heading("Ingredients", level=2)
        for ingredient in self.recipe_data.ingredients:
            doc.add_paragraph(f"• {ingredient}", style='ListBullet')
        
        # Add steps section
        doc.add_heading("Instructions", level=2)
        for i, step in enumerate(self.recipe_data.steps, 1):
            doc.add_paragraph(f"{i}. {step}", style='ListNumber')
        
        # Add notes section if available
        if self.recipe_data.notes:
            doc.add_heading("Notes", level=2)
            doc.add_paragraph(self.recipe_data.notes)
    
    def _save_document(self, doc: docx.Document) -> str:
        """
        Save the document to a file.
        
        Parameters
        ----------
        doc : docx.Document
            Document to save
            
        Returns
        -------
        str
            Path to the saved file
            
        Notes
        -----
        Creates a safe filename based on the recipe name and current timestamp
        to ensure uniqueness and prevent path issues.
        """
        # Create safe filename
        safe_name = "".join(c if c.isalnum() or c in " -_" else "_" for c in self.recipe_data.name)
        filename = f"{safe_name}_{int(time.time())}.docx"
        filepath = os.path.join(self.output_dir, filename)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save document
        doc.save(filepath)
        return filepath
    
    def _create_template(self) -> bool:
        """
        Create recipe template document.
        
        Generates a template document for recipes if it doesn't exist.
        
        Returns
        -------
        bool
            True if template created successfully, False otherwise
            
        Notes
        -----
        Creates a professional template with sections for recipe information,
        ingredients, instructions, and notes.
        """
        try:
            # Create template directory if it doesn't exist
            os.makedirs(os.path.dirname(self.template_path), exist_ok=True)
            
            # Create template document
            doc = docx.Document()
            
            # Add title placeholder
            doc.add_heading("Recipe Name", level=1)
            
            # Add metadata section
            doc.add_heading("Recipe Information", level=2)
            table = doc.add_table(rows=3, cols=2)
            table.style = 'Table Grid'
            
            # Set table headers and defaults
            cells = table.rows[0].cells
            cells[0].text = "Preparation Time"
            cells[1].text = "00 minutes"
            
            cells = table.rows[1].cells
            cells[0].text = "Cooking Time"
            cells[1].text = "00 minutes"
            
            cells = table.rows[2].cells
            cells[0].text = "Servings"
            cells[1].text = "0 servings"
            
            # Add ingredients section
            doc.add_heading("Ingredients", level=2)
            doc.add_paragraph("• Ingredient 1", style='ListBullet')
            doc.add_paragraph("• Ingredient 2", style='ListBullet')
            doc.add_paragraph("• Ingredient 3", style='ListBullet')
            
            # Add instructions section
            doc.add_heading("Instructions", level=2)
            doc.add_paragraph("1. Step 1", style='ListNumber')
            doc.add_paragraph("2. Step 2", style='ListNumber')
            doc.add_paragraph("3. Step 3", style='ListNumber')
            
            # Add notes section
            doc.add_heading("Notes", level=2)
            doc.add_paragraph("Add any additional notes, tips, or variations here.")
            
            # Save template
            doc.save(self.template_path)
            
            logger.info(f"Recipe template created at {self.template_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error creating recipe template: {e}")
            return False