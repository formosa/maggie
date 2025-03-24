"""
Maggie AI Assistant - Recipe Creator Extension
=============================================

Speech-to-document recipe creation extension for Maggie AI Assistant.

This extension provides a streamlined workflow for dictating recipes 
and converting them to formatted document files.
"""

import os
import time
import threading
from enum import Enum, auto
from dataclasses import dataclass, field
from typing import Dict, Any, Optional, List, Tuple, Union

import docx

from maggie.extensions.base import ExtensionBase
from maggie.utils.error_handling import (
    safe_execute, retry_operation, ErrorCategory, ErrorSeverity,
    with_error_handling, record_error, ExtensionError
)
from maggie.utils.logging import ComponentLogger, log_operation, logging_context
from maggie.service.locator import ServiceLocator

__all__ = ['RecipeState', 'RecipeData', 'RecipeCreator']


class RecipeState(Enum):
    """State machine states for the recipe creation workflow."""
    INITIAL = auto()
    NAME_INPUT = auto()
    DESCRIPTION = auto()
    PROCESSING = auto()
    CREATING = auto()
    COMPLETED = auto()
    CANCELLED = auto()
    ERROR = auto()


@dataclass
class RecipeData:
    """Data container for recipe information."""
    name: str = ''
    description: str = ''
    ingredients: List[str] = field(default_factory=list)
    steps: List[str] = field(default_factory=list)
    notes: str = ''


class RecipeCreator(ExtensionBase):
    """
    Recipe Creator extension for Maggie AI Assistant.
    
    This extension allows users to create recipe documents through speech input,
    using natural language processing to structure the recipe content.
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
        """
        super().__init__(event_bus, config)
        
        # Initialize state and data
        self.state = RecipeState.INITIAL
        self.recipe_data = RecipeData()
        
        # Parse configuration
        self.output_dir = config.get('output_dir', 'recipes')
        self.template_path = config.get('template_path', 'templates/recipe_template.docx')
        self._retry_count = 0
        self._max_retries = config.get('max_retries', 3)
        self.speech_timeout = config.get('speech_timeout', 30.0)
        
        # Initialize resources
        self._workflow_thread = None
        self.stt_processor = None
        self.llm_processor = None
        self.tts_processor = None
        
        # Set up logging
        self.logger = ComponentLogger('RecipeCreator')
        
        # Ensure required directories exist
        self._ensure_directories()
    
    def _ensure_directories(self) -> None:
        """Ensure required directories exist for templates and output."""
        try:
            os.makedirs(self.output_dir, exist_ok=True)
            os.makedirs(os.path.dirname(self.template_path), exist_ok=True)
            
            if not os.path.exists(self.template_path):
                self._create_template()
        except IOError as io_error:
            self.logger.error(f"IO error creating directories: {io_error}")
        except Exception as general_error:
            self.logger.error(f"Error creating directories: {general_error}")
    
    def get_trigger(self) -> str:
        """
        Get the trigger phrase for this extension.
        
        Returns
        -------
        str
            Trigger phrase that activates this extension
        """
        return 'new recipe'
    
    @log_operation(component='RecipeCreator')
    def initialize(self) -> bool:
        """
        Initialize the Recipe Creator extension.
        
        Returns
        -------
        bool
            True if initialized successfully, False otherwise
        """
        if self._initialized:
            return True
            
        try:
            success = self._acquire_component_references()
            
            if not success:
                self.logger.error('Failed to acquire required component references')
                return False
                
            self._initialized = True
            self.logger.info('Recipe Creator initialized successfully')
            return True
            
        except Exception as e:
            self.logger.error(f"Error initializing Recipe Creator: {e}")
            return False
    
    def _acquire_component_references(self) -> bool:
        """
        Acquire references to required components.
        
        Returns
        -------
        bool
            True if all required components were acquired
        """
        # Get component references from ServiceLocator
        self.stt_processor = self.get_service('stt_processor')
        self.llm_processor = self.get_service('llm_processor')
        self.tts_processor = self.get_service('tts_processor')
        
        # Check all required components are available
        return (self.stt_processor is not None and 
                self.llm_processor is not None and 
                self.tts_processor is not None)
    
    @log_operation(component='RecipeCreator')
    @with_error_handling(error_category=ErrorCategory.EXTENSION)
    def start(self) -> bool:
        """
        Start the Recipe Creator extension.
        
        Returns
        -------
        bool
            True if started successfully, False otherwise
        """
        try:
            # Reset state and data
            self.state = RecipeState.INITIAL
            self.recipe_data = RecipeData()
            
            # Check if already running
            if self.running:
                self.logger.warning('Recipe Creator already running')
                return False
                
            # Initialize if needed
            if not self.initialized and not self.initialize():
                self.logger.error('Failed to initialize Recipe Creator')
                return False
                
            # Start workflow in a separate thread
            self._workflow_thread = threading.Thread(
                target=self._workflow,
                name='RecipeWorkflow', 
                daemon=True
            )
            self._workflow_thread.start()
            
            self.running = True
            self.logger.info('Recipe Creator started')
            return True
            
        except Exception as e:
            self.logger.error(f"Error starting Recipe Creator: {e}")
            return False
    
    @log_operation(component='RecipeCreator')
    @with_error_handling(error_category=ErrorCategory.EXTENSION)
    def stop(self) -> bool:
        """
        Stop the Recipe Creator extension.
        
        Returns
        -------
        bool
            True if stopped successfully, False otherwise
        """
        if not self.running:
            return True
            
        try:
            # Cancel workflow
            self.state = RecipeState.CANCELLED
            self.running = False
            
            # Wait for workflow thread to finish
            if self._workflow_thread and self._workflow_thread.is_alive():
                self._workflow_thread.join(timeout=2.0)
                
            self.logger.info('Recipe Creator stopped')
            return True
            
        except Exception as e:
            self.logger.error(f"Error stopping Recipe Creator: {e}")
            return False
    
    def process_command(self, command: str) -> bool:
        """
        Process a command directed to this extension.
        
        Parameters
        ----------
        command : str
            Command to process
            
        Returns
        -------
        bool
            True if command was processed, False otherwise
        """
        if not self.running:
            return False
            
        command = command.lower().strip()
        
        # Handle cancel command
        if 'cancel' in command or command in ['stop', 'quit', 'exit']:
            self.logger.info('Recipe creation cancelled by user')
            self.state = RecipeState.CANCELLED
            return True
            
        # Handle name confirmation
        if self.state == RecipeState.NAME_INPUT:
            if command in ['yes', 'correct', 'right', 'yeah', 'yep', 'sure', 'okay']:
                self.logger.info(f"Recipe name confirmed: {self.recipe_data.name}")
                self.state = RecipeState.DESCRIPTION
                return True
            elif command in ['no', 'wrong', 'incorrect', 'nope']:
                self.logger.info('Recipe name rejected, asking again')
                self.recipe_data.name = ''
                return True
                
        return False
    
    @with_error_handling(error_category=ErrorCategory.EXTENSION)
    def _workflow(self) -> None:
        """Main workflow for recipe creation."""
        try:
            self._speak("Starting recipe creator. Let's create a new recipe.")
            
            while self.running and self.state not in [RecipeState.COMPLETED, RecipeState.CANCELLED]:
                # State machine implementation
                if self.state == RecipeState.INITIAL:
                    self.state = RecipeState.NAME_INPUT
                    
                elif self.state == RecipeState.NAME_INPUT:
                    self._process_name_input()
                    
                elif self.state == RecipeState.DESCRIPTION:
                    self._process_description_input()
                    
                elif self.state == RecipeState.PROCESSING:
                    success = self._process_with_llm()
                    self.state = RecipeState.CREATING if success else RecipeState.ERROR
                    
                elif self.state == RecipeState.CREATING:
                    success = self._create_document()
                    
                    if success:
                        self._speak(f"Recipe '{self.recipe_data.name}' has been created and saved.")
                        self.state = RecipeState.COMPLETED
                    else:
                        self._speak('There was an error creating the document.')
                        self.state = RecipeState.CANCELLED
                        
                elif self.state == RecipeState.ERROR:
                    self._handle_error_state()
                    
                time.sleep(0.1)
                
            self._finalize_workflow()
                
        except Exception as e:
            self.logger.error(f"Error in recipe workflow: {e}")
            self._speak('An error occurred while creating the recipe.')
            self.event_bus.publish('extension_error', 'recipe_creator')
            
        finally:
            self._cleanup_resources()
            self.running = False
    
    def _cleanup_resources(self) -> None:
        """Clean up resources used by the extension."""
        if hasattr(self, 'stt_processor') and self.stt_processor:
            try:
                self.stt_processor.stop_listening()
            except:
                pass
    
    def _process_name_input(self) -> None:
        """Process recipe name input from user."""
        if not self.recipe_data.name:
            self._speak('What would you like to name this recipe?')
            success, name = self._recognize_speech(timeout=10.0)
            
            if success and name:
                self.recipe_data.name = name
                self._speak(f"I heard {name}. Is that correct?")
            else:
                self._speak("I didn't catch that. Let's try again.")
    
    @with_error_handling(error_category=ErrorCategory.EXTENSION)
    def _recognize_speech(self, timeout: float = 10.0) -> Tuple[bool, str]:
        """
        Recognize speech using the STT processor.
        
        Parameters
        ----------
        timeout : float
            Maximum time to wait for speech input
            
        Returns
        -------
        Tuple[bool, str]
            Success flag and recognized text
        """
        try:
            # Ensure STT processor is listening
            if self.stt_processor and not getattr(self.stt_processor, 'listening', False):
                try:
                    self.stt_processor.start_listening()
                    time.sleep(0.5)
                except Exception as listen_error:
                    self.logger.error(f"Error starting listening: {listen_error}")
            
            # Recognize speech
            if self.stt_processor:
                return self.stt_processor.recognize_speech(timeout=timeout)
            else:
                self.logger.error('Speech processor not available')
                return False, ''
                
        except Exception as e:
            self.logger.error(f"Error recognizing speech: {e}")
            return False, ''
    
    def _process_description_input(self) -> None:
        """Process recipe description input from user."""
        self._speak("Please describe the recipe in detail, including ingredients with quantities, " 
                   "preparation steps in order, and any tips or variations. I'll listen for up to " 
                   "30 seconds, so take your time.")
                   
        success, description = self._recognize_speech(timeout=self.speech_timeout)
        
        if success and description:
            self.recipe_data.description = description
            self._speak('Got it. Processing your recipe.')
            self.state = RecipeState.PROCESSING
        else:
            self._retry_count += 1
            
            if self._retry_count <= self._max_retries:
                self._speak("I didn't catch that. Let's try again.")
            else:
                self._speak("I'm having trouble understanding. Let's try again later.")
                self.state = RecipeState.CANCELLED
    
    def _handle_error_state(self) -> None:
        """Handle error state with retries."""
        self._retry_count += 1
        
        if self._retry_count <= self._max_retries:
            self._speak("I encountered an issue. Let's try again.")
            self.state = RecipeState.DESCRIPTION
        else:
            self._speak("I'm having trouble completing this recipe after several attempts. "
                       "Let's try again later.")
            self.state = RecipeState.CANCELLED
    
    def _finalize_workflow(self) -> None:
        """Finalize the workflow with appropriate message and event."""
        if self.state == RecipeState.CANCELLED:
            self._speak('Recipe creation cancelled.')
        elif self.state == RecipeState.COMPLETED:
            self._speak('Recipe creation completed successfully.')
            
        self.event_bus.publish('extension_completed', 'recipe_creator')
    
    @log_operation(component='RecipeCreator')
    @with_error_handling(error_category=ErrorCategory.PROCESSING)
    def _process_with_llm(self) -> bool:
        """
        Process recipe description with LLM to extract structured data.
        
        Returns
        -------
        bool
            True if processing successful, False otherwise
        """
        try:
            # Create prompt for LLM
            prompt = self._create_llm_prompt()
            
            # Generate text using LLM
            response = self.llm_processor.generate_text(
                prompt=prompt,
                max_tokens=1024,
                temperature=0.7,
                top_p=0.95
            )
            
            if not response:
                self.logger.error('Empty response from LLM')
                return False
                
            # Parse response to extract structured data
            self._parse_llm_response(response)
            
            self.logger.info(
                f"Recipe processed: {len(self.recipe_data.ingredients)} ingredients, "
                f"{len(self.recipe_data.steps)} steps"
            )
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error processing recipe with LLM: {e}")
            return False
    
    def _create_llm_prompt(self) -> str:
        """
        Create prompt for LLM to extract recipe structure.
        
        Returns
        -------
        str
            Formatted prompt for LLM
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
        Parse LLM response to extract recipe components.
        
        Parameters
        ----------
        response : str
            Response from LLM to parse
        """
        current_section = None
        ingredients = []
        steps = []
        notes = []
        
        # Process response line by line
        for line in response.strip().split('\n'):
            line = line.strip()
            
            if not line:
                continue
                
            # Detect sections
            if 'INGREDIENTS:' in line:
                current_section = 'ingredients'
                continue
            elif 'STEPS:' in line:
                current_section = 'steps'
                continue
            elif 'NOTES:' in line:
                current_section = 'notes'
                continue
                
            # Process line based on current section
            if current_section == 'ingredients' and (line.startswith('-') or line.startswith('•')):
                ingredients.append(line[1:].strip())
            elif current_section == 'steps' and (line[0].isdigit() and '.' in line[:3]):
                steps.append(line[line.find('.')+1:].strip())
            elif current_section == 'steps' and line.startswith('Step '):
                steps.append(line[line.find(' ')+1:].strip())
            elif current_section == 'notes':
                notes.append(line)
                
        # Update recipe data
        self.recipe_data.ingredients = ingredients
        self.recipe_data.steps = steps
        self.recipe_data.notes = '\n'.join(notes)
        
        # Try fallback if no ingredients were found
        if not ingredients:
            self._extract_ingredients_fallback(response)
    
    def _extract_ingredients_fallback(self, response: str) -> None:
        """
        Fallback method to extract ingredients if standard parsing fails.
        
        Parameters
        ----------
        response : str
            Response from LLM to parse
        """
        ingredients = []
        
        # Look for lines containing common measurement units
        for line in response.strip().split('\n'):
            if any(unit in line.lower() for unit in ['cup', 'tbsp', 'tsp', 'gram', 'oz']):
                if line not in ingredients:
                    ingredients.append(line.strip())
                    
        if ingredients:
            self.recipe_data.ingredients = ingredients
    
    @log_operation(component='RecipeCreator')
    @with_error_handling(error_category=ErrorCategory.PROCESSING)
    def _create_document(self) -> bool:
        """
        Create Word document with recipe data.
        
        Returns
        -------
        bool
            True if document created successfully, False otherwise
        """
        try:
            # Load or create template
            if os.path.exists(self.template_path):
                doc = docx.Document(self.template_path)
            else:
                self._create_template()
                doc = docx.Document(self.template_path)
                
            # Populate document with recipe data
            self._populate_document(doc)
            
            # Save document
            filepath = self._save_document(doc)
            
            self.logger.info(f"Recipe document saved to {filepath}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating recipe document: {e}")
            return False
    
    def _populate_document(self, doc: docx.Document) -> None:
        """
        Populate Word document with recipe data.
        
        Parameters
        ----------
        doc : docx.Document
            Document to populate
        """
        # Clear existing content
        for paragraph in doc.paragraphs:
            if paragraph.text and paragraph.text.strip():
                paragraph.text = ''
                
        # Add recipe name
        doc.add_heading(self.recipe_data.name, level=1)
        
        # Add ingredients
        doc.add_heading('Ingredients', level=2)
        for ingredient in self.recipe_data.ingredients:
            doc.add_paragraph(f"• {ingredient}", style='ListBullet')
            
        # Add instructions
        doc.add_heading('Instructions', level=2)
        for i, step in enumerate(self.recipe_data.steps, 1):
            doc.add_paragraph(f"{i}. {step}", style='ListNumber')
            
        # Add notes if available
        if self.recipe_data.notes:
            doc.add_heading('Notes', level=2)
            doc.add_paragraph(self.recipe_data.notes)
    
    def _save_document(self, doc: docx.Document) -> str:
        """
        Save document to file with appropriate filename.
        
        Parameters
        ----------
        doc : docx.Document
            Document to save
            
        Returns
        -------
        str
            Path to saved document
        """
        # Create safe filename
        safe_name = ''.join(
            c if c.isalnum() or c in ' -_' else '_' 
            for c in self.recipe_data.name
        )
        
        filename = f"{safe_name}_{int(time.time())}.docx"
        filepath = os.path.join(self.output_dir, filename)
        
        # Ensure output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save document
        doc.save(filepath)
        
        return filepath
    
    @log_operation(component='RecipeCreator')
    @with_error_handling(error_category=ErrorCategory.SYSTEM)
    def _create_template(self) -> bool:
        """
        Create recipe document template if it doesn't exist.
        
        Returns
        -------
        bool
            True if template created successfully, False otherwise
        """
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.template_path), exist_ok=True)
            
            # Create document
            doc = docx.Document()
            
            # Add recipe name placeholder
            doc.add_heading('Recipe Name', level=1)
            
            # Add recipe information table
            doc.add_heading('Recipe Information', level=2)
            table = doc.add_table(rows=3, cols=2)
            table.style = 'Table Grid'
            
            # Add row values
            cells = table.rows[0].cells
            cells[0].text = 'Preparation Time'
            cells[1].text = '00 minutes'
            
            cells = table.rows[1].cells
            cells[0].text = 'Cooking Time'
            cells[1].text = '00 minutes'
            
            cells = table.rows[2].cells
            cells[0].text = 'Servings'
            cells[1].text = '0 servings'
            
            # Add ingredients placeholders
            doc.add_heading('Ingredients', level=2)
            doc.add_paragraph('• Ingredient 1', style='ListBullet')
            doc.add_paragraph('• Ingredient 2', style='ListBullet')
            doc.add_paragraph('• Ingredient 3', style='ListBullet')
            
            # Add instructions placeholders
            doc.add_heading('Instructions', level=2)
            doc.add_paragraph('1. Step 1', style='ListNumber')
            doc.add_paragraph('2. Step 2', style='ListNumber')
            doc.add_paragraph('3. Step 3', style='ListNumber')
            
            # Add notes placeholder
            doc.add_heading('Notes', level=2)
            doc.add_paragraph('Add any additional notes, tips, or variations here.')
            
            # Save template
            doc.save(self.template_path)
            
            self.logger.info(f"Recipe template created at {self.template_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating recipe template: {e}")
            return False
    
    def _speak(self, text: str) -> None:
        """
        Speak text using TTS processor.
        
        Parameters
        ----------
        text : str
            Text to speak
        """
        if self.tts_processor:
            try:
                self.tts_processor.speak(text)
            except Exception as e:
                self.logger.error(f"Error in speech synthesis: {e}")
                self.logger.info(f"Would have said: {text}")
        else:
            self.logger.warning(f"TTS processor unavailable: {text}")