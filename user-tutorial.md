# Maggie AI Assistant - User Tutorial

This tutorial will help you get started with using the Maggie AI Assistant and guide you through its main features.

## Getting Started

### Starting Maggie

1. Ensure that you've completed the installation process in the [Installation Guide](INSTALLATION.md).

2. Start Maggie from the terminal:
   ```bash
   # Navigate to the Maggie directory
   cd path/to/maggie
   
   # Activate the virtual environment
   # On Windows PowerShell:
   .\venv\Scripts\Activate.ps1
   
   # On Linux/macOS:
   source venv/bin/activate
   
   # Start Maggie
   python main.py
   ```

3. The Maggie GUI window will appear, showing the current state as "IDLE".

### Waking Up Maggie

1. When Maggie is in the "IDLE" state, it's listening only for the wake word.

2. Say the wake word "Maggie" clearly.

3. Maggie will transition to the "STARTUP" state, then to the "READY" state, and will greet you with "Ready for your command".

4. The GUI will show the state change and log the event.

## Basic Commands

When Maggie is in the "READY" state, you can use the following commands:

### System Control Commands

- **"Sleep"** or **"Go to sleep"**: Puts Maggie back into the "IDLE" state, conserving resources.
- **"Shutdown"** or **"Turn off"**: Completely shuts down the Maggie application.

### Utility Commands

- **"New recipe"**: Activates the Recipe Creator utility.

## Using the Recipe Creator Utility

The Recipe Creator helps you create recipe documents using your voice. Here's how to use it:

1. Say "New recipe" when Maggie is in the "READY" state.

2. Maggie will ask for the recipe name. Speak clearly to provide a name for your recipe.

3. Maggie will confirm the name by repeating it back to you and asking if it's correct:
   - Say "Yes" or "Correct" to confirm.
   - Say "No" or "Wrong" to try again.

4. After confirming the name, Maggie will ask you to describe the recipe, including ingredients and steps.

5. Speak your recipe description clearly. For best results:
   - List ingredients with quantities first ("2 cups of flour, 1 teaspoon of salt...")
   - Then describe preparation steps in order ("First, mix the dry ingredients...")
   - Try to speak in complete sentences.
   - You have approximately 30 seconds to describe your recipe.

6. After you finish speaking, Maggie will process your description using the LLM to extract ingredients and steps.

7. Maggie will create a Word document (.docx) with your recipe and save it to the "recipes" folder.

8. Once complete, Maggie will confirm "Recipe [name] has been created and saved" and return to the "READY" state.

### Canceling Recipe Creation

You can cancel the recipe creation process at any time by saying "Cancel". Maggie will stop the process and return to the "READY" state.

## Using the GUI

The Maggie GUI provides several features to help you interact with and monitor the assistant:

### Main Panels

- **Chat Log**: Shows the conversation between you and Maggie.
- **Event Log**: Displays system events like state transitions.
- **Error Log**: Shows any errors that occur during operation.

### Status Indicators

- **Current State Display**: Shows Maggie's current state with a color-coded background.
- **Status Bar**: Displays the current state at the bottom of the window.

### Control Buttons

- **Sleep Button**: Equivalent to saying "Sleep".
- **Shutdown Button**: Equivalent to saying "Shutdown".
- **Utility Buttons**: Direct access to available utilities (e.g., "Recipe Creator").

## Understanding Maggie's States

Maggie operates in seven distinct states:

1. **IDLE**: Minimal resource usage, listening only for wake word.
2. **STARTUP**: Transitional state where resources are being initialized.
3. **READY**: Listening for commands, all functions available.
4. **ACTIVE**: Executing a specific task through a utility module.
5. **BUSY**: Processing intensive operations like LLM inference.
6. **CLEANUP**: Releasing resources and preparing for next state.
7. **SHUTDOWN**: Gracefully terminating the application.

The current state is always displayed in the GUI with a color code:
- IDLE: Light gray
- STARTUP: Light blue
- READY: Light green
- ACTIVE: Yellow
- BUSY: Orange
- CLEANUP: Pink
- SHUTDOWN: Red

## Tips for Best Results

1. **Speaking Clearly**: Speak at a normal pace and volume, and articulate clearly.
2. **Quiet Environment**: Use Maggie in a quiet environment for best speech recognition.
3. **Proper Wake Word Detection**: Say "Maggie" clearly and wait for the response before giving commands.
4. **Command Timing**: Give commands only when Maggie is in the "READY" state.
5. **Recipe Organization**: When describing recipes, first list ingredients, then steps in a logical order.

## Troubleshooting

If Maggie is not responding as expected:

1. **Check the State**: Ensure Maggie is in the appropriate state for your command.
2. **Review Logs**: Check the Event and Error logs for any issues.
3. **Restart if Necessary**: If Maggie becomes unresponsive, use the Shutdown button and restart the application.
4. **Wake Word Sensitivity**: If Maggie doesn't respond to the wake word, try adjusting the sensitivity in the configuration file.

## Next Steps

After mastering the basic functionality, you can:

1. **Customize Configuration**: Edit `config.yaml` to adjust settings like wake word sensitivity and speech processing.
2. **Explore the Code**: Review the source code to understand how Maggie works and how to extend it.
3. **Add New Utilities**: Develop new utility modules following the pattern of the Recipe Creator.
