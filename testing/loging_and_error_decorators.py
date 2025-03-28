#!/usr/bin/env python3
"""
Demo script demonstrating the usage of Maggie AI Assistant's decorators.

This script shows how to use the @log_operation() and @with_error_handling() 
decorators from the Maggie AI Assistant project on an example method that
occasionally raises errors and returns various data types.
"""

import random
import time
import os
from typing import Any, Dict, Union

# Import necessary components from the Maggie AI Assistant project
from .maggie.utils.logging import log_operation, ComponentLogger, LoggingManager
from .maggie.utils.error_handling import with_error_handling, safe_execute, ErrorCategory, ErrorSeverity
from .maggie.utils.abstractions import CapabilityRegistry
from .maggie.utils.adapters import LoggingManagerAdapter, ErrorHandlerAdapter


@log_operation(component='DataProcessor', log_args=True, log_result=True)
@with_error_handling(
    error_category=ErrorCategory.PROCESSING,
    error_severity=ErrorSeverity.WARNING,
    publish_error=True,
    include_state_info=True
)
def process_data(data_id: int, complexity: int = 1) -> Union[Dict[str, Any], int, str, None]:
    """
    Example function that processes data and occasionally raises errors.
    
    Parameters
    ----------
    data_id : int
        ID of the data to process
    complexity : int, optional
        Complexity level of the processing, by default 1
        
    Returns
    -------
    Union[Dict[str, Any], int, str, None]
        Processed data in different formats based on data_id
        
    Raises
    ------
    ValueError
        If a random processing failure occurs (20% chance)
    """
    # Simulate processing time based on complexity
    time.sleep(random.uniform(0.1, 0.3) * complexity)
    
    # Randomly fail with 20% probability
    if random.random() < 0.2:
        raise ValueError(f"Random processing failure for data_id={data_id}")
    
    # Return different types based on data_id modulo 3
    if data_id % 3 == 0:
        # Return a dictionary for data_id divisible by 3
        return {
            "data_id": data_id,
            "result": f"Processed data {data_id}",
            "timestamp": time.time(),
            "complexity": complexity
        }
    elif data_id % 3 == 1:
        # Return an integer for data_id % 3 == 1
        return data_id * complexity
    else:
        # Return a string for data_id % 3 == 2
        return f"String result for data_id={data_id}"


# Alternative method using safe_execute directly to demonstrate default_return
def process_data_alternative(data_id: int, complexity: int = 1) -> Union[Dict[str, Any], int, str, None]:
    """Alternative implementation using safe_execute directly instead of decorator."""
    
    def _process_data_internal(data_id: int, complexity: int) -> Union[Dict[str, Any], int, str]:
        # Same implementation as process_data
        time.sleep(random.uniform(0.1, 0.3) * complexity)
        
        if random.random() < 0.2:
            raise ValueError(f"Random processing failure for data_id={data_id}")
        
        if data_id % 3 == 0:
            return {
                "data_id": data_id,
                "result": f"Processed data {data_id}",
                "timestamp": time.time(),
                "complexity": complexity
            }
        elif data_id % 3 == 1:
            return data_id * complexity
        else:
            return f"String result for data_id={data_id}"
    
    # Use safe_execute with default_return parameter
    return safe_execute(
        _process_data_internal,
        data_id,
        complexity,
        error_category=ErrorCategory.PROCESSING,
        error_severity=ErrorSeverity.WARNING,
        default_return=None
    )


def main():
    """Main function to demo the decorated function."""
    # Ensure logs directory exists
    os.makedirs('logs', exist_ok=True)
    
    # Initialize logging manager with basic configuration
    config = {
        'logging': {
            'path': 'logs',
            'console_level': 'INFO',
            'file_level': 'DEBUG'
        }
    }
    
    # Initialize the logging system
    logging_mgr = LoggingManager.initialize(config)
    
    # Register necessary adapters with the CapabilityRegistry
    registry = CapabilityRegistry.get_instance()
    logging_adapter = LoggingManagerAdapter(logging_mgr)
    error_handler = ErrorHandlerAdapter()
    
    # Create a logger for the main script
    logger = ComponentLogger('DecoratorDemo')
    logger.info("Starting decorator demonstration script")
    
    # Process multiple data items using the decorated function
    logger.info("=== Testing with decorator pattern ===")
    for i in range(5):
        # Randomly vary the complexity
        complexity = random.randint(1, 3)
        
        # Process the data (errors will be handled by the decorator)
        result = process_data(i, complexity)
        
        # Log the result
        logger.info(f"Processed data {i} with complexity {complexity}: {result}")
    
    # Demonstrate alternative approach with direct safe_execute
    logger.info("=== Testing with direct safe_execute ===")
    for i in range(5, 10):
        complexity = random.randint(1, 3)
        result = process_data_alternative(i, complexity)
        logger.info(f"Processed data {i} with complexity {complexity}: {result}")
    
    logger.info("Decorator demonstration completed")


if __name__ == "__main__":
    main()