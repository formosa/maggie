from typing import Optional, Dict, Any
from maggie.utils.abstractions import (
    get_logger_provider,
    get_error_handler,
    get_event_publisher,
    get_state_provider
)
from maggie.utils.logging import log_operation
from maggie.utils.error_handling import with_error_handling, ErrorCategory, ErrorSeverity


class RobustComponent:
    """
    Template for a robust component using the abstraction layer properly.
    
    Demonstrates appropriate capability retrieval, null checking,
    and graceful degradation when capabilities are unavailable.
    """
    
    def __init__(self):
        # Retrieve capabilities through abstractions
        self.logger = get_logger_provider()
        self.error_handler = get_error_handler()
        self.event_publisher = get_event_publisher()
        self.state_provider = get_state_provider()
        
        # Log initialization with null check
        if self.logger:
            self.logger.info("RobustComponent initialized")
        else:
            print("WARNING: Logger not available, using fallback logging")
    
    @log_operation(component="RobustComponent")
    @with_error_handling(error_category=ErrorCategory.PROCESSING, error_severity=ErrorSeverity.ERROR)
    def process_data(self, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Process data with proper capability handling and error management."""
        # Check current state if available
        current_state = None
        if self.state_provider:
            current_state = self.state_provider.get_current_state()
            
            # Log with null check
            if self.logger:
                self.logger.info(f"Processing data in state: {current_state.name}")
        else:
            # Fallback logging when state provider is unavailable
            if self.logger:
                self.logger.warning("State provider unavailable, proceeding without state validation")
                
        # Process with safe execution if error handler is available
        result = None
        if self.error_handler:
            result = self.error_handler.safe_execute(
                self._internal_processing,
                data,
                error_code="DATA_PROCESSING_FAILED",
                default_return=None
            )
        else:
            # Fallback processing without error handler
            try:
                result = self._internal_processing(data)
            except Exception as e:
                if self.logger:
                    self.logger.error(f"Error processing data: {e}")
                return None
                
        # Publish event if publisher is available
        if self.event_publisher and result:
            self.event_publisher.publish('data_processed', {
                'success': result is not None,
                'data_size': len(data) if data else 0
            })
            
        return result
    
    def _internal_processing(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Internal processing logic that might raise exceptions."""
        # Processing implementation
        return {"processed": True, "original_data": data}

if __name__ == "__main__":

    if result := RobustComponent().process_data({"key_1": "value_1", "key_2": "value_2", "key_3": "value_3"}):
        print(f"Type:     {type(result)}")
        print(f"Length:   {len(result)}")
        print(f"Content:  {result}")
    else:
        print("Processing failed or returned None")
    # This is an example of how to use the RobustComponent class.
