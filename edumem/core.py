"""
Core functionality for the edumem package.
"""

class Memory:
    """
    A simple memory store that can remember and recall values.
    """
    
    def __init__(self):
        self._store = {}
    
    def remember(self, key, value):
        """
        Store a value with the given key.
        
        Args:
            key: A hashable object to use as the key
            value: The value to store
        """
        self._store[key] = value
        return self
    
    def recall(self, key, default=None):
        """
        Retrieve a value by its key.
        
        Args:
            key: The key to look up
            default: Value to return if key is not found
            
        Returns:
            The stored value or default if key not found
        """
        return self._store.get(key, default)
    
    def forget(self, key):
        """
        Remove a key-value pair from memory.
        
        Args:
            key: The key to remove
            
        Returns:
            True if key was found and removed, False otherwise
        """
        if key in self._store:
            del self._store[key]
            return True
        return False
    
    def clear(self):
        """Clear all stored memories."""
        self._store.clear()
        return self 