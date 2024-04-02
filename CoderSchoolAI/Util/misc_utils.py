from typing import List, Tuple, Dict, Any, Optional, Union, Callable


def register_function(instance: Any, method_name: str, new_method: Callable):
    """
    This function is used to register a new method on an instance.
    instance: The instance on which the method is to be registered on.
    method_name: The name of the method to be registered.
    new_method: The method to be registered for the provided function name.
    """
    setattr(instance, method_name, new_method.__get__(instance, type(instance)))
