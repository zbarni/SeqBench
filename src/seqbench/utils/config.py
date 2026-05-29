# SPDX-License-Identifier: MIT
# Copyright (c) 2025-present, SeqBench Contributors

"""
Configuration module for managing YAML-based configuration files with nested support.
"""

"""
Configuration Module for SeqBench

This module provides a configuration class that wraps YAML configuration files
with a dictionary-like interface and support for nested configurations.
"""

import yaml


class Config:
    """
    Configuration wrapper class for YAML-based configuration files.
    
    This class provides a convenient interface for accessing configuration values
    with support for:
    - Nested configurations (automatically converted to Config objects)
    - Default values using tuple syntax: config['key', default_value]
    - Dictionary-like access patterns
    - Configuration validation
    
    Attributes:
        config: The underlying dictionary containing configuration values
    
    Example:
        >>> config = Config.parse_config_from_path('config.yaml')
        >>> value = config['key']  # Raises ValueError if key doesn't exist
        >>> value = config['key', 'default']  # Returns 'default' if key doesn't exist
        >>> value = config.get('key', 'default')  # Alternative syntax
        >>> nested = config['nested']['subkey']  # Access nested configs
    """

    def __init__(self, config):
        """
        Initialize a Config object from a dictionary.
        
        Args:
            config: Dictionary containing configuration values
        """
        self.config = config

    @staticmethod
    def parse_config_from_args(args):
        """
        Parse configuration from command-line arguments.
        
        This method loads a YAML configuration file specified in args['config']
        and then overwrites any values with those provided in args.
        
        Args:
            args: Dictionary containing 'config' key (path to YAML file) and
                optionally other keys to overwrite config values
        
        Returns:
            Config: Configuration object with merged values
            
        Raises:
            AssertionError: If the config file is empty or None
        """
        with open(args['config'], 'r') as f:    
            config = yaml.safe_load(f)

        assert config is not None, 'Provided config seems to be empty' 

        for k, v in args.items():
            if v is not None:
                if k in config.keys():
                    print(f'CLI config overwrite for "{k}"!')
                config[k] = v

        return Config(config)

    @staticmethod
    def parse_config_from_path(path):
        """
        Parse configuration from a YAML file path.
        
        Args:
            path: Path to the YAML configuration file
        
        Returns:
            Config: Configuration object loaded from the file
            
        Raises:
            AssertionError: If the config file is empty or None
        """
        with open(path, 'r') as f:    
            config = yaml.safe_load(f)
        
        assert config is not None, 'Provided config seems to be empty' 

        return Config(config)

    def asdict(self):
        """
        Convert the configuration to a plain dictionary.
        
        Recursively converts nested Config objects to dictionaries.
        
        Returns:
            dict: Plain dictionary representation of the configuration
        """
        as_dict = {}
        for k, v in self.items():
            if isinstance(v, Config):
                as_dict[k] = v.asdict()
            else:
                as_dict[k] = v
        return as_dict

    def print_config(self):
        """
        Print the configuration in a formatted, aligned manner.
        
        All keys are left-aligned to the length of the longest key name,
        and values are padded to 100 characters for readability.
        """
        max_key_length = max([len(k) for k in self.config.keys()])
        for key in self.config:
            print(key.ljust(max_key_length, '-'), str(self.config[key]).ljust(100, '-'))

    def assert_has_key(self, key):
        """
        Assert that the configuration contains a specific key.
        
        Args:
            key: Key to check for in the configuration
            
        Raises:
            ValueError: If the key is not present in the configuration
        """
        if not self.has_key(key):
            error_msg = f'Config is missing key "{key}".'
            raise ValueError(error_msg)
     
    def has_key(self, key):
        """
        Check if the configuration contains a specific key.
        
        Args:
            key: Key to check for in the configuration
        
        Returns:
            bool: True if the key exists, False otherwise
        """
        return key in self.config.keys()

    def __getitem__(self, key):
        """
        Get a configuration value by key.
        
        Supports two syntaxes:
        1. config['key'] - Raises ValueError if key doesn't exist
        2. config['key', default] - Returns default if key doesn't exist
        
        Args:
            key: Either a string key, or a tuple (key, default_value)
        
        Returns:
            The configuration value, or default if provided and key doesn't exist
            
        Raises:
            ValueError: If key doesn't exist and no default is provided
        """
        if isinstance(key, tuple):
            assert len(key) == 2
            default = key[1]
            key = key[0]
            return self.__get_item_with_default(key, default)
        else:
            return self.__get_item(key)

    def __contains__(self, key):
        """
        Check if a key exists in the configuration.
        
        Allows the use of `'key' in config` syntax without raising errors.
        
        Args:
            key: Key to check for in the configuration
        
        Returns:
            bool: True if the key exists, False otherwise
        """
        try:
            self.__get_item(key)
            return True
        except ValueError:
            return False

    def get(self, key, default=None):
        """
        Get a configuration value with a default fallback.
        
        Args:
            key: Key to retrieve from the configuration
            default: Default value to return if key doesn't exist (default: None)
        
        Returns:
            The configuration value if key exists, otherwise the default value
        """
        try:
            return self.__get_item(key)
        except ValueError:
            return default

    def __get_item(self, key):
        self.assert_has_key(key)
        item = self.__parse_item(self.config[key])
        return item

    def __get_item_with_default(self, key, default):
        if self.has_key(key):    
            item = self.__parse_item(self.config[key])
            return item
        else:
            return self.__parse_item(default)

    def __parse_item(self, item):
        if isinstance(item, dict):
            return Config(item)
        else:
            return item

    def __setitem__(self, key, value):
        """
        Set a configuration value.
        
        Args:
            key: Configuration key
            value: Value to set
        """
        self.config[key] = value
    
    def update(self, dict, prefix=''):
        """
        Update the configuration with values from a dictionary.
        
        Args:
            dict: Dictionary of key-value pairs to update
            prefix: Optional prefix to prepend to all keys (default: '')
        """
        for k, v in dict.items():
            self.config[f'{prefix}{k}'] = v

    def items(self):
        return self.config.items()

    def keys(self):
        return self.config.keys()