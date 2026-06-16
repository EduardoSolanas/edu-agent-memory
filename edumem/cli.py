"""
Command-line interface for edumem.
"""
import argparse
import sys
import json
import os.path
from edumem.core import Memory


def main():
    """Main entry point for the CLI."""
    parser = argparse.ArgumentParser(description="edumem - A memory utility")
    subparsers = parser.add_subparsers(dest="command", help="Command to execute")
    
    # Remember command
    remember_parser = subparsers.add_parser("remember", help="Store a value")
    remember_parser.add_argument("key", help="Key to store")
    remember_parser.add_argument("value", help="Value to store")
    remember_parser.add_argument("--file", "-f", help="JSON file to store memory")
    
    # Recall command
    recall_parser = subparsers.add_parser("recall", help="Retrieve a value")
    recall_parser.add_argument("key", help="Key to recall")
    recall_parser.add_argument("--default", "-d", help="Default value if key not found")
    recall_parser.add_argument("--file", "-f", help="JSON file to load memory from")
    
    # Forget command
    forget_parser = subparsers.add_parser("forget", help="Remove a value")
    forget_parser.add_argument("key", help="Key to forget")
    forget_parser.add_argument("--file", "-f", help="JSON file to modify")
    
    # Parse arguments
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Initialize memory
    memory = Memory()
    
    # Load from file if specified
    if hasattr(args, "file") and args.file and os.path.exists(args.file):
        try:
            with open(args.file, "r", encoding="utf-8") as f:
                data = json.load(f)
                for k, v in data.items():
                    memory.remember(k, v)
        except json.JSONDecodeError:
            print(f"Error: {args.file} is not a valid JSON file")
            sys.exit(1)
    
    # Process command
    if args.command == "remember":
        # Try to parse value as JSON
        try:
            value = json.loads(args.value)
        except json.JSONDecodeError:
            value = args.value
        
        memory.remember(args.key, value)
        
        # Save to file if specified
        if args.file:
            data = {k: memory.recall(k) for k in memory._store}
            with open(args.file, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            
    elif args.command == "recall":
        value = memory.recall(args.key, args.default)
        if value is not None:
            if isinstance(value, (dict, list)):
                print(json.dumps(value, indent=2))
            else:
                print(value)
        else:
            print(f"Key '{args.key}' not found")
            sys.exit(1)
            
    elif args.command == "forget":
        if memory.forget(args.key):
            print(f"Forgot '{args.key}'")
            # Save to file if specified
            if args.file:
                data = {k: memory.recall(k) for k in memory._store}
                with open(args.file, "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2)
        else:
            print(f"Key '{args.key}' not found")
            sys.exit(1)


if __name__ == "__main__":
    main() 