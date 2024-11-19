import json
import sys
import pandas as pd
from collections import Counter
from typing import Dict, List, Any
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
import numpy as np
class DatasetValidator:
    def __init__(self, dataset_path: str):
        self.dataset_path = dataset_path
        self.data = None
        self.stats = {}
        
    def load_dataset(self) -> bool:
        """Load and perform initial validation of the dataset."""
        try:
            with open(self.dataset_path, 'r', encoding='utf-8') as f:
                self.data = json.load(f)
            print(f"✅ Successfully loaded dataset from {self.dataset_path}")
            return True
        except json.JSONDecodeError as e:
            print(f"❌ Invalid JSON format: {e}")
            return False
        except Exception as e:
            print(f"❌ Failed to load dataset: {e}")
            return False

    def validate_structure(self) -> Dict[str, Any]:
        """Validate the basic structure of the dataset."""
        structure_stats = {
            "total_dialogues": 0,
            "total_turns": 0,
            "dialogues_with_missing_keys": 0,
            "turns_with_missing_keys": 0,
            "unique_services": set(),
            "speaker_distribution": Counter(),
        }

        if not isinstance(self.data, list):
            print("❌ Dataset is not a list of dialogues")
            return structure_stats

        required_dialogue_keys = {'dialogue_id', 'services', 'turns'}
        required_turn_keys = {'turn_id', 'speaker', 'utterance', 'frames', 'dialogue_acts'}

        structure_stats["total_dialogues"] = len(self.data)

        for dialogue in self.data:
            # Check dialogue structure
            missing_dialogue_keys = required_dialogue_keys - set(dialogue.keys())
            if missing_dialogue_keys:
                structure_stats["dialogues_with_missing_keys"] += 1
                print(f"⚠️ Dialogue missing keys: {missing_dialogue_keys}")

            # Track services
            services = dialogue.get('services', [])
            structure_stats["unique_services"].update(services)

            # Check turns
            turns = dialogue.get('turns', [])
            structure_stats["total_turns"] += len(turns)

            for turn in turns:
                missing_turn_keys = required_turn_keys - set(turn.keys())
                if missing_turn_keys:
                    structure_stats["turns_with_missing_keys"] += 1

                speaker = turn.get('speaker', 'UNKNOWN')
                structure_stats["speaker_distribution"][speaker] += 1

        return structure_stats

    def analyze_content(self) -> Dict[str, Any]:
        """Analyze the content of the dataset."""
        content_stats = {
            "utterance_lengths": [],
            "frames_per_turn": [],
            "dialogue_acts_per_turn": [],
            "turns_per_dialogue": [],
        }

        for dialogue in self.data:
            turns = dialogue.get('turns', [])
            content_stats["turns_per_dialogue"].append(len(turns))

            for turn in turns:
                utterance = turn.get('utterance', '')
                frames = turn.get('frames', [])
                dialogue_acts = turn.get('dialogue_acts', {})

                content_stats["utterance_lengths"].append(len(utterance.split()))
                content_stats["frames_per_turn"].append(len(frames))
                content_stats["dialogue_acts_per_turn"].append(len(dialogue_acts))

        return content_stats

    def generate_plots(self, content_stats: Dict[str, Any], output_dir: str = "validation_plots"):
        """Generate visualization plots for the dataset statistics."""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Set style
        plt.style.use('seaborn')
        
        # Plot distributions
        metrics = {
            "utterance_lengths": "Words per Utterance",
            "frames_per_turn": "Frames per Turn",
            "dialogue_acts_per_turn": "Dialogue Acts per Turn",
            "turns_per_dialogue": "Turns per Dialogue"
        }

        for metric, title in metrics.items():
            plt.figure(figsize=(10, 6))
            sns.histplot(content_stats[metric], kde=True)
            plt.title(f"Distribution of {title}")
            plt.xlabel(title)import numpy as np
            plt.close()

    def print_summary(self, structure_stats: Dict[str, Any], content_stats: Dict[str, Any]):
        """Print a summary of the dataset statistics."""
        print("\n=== Dataset Summary ===")
        print(f"Total Dialogues: {structure_stats['total_dialogues']}")
        print(f"Total Turns: {structure_stats['total_turns']}")
        print(f"Unique Services: {len(structure_stats['unique_services'])}")
        print(f"Services: {', '.join(structure_stats['unique_services'])}")
        
        print("\n=== Quality Metrics ===")
        print(f"Dialogues with Missing Keys: {structure_stats['dialogues_with_missing_keys']}")
        print(f"Turns with Missing Keys: {structure_stats['turns_with_missing_keys']}")
        
        print("\n=== Content Statistics ===")
        print(f"Average Words per Utterance: {sum(content_stats['utterance_lengths']) / len(content_stats['utterance_lengths']):.2f}")
        print(f"Average Turns per Dialogue: {sum(content_stats['turns_per_dialogue']) / len(content_stats['turns_per_dialogue']):.2f}")
        
        print("\n=== Speaker Distribution ===")
        for speaker, count in structure_stats['speaker_distribution'].items():
            print(f"{speaker}: {count} turns ({count/structure_stats['total_turns']*100:.1f}%)")

def main():
    if len(sys.argv) != 2:
        print("Usage: python validate_dataset.py <path_to_dataset.json>")
        sys.exit(1)

    dataset_path = sys.argv[1]
    validator = DatasetValidator(dataset_path)

    if not validator.load_dataset():
        sys.exit(1)

    # Perform validations
    structure_stats = validator.validate_structure()
    content_stats = validator.analyze_content()

    # Generate visualizations
    validator.generate_plots(content_stats)

    # Print summary
    validator.print_summary(structure_stats, content_stats)

if __name__ == "__main__":
    main()