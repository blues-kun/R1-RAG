#!/usr/bin/env python3
"""
Data Preparation Script for R1-RAG

This script:
1. Loads multi-hop QA datasets (HotpotQA, 2WikiMultihopQA, etc.)
2. Generates golden plan annotations using GPT-4o
3. Filters annotations against ground truth
4. Exports training data in parquet format

Usage:
    python scripts/prepare_data.py --dataset hotpotqa --api_key YOUR_KEY
"""

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from r1_rag.data import PlanningDataProcessor, GPT4oPlanGenerator


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare R1-RAG training data")
    
    parser.add_argument(
        "--dataset",
        type=str,
        default="hotpotqa",
        choices=["hotpotqa", "2wiki", "musique", "custom"],
        help="Dataset to process"
    )
    parser.add_argument(
        "--input_file",
        type=str,
        default=None,
        help="Path to custom JSONL input file"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/r1_rag",
        help="Output directory for processed data"
    )
    parser.add_argument(
        "--api_key",
        type=str,
        default=None,
        help="OpenAI API key for GPT-4o annotation (optional)"
    )
    parser.add_argument(
        "--generate_annotations",
        action="store_true",
        help="Generate new annotations using GPT-4o"
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum samples to process (for testing)"
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=0.9,
        help="Ratio of data for training"
    )
    
    return parser.parse_args()


def load_dataset(dataset_name: str, input_file: str = None) -> PlanningDataProcessor:
    """Load dataset based on name or custom file."""
    
    processor = PlanningDataProcessor(data_source=dataset_name)
    
    if input_file:
        # Load from custom file
        processor.load_from_jsonl(input_file)
    else:
        # Load from HuggingFace
        if dataset_name == "hotpotqa":
            processor.load_from_huggingface(
                "hotpot_qa",
                split="train",
                question_key="question",
                answer_key="answer"
            )
        elif dataset_name == "2wiki":
            processor.load_from_huggingface(
                "THUDM/2WikiMultihopQA",
                split="train"
            )
        elif dataset_name == "musique":
            processor.load_from_huggingface(
                "drt/musique",
                split="train"
            )
        else:
            raise ValueError(f"Unknown dataset: {dataset_name}")
    
    return processor


def main():
    args = parse_args()
    
    print("=" * 60)
    print("R1-RAG Data Preparation")
    print("=" * 60)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Load dataset
    print(f"\n[Step 1] Loading dataset: {args.dataset}")
    processor = load_dataset(args.dataset, args.input_file)
    
    # Print statistics
    stats = processor.get_statistics()
    print(f"\nDataset statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  With plan: {stats['with_plan']}")
    print(f"  With graph: {stats['with_graph']}")
    
    # Generate annotations if requested
    if args.generate_annotations:
        if not args.api_key:
            print("\n[Error] API key required for annotation generation")
            print("Use --api_key YOUR_OPENAI_KEY or set OPENAI_API_KEY env var")
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                sys.exit(1)
        else:
            api_key = args.api_key
        
        print(f"\n[Step 2] Generating GPT-4o annotations...")
        processor.generate_annotations(
            api_key=api_key,
            max_samples=args.max_samples
        )
        
        # Updated statistics
        stats = processor.get_statistics()
        print(f"\nAfter annotation:")
        print(f"  With plan: {stats['with_plan']}")
        print(f"  With graph: {stats['with_graph']}")
    
    # Filter samples with annotations
    print(f"\n[Step 3] Filtering samples with annotations...")
    processor.filter_with_annotations()
    
    # Export data
    print(f"\n[Step 4] Exporting to {args.output_dir}")
    paths = processor.export_parquet(
        args.output_dir,
        train_ratio=args.train_ratio
    )
    
    # Also export JSONL for inspection
    processor.export_jsonl(os.path.join(args.output_dir, "data.jsonl"))
    
    print("\n" + "=" * 60)
    print("Data preparation complete!")
    print(f"  Train: {paths['train']}")
    print(f"  Val: {paths['val']}")
    print("=" * 60)


if __name__ == "__main__":
    main()

