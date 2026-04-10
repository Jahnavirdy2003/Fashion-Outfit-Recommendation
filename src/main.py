"""
Entry point for the Fashion Outfit Recommendation system.

Usage:
  python src/main.py train
  python src/main.py evaluate
  python src/main.py recommend --text "blue denim jacket"
"""
import argparse
import sys


def run_train():
    from train import train
    train()


def run_evaluate():
    from evaluate import evaluate
    evaluate()


def run_recommend(text, top_k):
    from pathlib import Path
    from recommend import recommend, load_model, build_catalog

    EMB_CACHE = "models/catalog_embeddings.pt"

    # Build catalog if it doesn't exist yet
    if not Path(EMB_CACHE).exists():
        print("Catalog not found — building it first...")
        model = load_model()
        build_catalog(model, max_items=5000)

    # Use a sample image from the dataset as query
    from datasets import load_from_disk
    ds  = load_from_disk("data/polyvore_outfits")["data"]
    row = ds[0]
    img = row["image"].convert("RGB")
    img.save("data/sample_query.jpg")

    query_text = text if text else f"{row['category']}. {row['text']}"
    recommend("data/sample_query.jpg", query_text, top_k=top_k)


def main():
    parser = argparse.ArgumentParser(
        description="Fashion Outfit Recommendation System"
    )
    subparsers = parser.add_subparsers(dest="command")

    # Train
    subparsers.add_parser("train", help="Train the model")

    # Evaluate
    subparsers.add_parser("evaluate", help="Evaluate AUC + Accuracy")

    # Recommend
    rec = subparsers.add_parser("recommend", help="Get outfit recommendations")
    rec.add_argument("--text", type=str, default="",
                     help="Text description of query item")
    rec.add_argument("--topk", type=int, default=5,
                     help="Number of recommendations to return")

    args = parser.parse_args()

    if args.command == "train":
        run_train()
    elif args.command == "evaluate":
        run_evaluate()
    elif args.command == "recommend":
        run_recommend(args.text, args.topk)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()