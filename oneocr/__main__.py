import argparse
import json
import sys
from pathlib import Path
from .engine import OneOCR

def main():
    parser = argparse.ArgumentParser(
        description="OneOCR CLI: A command-line tool for running the Microsoft Snipping Tool OCR engine locally."
    )
    parser.add_argument("image_path", type=str, nargs="?", help="Path to the input image file.")
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Download, extract, and decrypt models automatically (requires Windows)."
    )
    parser.add_argument(
        "-l", "--lang", "--language",
        type=str,
        default=None,
        help="Target language code (e.g. 'ru', 'en') or script group name (e.g. 'cyrillic', 'latin')."
    )
    parser.add_argument(
        "-r", "--rotation",
        type=int,
        choices=[0, 90, 180, 270],
        default=None,
        help="Fixed image rotation angle (0, 90, 180, 270) to skip auto-orientation."
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Optional path to write the text output."
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1536,
        help="Resize image's longest side to this limit for detection."
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Detector confidence score threshold (default: 0.5)."
    )
    parser.add_argument(
        "--link-threshold",
        type=float,
        default=0.0,
        help="Detector link threshold (default: 0.0)."
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Force enable GPU hardware acceleration."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output OCR details in JSON format (lines, words, boxes, confidences)."
    )
    
    args = parser.parse_args()
    
    if args.prepare:
        from . import prepare
        try:
            prepare()
            sys.exit(0)
        except Exception as e:
            print(f"Preparation error: {e}", file=sys.stderr)
            sys.exit(1)
            
    if not args.image_path:
        parser.print_help()
        sys.exit(1)
        
    image_path = Path(args.image_path)
    if not image_path.exists():
        print(f"Error: Image file not found: {args.image_path}", file=sys.stderr)
        sys.exit(1)
        
    try:
        # Initialize engine
        ocr = OneOCR(
            use_gpu=args.gpu,
            default_language=args.lang,
            default_rotation=args.rotation,
            max_side=args.max_side,
            score_threshold=args.score_threshold,
            link_threshold=args.link_threshold,
        )
    except Exception as e:
        print(f"Initialization error: {e}", file=sys.stderr)
        sys.exit(1)
        
    try:
        # Run OCR
        result = ocr.recognize_file(image_path)
    except Exception as e:
        print(f"OCR execution error: {e}", file=sys.stderr)
        sys.exit(1)
        
    # Output formatting
    if args.json:
        output_data = {
            "image_angle": result.image_angle,
            "full_text": result.full_text,
            "lines": [
                {
                    "text": line.text,
                    "style": line.style,
                    "bbox": line.bbox.as_rect() if line.bbox else None,
                    "quad": [line.bbox.x1, line.bbox.y1, line.bbox.x2, line.bbox.y2, line.bbox.x3, line.bbox.y3, line.bbox.x4, line.bbox.y4] if line.bbox else None,
                    "words": [
                        {
                            "text": word.text,
                            "confidence": word.confidence,
                            "bbox": word.bbox.as_rect() if word.bbox else None,
                            "quad": [word.bbox.x1, word.bbox.y1, word.bbox.x2, word.bbox.y2, word.bbox.x3, word.bbox.y3, word.bbox.x4, word.bbox.y4] if word.bbox else None
                        } for word in line.words
                    ]
                } for line in result.lines
            ]
        }
        out_str = json.dumps(output_data, ensure_ascii=False, indent=2)
    else:
        out_str = result.full_text
        
    # Write to stdout or file
    if args.output:
        try:
            with open(args.output, "w", encoding="utf-8") as f:
                f.write(out_str)
            print(f"OCR output saved successfully to: {args.output}")
        except Exception as e:
            print(f"Error writing to output file: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        if hasattr(sys.stdout, 'buffer'):
            sys.stdout.buffer.write(out_str.encode('utf-8'))
            sys.stdout.write('\n')
        else:
            print(out_str)

if __name__ == "__main__":
    main()
